#!/usr/bin/env python3

import argparse
import os
import sys
import pandas
import json

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist
import matplotlib.pylab as plt
import seaborn as sns

here = os.path.dirname(os.path.abspath(__file__))
analysis_root = os.path.dirname(here)
root = os.path.dirname(analysis_root)
sys.path.insert(0, analysis_root)

import performance_study as ps

sns.set_theme(style="whitegrid", palette="muted")


# Keep final json structure of compats so we can make a web interface
compats = {
  # Let's do this ABC so it's easier to lookup
  "instance_names": set(),
  "instances": {},
}

arches_lookup = {
    "c6a.16xlarge": "AMD EPYC 7R13 x86_64",
    "c6i.16xlarge": "Intel Xeon 8375C (Ice Lake)",
    "c6id.12xlarge": "Intel Xeon 8375C (Ice Lake)",  # 'i' for Intel, 'd' for local NVMe
    "c6in.12xlarge": "Intel Xeon 8375C (Ice Lake)",  # 'i' for Intel, 'n' for network
    "c7g.16xlarge": "AWS Graviton3 ARM",
    "d3.4xlarge": "Intel Xeon Platinum 8259 (Cascade Lake)",     # Storage-optimized, typically Intel
    "hpc6a.48xlarge": "AMD EPYC 7R13 x86_64",
    "hpc7g.16xlarge": "AWS Graviton3 ARM",
    "inf2.8xlarge": "AMD EPYC 7R13 x86_64",
    "m6a.12xlarge": "AMD EPYC 7R13 x86_64",
    "m6i.12xlarge": "Intel Xeon 8375C (Ice Lake)",
    "t3.2xlarge": "Intel Skylake E5 2686 v5",
    "t3a.2xlarge": "AMD EPYC 7571 x86_64", 
    "m6g.12xlarge": "AWS Graviton2 ARM",
    "c7a.12xlarge": "AMD EPYC 9R14 x86_64",
    "i4i.8xlarge": "Intel Xeon 8375C (Ice Lake)",
    "m6id.12xlarge": "Intel Xeon 8375C (Ice Lake)",
    "r6a.12xlarge": "AMD EPYC 7R13 X86_64",
    "m7g.16xlarge": "AWS Graviton3 ARM",
}

def get_parser():
    parser = argparse.ArgumentParser(
        description="Run analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--root",
        help="root directory with experiments",
        default=os.path.join(root, "experiment"),
    )
    parser.add_argument(
        "--out",
        help="directory to save parsed results",
        default=os.path.join(here, "data"),
    )
    return parser


def main():
    """
    Find application result files to parse.
    """
    global compats
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Output images and data
    outdir = os.path.abspath(args.out)
    indir = os.path.abspath(args.root)

    # We absolutely want on premises results here
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Find input files (skip anything with test)
    files = ps.find_inputs(indir, "hpcg.out")

    # Saves raw data to file
    df = parse_data(indir, outdir, files)
    plot_results(df, outdir)
    
    # Save final interface for compats
    compats['instance_names'] = list(compats['instance_names'])
    ps.write_json(compats, os.path.join(outdir, 'compatibility_data.json'))

def get_ordered_matrix_and_labels(df_matrix):
    """
    We need to get clustering and labels without clustermap.
    """
    row_linkage = linkage(pdist(df_matrix.values, metric='jaccard'), method='ward')
    row_order_indices = dendrogram(row_linkage, no_plot=True)['leaves']
    ordered_rows = df_matrix.index[row_order_indices].tolist()
    col_linkage = linkage(pdist(df_matrix.T.values, metric='jaccard'), method='ward')
    col_order_indices = dendrogram(col_linkage, no_plot=True)['leaves']
    ordered_cols = df_matrix.columns[col_order_indices].tolist()
    df_ordered = df_matrix.loc[ordered_rows, ordered_cols]
    return df_ordered.values.tolist(), ordered_rows, ordered_cols

def add_hpcg_result(p, indir, filename, ebpf=None, gpu=False):
    """
    Add a new hpcg result
    """
    exp = ps.ExperimentNameParser(filename, indir)

    # Sanity check the files we found
    print(filename)
    env_name = filename.split(os.sep)[-2]
    instance = filename.split(os.sep)[-3]
    env_name = env_name.replace('-arm', '')
    exp.show()

    # Use the env field for the instance type.
    p.set_context(exp.cloud, instance, exp.env_type, exp.size)

    item = ps.read_file(filename)
    if "exited with exit code 132" in item:
        print(f"Study {filename} was not compatible")
        p.add_result("compatible", False, env_name)
        return p
    p.add_result("compatible", True, env_name)
    
    # Get the benchmark total times and FOMs (each of these is an iteration, should be 3)
    lines = item.split('\n')
    times = [float(x.split('=')[-1]) for x in lines if "Benchmark Time Summary::Total" in x]
    foms = [float(x.split("=")[-1]) for x in lines if "Final Summary::HPCG result" in x]
    for i, duration in enumerate(times):
        p.add_result("fom", foms[i], env_name)
        p.add_result("duration", duration, env_name)
    return p


def parse_data(indir, outdir, files):
    """
    Parse filepaths for environment, etc., and results files for data.
    """
    p = ps.ProblemSizeParser("hpcg")

    # It's important to just parse raw data once, and then use intermediate
    for filename in files:
        if "test" in filename:
            continue
        p = add_hpcg_result(p, indir, filename)

    # Save stuff to file first
    p.df.to_csv(os.path.join(outdir, "hpcg-results.csv"))
    return p.df


def plot_results(df, outdir):
    """
    Plot analysis results
    """
    img_outdir = os.path.join(outdir, "img")
    for path in ["fom", "duration", "compatibility"]:
        path = os.path.join(img_outdir, path)
        if not os.path.exists(path):
            os.makedirs(path)

    df["optimization"] = [x.split("-")[-1] for x in df.problem_size.tolist()]

    frames = {}
    # Make a plot for seconds runtime, and each FOM set.
    # We can look at the metric across sizes, colored by experiment
    for metric in df.metric.unique():
        metric_df = df[df.metric == metric]
        frames[metric] = {}
        for instance in metric_df.env.unique():
            print(instance)
            instance_df = metric_df[metric_df.env == instance]
            frames[metric][instance] = instance_df

    print(metric_df.problem_size.unique())
    for metric, instances in frames.items():
        for instance, data_frame in instances.items():
            if metric == "compatible":
                plot_compatible(data_frame, instance, os.path.join(img_outdir, "compatibility"))
                continue
            hue = "optimization"
            fig = plt.figure(figsize=(22, 5))
            axes = []
            gs = plt.GridSpec(1, 2, width_ratios=[7, 1])
            axes.append(fig.add_subplot(gs[0, 0]))
            axes.append(fig.add_subplot(gs[0, 1]))

            data_sorted = data_frame.sort_values(by="value", ascending=True)
            order = (
                data_sorted.groupby("problem_size")["value"]
                .mean()
                .sort_values(ascending=True)
                .index
            )

            sns.set_style("whitegrid")
            func = sns.barplot
            func(
                data_sorted,
                ax=axes[0],
                x="problem_size",
                y="value",
                hue=hue,
                err_kws={"color": "darkred"},
                order=order,
                palette=ps.colors,
            )
            if metric in ["duration"]:
                axes[0].set_title(
                    f"HPCG (xhpcg) {metric.capitalize()} for {instance}", fontsize=14
                )
                axes[0].set_ylabel("Seconds", fontsize=14)
            elif "fom" in metric:
                axes[0].set_title(f"HPCG (xhpcg) Gflop/s for {instance}", fontsize=14)
                axes[0].set_ylabel("Gflop/second", fontsize=14)
            axes[0].set_xlabel("Micro-architecture and Optimization", fontsize=14)

            axes[0].tick_params(axis="x", rotation=90)

            handles, labels = axes[0].get_legend_handles_labels()
            labels = ["/".join(x.split("/")[0:2]) for x in labels]
            axes[1].legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(-0.1, 0.5),
                frameon=False,
            )
            for ax in axes[0:1]:
                ax.get_legend().remove()
            axes[1].axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(img_outdir, metric, f"xhpcg-{instance}-{metric}.svg"))
            plt.clf()

            # Print the total number of data points
            print(f"Total number of datum: {data_frame.shape[0]}")


def plot_compatible(df, instance, img_outdir):
    """
    Plot compatibility matrix
    """
    global compats
    optims = list(set([x.split("-")[-1] for x in df.problem_size.unique().tolist()]))
    microarches = list(
        set([x.rsplit("-", 1)[0] for x in df.problem_size.unique().tolist()])
    )
    microarches.sort()
    optims.sort()
    compat_df = pandas.DataFrame(0, columns=microarches, index=optims)
    compats['instance_names'].add(instance)

    # Fill it in!
    for idx, row in df.iterrows():
        if row.value is True:
            arch, optim = row.problem_size.rsplit("-", 1)
            compat_df.loc[optim, arch] = 1

    ordered_df, rows, cols = get_ordered_matrix_and_labels(compat_df)

    # Add to global set
    compats['instances'][instance] = {
        "heatmap": ordered_df,
        "optimizations": rows,
        "microarches": cols,
        "platform": arches_lookup[instance],
    }    
    fig = plt.figure(figsize=(12, 8))
    axes = []
    gs = plt.GridSpec(1, 2, width_ratios=[7, 0])
    axes.append(fig.add_subplot(gs[0, 0]))
    sns.set_style("whitegrid")
    g = sns.clustermap(
        compat_df,
        cmap="Blues",
        cbar_pos=None,
        annot=True,
    )
    g.fig.suptitle(f"HPCG (xhpcg) Compatibility for {instance}")
    plt.xlabel("Micro-architecture")
    plt.ylabel("Optimization")
    plt.tight_layout()
    plt.savefig(os.path.join(img_outdir, f"xhpcg-optimization-{instance}.svg"))
    plt.clf()


if __name__ == "__main__":
    main()
