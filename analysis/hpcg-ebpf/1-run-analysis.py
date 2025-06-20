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

    # Don't include within-instance run
    files = [x for x in files if 'within-instance' not in x]

    # Create outdirs for images - stay organized!
    img_outdir = os.path.join(outdir, "img")
    for path in ["fom", "duration", "compatibility", "heatmap"]:
        path = os.path.join(img_outdir, path)
        if not os.path.exists(path):
            os.makedirs(path)

    # This generates more detailed json for interface
    parse_metrics(indir, outdir, files)

    # Saves raw data to file
    df = parse_data(indir, outdir, files)
    plot_results(df, outdir)

    # Save final interface for compats
    compats["instance_names"] = list(compats["instance_names"])
    ps.write_json(compats, os.path.join(outdir, "compatibility_data.json"))


def get_ordered_matrix_and_labels(df_matrix):
    """
    We need to get clustering and labels without clustermap.
    """
    row_linkage = linkage(pdist(df_matrix.values, metric="jaccard"), method="ward")
    row_order_indices = dendrogram(row_linkage, no_plot=True)["leaves"]
    ordered_rows = df_matrix.index[row_order_indices].tolist()
    col_linkage = linkage(pdist(df_matrix.T.values, metric="jaccard"), method="ward")
    col_order_indices = dendrogram(col_linkage, no_plot=True)["leaves"]
    ordered_cols = df_matrix.columns[col_order_indices].tolist()
    df_ordered = df_matrix.loc[ordered_rows, ordered_cols]
    return df_ordered.values.tolist(), ordered_rows, ordered_cols


def add_hpcg_result(p, indir, filename, ebpf=None, gpu=False, metrics=None):
    """
    Add a new hpcg result
    """
    exp = ps.ExperimentNameParser(filename, indir)

    # Sanity check the files we found
    env_name = filename.split(os.sep)[-2]
    instance = filename.split(os.sep)[-3]
    env_name = env_name.replace("-arm", "")

    # Use the env field for the instance type.
    p.set_context(exp.cloud, instance, exp.env_type, exp.size)

    # We don't care about iterations here
    item = ps.read_file(filename)
    if "exited with exit code 132" in item:
        p.add_result("compatible", False, env_name, filename=filename)
        return p
    p.add_result("compatible", True, env_name, filename=filename)

    # Get the benchmark total times and FOMs (each of these is an iteration, should be 3)
    lines = item.split("\n")
    if metrics is None:
        metrics = {
            "fom": "Final Summary::HPCG result",
            "duration": "Benchmark Time Summary::Total",
        }

    # Get all metrics, actual lists of
    others = {}
    for key, prefix in metrics.items():
        values = [float(x.rsplit("=", 1)[-1]) for x in lines if prefix in x]
        others[key] = values

    for key, values in others.items():
        # The ordering is consistent between lists
        for iteration, value in enumerate(values):
            p.add_result(key, value, env_name, filename=filename, iteration=iteration)
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


def parse_metrics(indir, outdir, files):
    """
    This goes through one metric at a time, and yields the data frame to export.
    
    These will be many of our Y values.
    """
    # Get the benchmark total times and FOMs (each of these is an iteration, should be 3)
    metrics = {
        # This is for division
        "total_cg_iterations": "Iteration Count Information::Total number of optimized iterations",
        "fom": "Final Summary::HPCG result",
        "duration": "Benchmark Time Summary::Total",
        "mpi_allreduce_min": "DDOT Timing Variations::Min DDOT MPI_Allreduce",
        "mpi_allreduce_max": "DDOT Timing Variations::Max DDOT MPI_Allreduce",
        "mpi_allreduce_avg": "DDOT Timing Variations::Avg DDOT MPI_Allreduce",
        "processes": "Machine Summary::Distributed Processes",
        "threads_per_process": "Machine Summary::Threads per processes",
        "dimension_nx": "Global Problem Dimensions::Global nx",
        "dimension_ny": "Global Problem Dimensions::Global ny",
        "dimension_nz": "Global Problem Dimensions::Global nz",
        "processor_dimension_x": "Processor Dimensions::npx",
        "processor_dimension_y": "Processor Dimensions::npy",
        "processor_dimension_z": "Processor Dimensions::npz",
        # The descriptions below are from asking an LLM. The code here is NOT generated by an LLM, it is written by me.
        # I asked an LLM because I work on my own and want to understand the results better.
        # Sparse Matrix-Vector Multiply is often memory bandwidth bound.
        # Differences here can point to how well different compilers/architectures utilize memory bandwidth for this crucial operation.
        "gflops_per_second_spmv": "GFLOP/s Summary::Raw SpMV",
        # MG is multigrid solver -  it involves a mix of computation (SpMV, smoothers like WAXPBY) and communication (especially in distributed memory).
        # This metric reflects the efficiency of the most complex part of HPCG.
        "gflops_per_second_mg": "GFLOP/s Summary::Raw MG",
        # Dot products involve computation and a global reduction (MPI_Allreduce).
        # While the GFLOP/s might be low due to the reduction overhead, variations can indicate differences in floating-point
        # performance for simple loops or the efficiency of the reduction.
        "gflops_per_second_ddot": "GFLOP/s Summary::Raw DDOT",
        # Vector updates (Y = a*X + Y or similar) are typically memory bandwidth bound but very simple computationally.
        # High values here indicate good memory streaming performance.
        "gflops_per_second_waxpby": "GFLOP/s Summary::Raw WAXPBY",
        # Achieved memory bandwidth across the key kernels, total, read and write
        "memory_bandwidth_across_kernels_total": "GB/s Summary::Raw Total B/W",
        "memory_bandwidth_across_kernels_read": "GB/s Summary::Raw Read B/W",
        "memory_bandwidth_across_kernels_write": "GB/s Summary::Raw Write B/W",
        # Should be fairly constant for a fixed problem size, but slight variations might occur due to compiler optimizations, etc.
        "memory_used_data_total_gbytes": "Memory Use Information::Total memory used for data (Gbytes)",
        # Time taken to initialize data structures.
        # Significant variations might point to differences in how compilers/libraries handle memory allocation and initial data placement,
        # though for long runs this is less critical.
        "setup_time_seconds": "Setup Information::Setup Time",
    }

    # IMPORTANT - this is essentially a weak scaling study, so we need to normalize to make results comparable
    divide_by_n = [
        "gflops_per_second_spmv",
        "gflops_per_second_mg",
        "gflops_per_second_ddot",
        "gflops_per_second_waxpby",
        "fom",
        "memory_bandwidth_across_kernels_total",
        "memory_bandwidth_across_kernels_read",
        "memory_bandwidth_across_kernels_write",
        "memory_used_data_total_gbytes",
        "setup_time_seconds",
    ]
    divide_by_iterations = ["duration"]
    raw_values = [
        "mpi_allreduce_min",
        "mpi_allreduce_max",
        "mpi_allreduce_avg",
        "processes",
        "threads_per_process",
        "dimension_nx",
        "dimension_ny",
        "dimension_nz",
        "processor_dimension_x",
        "processor_dimension_y",
        "processor_dimension_z",
        "total_cg_iterations",
    ]

    img_outdir = os.path.join(outdir, "img", "heatmap")
    data_outdir = os.path.join(outdir, "heatmap")
    for path in ["json", "csv"]:
        path = os.path.join(data_outdir, path)
        if not os.path.exists(path):
            os.makedirs(path)

    # Do one at a time since data frames will take up memory
    total_cg_iterations = None
    for metric, prefix in metrics.items():
        p = ps.ProblemSizeParser("hpcg")
        for filename in files:
            if "test" in filename:
                continue
            p = add_hpcg_result(p, indir, filename, metrics={metric: prefix})
        print(metric + " " + prefix)
        # We save this for later and can plot it.
        if metric == "total_cg_iterations":
            total_cg_iterations = p.df
        p.df.to_csv(os.path.join(data_outdir, "csv", f"hpcg_{metric}.csv"))
        # Parse the "science per unit cost"
        if metric == "fom":
            cost_df = p.df.copy()
            cost_df.value = [float(x.value / ps.cost_lookup[x.env]) for _, x in cost_df.iterrows()]
            cost_df.to_csv(os.path.join(data_outdir, "csv", f"hpcg_fom_per_dollar.csv"))
        instances = p.df.env.unique().tolist()
        build_config = p.df.problem_size.unique().tolist()
        df = pandas.DataFrame(0.0, columns=instances, index=list(build_config))
        idx = 0
        for i, row in p.df.iterrows():
            # Don't plot compatibility metadata here!
            if row.metric == "compatible":
                continue
            # Divide by the number of procs of the instance type
            if metric in divide_by_n:
                value = float(row.value) / ps.core_lookup[row.env]
            elif metric in raw_values:
                value = float(row.value)
            elif metric in divide_by_iterations:
                iters = total_cg_iterations[
                    (total_cg_iterations.metric == "total_cg_iterations")
                    & (total_cg_iterations.env == row.env)
                    & (total_cg_iterations.iteration == row.iteration)
                    & (total_cg_iterations.filename == row.filename)
                ]
                if iters.shape[0] != 1:
                    raise ValueError(
                        f"Found more than one iteration count for results {row}, this should not happen."
                    )
                iters = iters.value.tolist()[0]
                value = float(row.value) / iters
            else:
                raise ValueError(f"Not handled {key}")
            df.loc[row.problem_size, row.env] = value
            idx += 1

        fig = plt.figure(figsize=(24, 24))
        axes = []
        gs = plt.GridSpec(1, 2, width_ratios=[7, 0])
        axes.append(fig.add_subplot(gs[0, 0]))
        sns.set_style("whitegrid")
        g1 = sns.clustermap(
            df,
            cmap="crest",
            annot=False,
        )
        title = " ".join([x.capitalize() for x in metric.split("_")])
        g1.fig.suptitle(f"HPCG (xhpcg) {title}")
        plt.tight_layout()
        plt.savefig(os.path.join(img_outdir, f"xhpcg_{metric}.svg"))
        plt.clf()
        plt.close()

        # Save both to data frames - we need to extract clustering
        row_idx = df.index[g1.dendrogram_row.reordered_ind]
        col_idx = df.columns[g1.dendrogram_col.reordered_ind]
        values = [list(x) for x in list(df.loc[row_idx, col_idx].values)]
        fom_export = {
            "name": f"HPCG (xhpcg) {title}",
            "rowLabels": list(row_idx),
            "colLabels": list(col_idx),
            "values": values,
            "colorscale": "Viridis",
            "reversescale": True,
            "zmin": df.min().min(),
            "zmax": df.max().max(),
            "coreMap": ps.core_lookup,
        }
        ps.write_json(
            fom_export, os.path.join(data_outdir, "json", f"data_{metric}.json")
        )


def plot_results(df, outdir):
    """
    Plot analysis results
    """
    img_outdir = os.path.join(outdir, "img")
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

    # We need to know unique instances and optim/arch combos
    instance_set = set()
    build_config = set()
    for metric, instances in frames.items():
        # Only plot fom, compatible, and duration here
        # We will show the rest in the interactive plot
        if metric not in ["fom", "duration", "compatible"]:
            continue
        for instance, data_frame in instances.items():
            instance_set.add(instance)
            [build_config.add(x) for x in data_frame.problem_size.unique()]
            if metric == "compatible":
                plot_compatible(
                    data_frame, instance, os.path.join(img_outdir, "compatibility")
                )
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
            plt.savefig(
                os.path.join(img_outdir, metric, f"xhpcg-{instance}-{metric}.svg")
            )
            plt.clf()

            # Print the total number of data points
            print(f"Total number of datum: {data_frame.shape[0]}")

    # Now we want to calculate the cost per unit of science.
    # Add cost per hour
    fom_cost_df = pandas.DataFrame(0.0, columns=instances, index=list(build_config))

    # This one doesn't account for cost
    fom_df = pandas.DataFrame(0.0, columns=instances, index=list(build_config))
    idx = 0
    for metric, instances in frames.items():
        for instance, data_frame in instances.items():
            for i, row in data_frame.iterrows():
                fom_cost_df.loc[row.problem_size, row.env] = (
                    row.value / ps.cost_lookup[row.env]
                )
                fom_df.loc[row.problem_size, row.env] = float(row.value)
                idx += 1

    fig = plt.figure(figsize=(24, 24))
    axes = []
    gs = plt.GridSpec(1, 2, width_ratios=[7, 0])
    axes.append(fig.add_subplot(gs[0, 0]))
    sns.set_style("whitegrid")
    g1 = sns.clustermap(
        fom_cost_df,
        cmap="crest",
        annot=False,
    )
    g1.fig.suptitle(f"HPCG (xhpcg) FOM (Gflops/sec) Units Per Dollar")
    g1.ax_cbar.set_title("Glops/Second/1 USD")
    plt.tight_layout()
    plt.savefig(os.path.join(img_outdir, f"xhpcg-science-per-dollar.svg"))
    plt.clf()

    fig = plt.figure(figsize=(24, 24))
    axes = []
    gs = plt.GridSpec(1, 2, width_ratios=[7, 0])
    axes.append(fig.add_subplot(gs[0, 0]))
    sns.set_style("whitegrid")
    g2 = sns.clustermap(
        fom_df,
        cmap="crest",
        annot=False,
    )
    g2.fig.suptitle(f"HPCG (xhpcg) FOM (Gflops/sec)")
    g2.ax_cbar.set_title("Glops/Second")
    plt.tight_layout()
    plt.savefig(os.path.join(img_outdir, f"xhpcg-fom-clustermap.svg"))
    plt.clf()

    # Save both to data frames - we need to extract clustering
    row_idx = fom_cost_df.index[g1.dendrogram_row.reordered_ind]
    col_idx = fom_cost_df.columns[g1.dendrogram_col.reordered_ind]
    values = [list(x) for x in list(fom_cost_df.loc[row_idx, col_idx].values)]
    fom_per_dollar = {
        "name": "HPCG FOM (Gflops/sec per Dollar)",
        "rowLabels": list(row_idx),
        "colLabels": list(col_idx),
        "values": values,
        "colorscale": "Viridis",
        "reversescale": True,
        "zmin": fom_cost_df.min().min(),
        "zmax": fom_cost_df.max().max(),
        "coreMap": ps.core_lookup,
    }
    ps.write_json(fom_per_dollar, os.path.join(outdir, "heatmap", "json", "data_fom_per_dollar.json"))
    row_idx = fom_df.index[g2.dendrogram_row.reordered_ind]
    col_idx = fom_df.columns[g2.dendrogram_col.reordered_ind]
    values = [list(x) for x in list(fom_df.loc[row_idx, col_idx].values)]
    fom_overall = {
        "name": "HPCG FOM (Gflops/sec)",
        "rowLabels": list(row_idx),
        "colLabels": list(col_idx),
        "values": values,
        "colorscale": "Viridis",
        "reversescale": True,
        "zmin": fom_df.min().min(),
        "zmax": fom_df.max().max(),
        "coreMap": ps.core_lookup,
    }
    ps.write_json(fom_overall, os.path.join(outdir, "heatmap", "json", "data_raw_fom.json"))



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
    compats["instance_names"].add(instance)

    # 1. plot just FOM
    # 2. study across instance sizes of same type
    # Fill it in!
    for idx, row in df.iterrows():
        if row.value is True:
            arch, optim = row.problem_size.rsplit("-", 1)
            compat_df.loc[optim, arch] = 1

    ordered_df, rows, cols = get_ordered_matrix_and_labels(compat_df)

    # Add to global set
    compats["instances"][instance] = {
        "heatmap": ordered_df,
        "optimizations": rows,
        "microarches": cols,
        "platform": ps.arches_lookup[instance],
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
