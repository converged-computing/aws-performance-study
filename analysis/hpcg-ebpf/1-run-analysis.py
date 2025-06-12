#!/usr/bin/env python3

import argparse
import os
import sys
import pandas
import json

import matplotlib.pylab as plt
import seaborn as sns

here = os.path.dirname(os.path.abspath(__file__))
analysis_root = os.path.dirname(here)
root = os.path.dirname(analysis_root)
sys.path.insert(0, analysis_root)

import performance_study as ps

sns.set_theme(style="whitegrid", palette="muted")

# These are files I found erroneous - no result, or incomplete result
# Details included with each, and more exploration is likely needed to quantify
# error types
errors = []
error_regex = "(%s)" % "|".join(errors)


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
        "--non-anon",
        help="Generate non-anon",
        action="store_true",
        default=False,
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
    plot_results(df, outdir, args.non_anon)


def parse_flux_jobs(item):
    """
    Parse flux jobs. We first find output, then logs and events
    """
    jobs = {}
    current_job = None
    jobid = None
    lines = item.split("\n")

    # FOM for HPCG is the Gflop/s rate based on the apparent number of floating-point operations executed during the benchmark run.
    # Get final results. IMPORTANT: there are more here we could plot
    foms = [float(x.split("=")[-1]) for x in lines if "Final Summary::HPCG result" in x]

    # Get the results.
    while lines:
        line = lines.pop(0)

        # This is the start of a job
        if "FLUX-RUN START" in line and "echo" not in line:
            jobid = line.split(" ")[-1]
            current_job = []
            jobs[jobid] = {}
        # This is the end of a job
        elif "FLUX-RUN END" in line and "echo" not in line:
            jobs[jobid]["log"] = "\n".join(current_job)
            jobid = None
        elif jobid is not None:
            current_job.append(line)
            continue

    lines = item.split("\n")
    while lines:
        line = lines.pop(0)

        # Here, study id is job id above (e.g. amg2023-iter-1)
        if "FLUX-JOB START" in line and "echo" not in line:
            jobid, study_id = line.split(" ")[-2:]
            # I shelled in and ran hostname for osu, oops
            if study_id == "null":
                ps.find_section(lines, "FLUX-JOB-JOBSPEC")
                ps.find_section(lines, "FLUX-JOB-RESOURCES")
                ps.find_section(lines, "FLUX-JOB-EVENTLOG")
                continue
            jobs[study_id]["fluxid"] = jobid
            jobspec, lines = ps.find_section(lines, "FLUX-JOB-JOBSPEC")
            jobs[study_id]["jobspec"] = jobspec
            jobs[study_id]["fom"] = foms.pop(0)
            resources, lines = ps.find_section(lines, "FLUX-JOB-RESOURCES")
            jobs[study_id]["resources"] = resources
            events, lines = ps.find_section(lines, "FLUX-JOB-EVENTLOG")
            jobs[study_id]["events"] = events

            # Calculate duration
            start = [x for x in events if x["name"] == "shell.start"][0]["timestamp"]
            done = [x for x in events if x["name"] == "done"][0]["timestamp"]
            jobs[study_id]["duration"] = done - start

            assert "FLUX-JOB END" in lines[0]
            lines.pop(0)

        # Note that flux job stats are at the end, we don't parse

    return jobs


def add_hpcg_result(p, indir, filename, ebpf=None, gpu=False):
    """
    Add a new hpcg result
    """
    exp = ps.ExperimentNameParser(filename, indir)

    # Sanity check the files we found
    print(filename)
    env_name = filename.split(os.sep)[-2]
    instance = filename.split(os.sep)[-3]
    exp.show()

    # Use the env field for the instance type.
    p.set_context(exp.cloud, instance, exp.env_type, exp.size)

    item = ps.read_file(filename)
    if "exited with exit code 132" in item:
        print(f"Study {filename} was not compatible")
        p.add_result("compatible", False, env_name)
        return p
    p.add_result("compatible", True, env_name)

    jobs = parse_flux_jobs(item)
    for _, metadata in jobs.items():
        if not metadata:
            continue
        p.add_result("fom", metadata["fom"], env_name)
        p.add_result("duration", metadata["duration"], env_name)
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


def plot_results(df, outdir, non_anon=False):
    """
    Plot analysis results
    """
    img_outdir = os.path.join(outdir, "img")
    if not os.path.exists(img_outdir):
        os.makedirs(img_outdir)

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
                plot_compatible(data_frame, instance, img_outdir)
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
            plt.savefig(os.path.join(img_outdir, f"xhpcg-{instance}-{metric}.svg"))
            # plt.savefig(os.path.join(img_outdir, f"xhpcg-{instance}-{metric}.png"))
            plt.clf()

            # Print the total number of data points
            print(f"Total number of datum: {data_frame.shape[0]}")


def plot_compatible(df, instance, img_outdir):
    """
    Plot compatibility matrix
    """
    optims = list(set([x.split("-")[-1] for x in df.problem_size.unique().tolist()]))
    microarches = list(
        set([x.rsplit("-", 1)[0] for x in df.problem_size.unique().tolist()])
    )
    microarches.sort()
    optims.sort()
    compat_df = pandas.DataFrame(0, columns=microarches, index=optims)

    # Fill it in!
    for idx, row in df.iterrows():
        if row.value is True:
            arch, optim = row.problem_size.rsplit("-", 1)
            compat_df.loc[optim, arch] = 1

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
    plt.savefig(os.path.join(img_outdir, f"xhpcg-optimization-{instance}.png"))
    plt.clf()


if __name__ == "__main__":
    main()
