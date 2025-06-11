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

    #print('FIND OUTPUT')
    #import IPython
    #IPython.embed()
    #sys.exit()
    # Output is cat at the end.
    # start_idx = lines.index([x for x in lines if "FLUX-RUN END" in x][-1]) + 1
    # end_idx = lines.index([x for x in lines if x.startswith("FLUX-JOB START")][0]) -1 
    # results = lines[start_idx:end_idx]

    # FOM for HPCG is the Gflop/s rate based on the apparent number of floating-point operations executed during the benchmark run. 
    
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
    p.set_context(exp.cloud, exp.env, exp.env_type, exp.size)

    # Sanity check the files we found
    print(filename)
    env_name = filename.split(os.sep)[-2]
    exp.show()

    item = ps.read_file(filename)
    jobs = parse_flux_jobs(item)
    for _, metadata in jobs.items():
        if not metadata:
            continue
        #step, seconds = parse_matom_steps(metadata["log"])
        #p.add_result(step, seconds, env_name)
        #p.add_result("cpu-usage", cpu_use, env_name)
        #p.add_result("wall-time", wall_time, env_name)
        p.add_result("duration", metadata["duration"], env_name)
        # p.add_result("hookup-time", metadata["duration"] - wall_time, env_name)
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
        basename = os.path.basename(filename)
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

    df['optimization'] = [x.split('-')[-1] for x in df.problem_size.tolist()]

    frames = {}
    # Make a plot for seconds runtime, and each FOM set.
    # We can look at the metric across sizes, colored by experiment
    for metric in df.metric.unique():
        metric_df = df[df.metric == metric]
        frames[metric] = {"cpu": metric_df}

    print(metric_df.problem_size.unique())
    for metric, data_frames in frames.items():
        fig = plt.figure(figsize=(18, 5))
        axes = []
        gs = plt.GridSpec(1, 2, width_ratios=[7, 1])
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))

        data_sorted = data_frames["cpu"].sort_values(by='value', ascending=True)    
        order = data_sorted.groupby('problem_size')['value'].mean().sort_values(ascending=True).index

        sns.set_style("whitegrid")
        sns.barplot(
            data_sorted,
            ax=axes[0],
            x="problem_size",
            y="value",
            hue="optimization",
            err_kws={"color": "darkred"},
            order=order,
        )
        if metric in ["duration", "wall-time", "hookup-time"]:
            axes[0].set_title(f"HPCG (xhpcg) {metric.capitalize()}", fontsize=14)
            axes[0].set_ylabel("Seconds", fontsize=14)
        elif "cpu" in metric:
            axes[0].set_title("LAMMPS CPU Usage", fontsize=14)
            axes[0].set_ylabel("% CPU usage", fontsize=14)
        elif "katom" in metric:
            axes[0].set_title("LAMMPS K/Atom Steps per Second", fontsize=14)
            axes[0].set_ylabel("M/Atom Steps Per Second", fontsize=14)
        else:
            axes[0].set_title("LAMMPS M/Atom Steps per Second", fontsize=14)
            axes[0].set_ylabel("M/Atom Steps Per Second", fontsize=14)
        axes[0].set_xlabel("Micro-architecture and Optimization", fontsize=14)
        axes[0].tick_params(axis='x', rotation=90)

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
        plt.savefig(os.path.join(img_outdir, f"xhpcg-{metric}.svg"))
        plt.savefig(os.path.join(img_outdir, f"xhpcg-{metric}.png"))
        plt.clf()

        # Print the total number of data points
        print(f'Total number of datum: {data_frames["cpu"].shape[0]}')


if __name__ == "__main__":
    main()
