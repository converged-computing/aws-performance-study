#!/usr/bin/env python3

import argparse
import os
import re
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


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run analysis",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--root",
        help="root directory with experiments",
        default=os.path.join(root, "experiment", "eks", "cpu", "models"),
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
    files = [x for x in ps.find_inputs(indir, "hpcg.out") if "fom-per-dollar" not in x and "write" not in x]

    # Create outdirs for images - stay organized!
    img_outdir = os.path.join(outdir, "img")

    # Saves raw data to file
    df = parse_data(indir, outdir, files)
    plot_results(df, outdir)


def add_hpcg_result(p, indir, filename, ebpf=None, gpu=False):
    """
    Add a new hpcg result
    """
    exp = ps.ExperimentNameParser(filename, indir)

    # Sanity check the files we found
    model = filename.split(os.sep)[-3].replace('-normalized', '')

    # Use the env field for the instance type.
    p.set_context(exp.cloud, model, exp.env_type, exp.size)

    # We don't care about iterations here
    item = ps.read_file(filename)

    pod = ps.read_json(filename.replace("hpcg.out", "pod.json"))
    pod_times = ps.read_file(filename.replace("hpcg.out", "pod-time.txt"))
    waiting_time = parse_duration_to_seconds(pod_times.split("\n")[1].split(" ")[-1])
    instance = pod["spec"]["nodeSelector"]["node.kubernetes.io/instance-type"]

    # Get the benchmark total times and FOMs (each of these is an iteration, should be 3)
    lines = item.split("\n")
    metrics = {
        "fom": "Final Summary::HPCG result",
        "duration": "Benchmark Time Summary::Total",
        "total_cg_iterations": "Iteration Count Information::Total number of optimized iterations",
        "gflops_per_second_waxpby": "GFLOP/s Summary::Raw WAXPBY",
        "gflops_per_second_spmv": "GFLOP/s Summary::Raw SpMV",
        "memory_bandwidth_across_kernels_total": "GB/s Summary::Raw Total B/W",
    }

    # Get all metrics, actual lists of
    others = {}
    for key, prefix in metrics.items():
        values = [float(x.rsplit("=", 1)[-1]) for x in lines if prefix in x]
        others[key] = values

    costs = [ps.cost_lookup[instance] * x for x in values]
    
    # Don't forget to normalize
    #for key, value in others.items():
    #    if "fom" in key or "gflops" in key or "memory" in key:
    #        others[key] = [x/ps.core_lookup[instance] for x in value]

    # Calculate fom per dollar if running for an hour
    fraction_of_hour = [x / 3600 for x in others["duration"]]
    total_costs = [
        ps.cost_lookup[instance] * cost for i, cost in enumerate(fraction_of_hour)
    ]
    others["fom_per_dollar"] = [
        fom_value / total_costs[i] for i, fom_value in enumerate(others["fom"])
    ]

    for key, values in others.items():
        # The ordering is consistent between lists
        for iteration, value in enumerate(values):
            p.add_result(key, value, model, filename=filename, iteration=iteration)
            # waiting times just is added once
            if iteration == 0:
                p.add_result(
                    "waiting_time",
                    waiting_time,
                    model,
                    filename=filename,
                    iteration=iteration,
                )
            if key == "duration":
                p.add_result(
                    "cost",
                    total_costs[iteration],
                    model,
                    filename=filename,
                    iteration=iteration,
                )
            p.add_result(
                "instance", instance, model, filename=filename, iteration=iteration
            )

    return p


def parse_data(indir, outdir, files):
    """
    Parse filepaths for environment, etc., and results files for data.
    """
    p = ps.ProblemSizeParser("hpcg")

    # It's important to just parse raw data once, and then use intermediate
    for filename in files:
        pod = ps.read_json(filename.replace("hpcg.out", "pod.json"))
        pod_times = ps.read_file(filename.replace("hpcg.out", "pod-time.txt"))
        waiting_time = pod_times.split("\n")[1].split(" ")[-1]

        model = filename.split(os.sep)[-3]
        instance = pod["spec"]["nodeSelector"]["node.kubernetes.io/instance-type"]
        if instance not in ps.cost_lookup:
            # This is just a GPU instance inf2.xlarge and we (unfairly) didn't use the GPU, so skip it
            print(f"Warning: no cost for {instance}, not including in data")
            continue
        # There are 2 inf instances we will skip, which is OK didn't use GPU
        p = add_hpcg_result(p, indir, filename)

    # Save stuff to file first
    p.df.to_csv(os.path.join(outdir, "hpcg-results.csv"))
    return p.df


def parse_duration_to_seconds(duration_str):
    # Prepare string: make lowercase, remove all whitespace.
    processed_str = duration_str.lower().replace(" ", "")
    if not processed_str:
        raise ValueError("Input string cannot be empty.")

    pairs = re.findall(r"(\d+)([hms])", processed_str)
    reconstructed_str = "".join([val + unit for val, unit in pairs])
    if not pairs or reconstructed_str != processed_str:
        raise ValueError(f"Invalid duration format: '{duration_str}'")

    multipliers = {"h": 3600, "m": 60, "s": 1}
    total_seconds = 0
    for value_str, unit in pairs:
        total_seconds += int(value_str) * multipliers[unit]
    return total_seconds


def plot_results(df, outdir):
    """
    Plot analysis results
    """
    img_outdir = os.path.join(outdir, "img")
    colors = list(sns.color_palette())
    palette = {}
    for key in [
        "random",
        "fom-model",
        "memory-bandwidth-across-kernels-total",
        "memory-bandwidth-across-kernels-write",
        "gflops-per-second-waxpby",
        "gflops-per-second-spmv",
    ]:
        palette[key] = list(colors.pop(0))

    # Orders vary by plot
    orders = {
        "cost": [
            "fom-model",
            "memory-bandwidth-across-kernels-total",
            "random",
            "gflops-per-second-spmv",
            "gflops-per-second-waxpby",
        ],
        "fom": [
            "random",
#            "gflops-per-second-spmv",
#            "gflops-per-second-waxpby",
#            "memory-bandwidth-across-kernels-total",
            "fom-model",
        ],
        "memory_bandwidth_across_kernels_total": [
            "random",
            "gflops-per-second-spmv",
            "gflops-per-second-waxpby",
            "fom-model",
            "memory-bandwidth-across-kernels-total",
        ],
        "gflops_per_second_spmv": [
            "random",
            "gflops-per-second-spmv",
            "gflops-per-second-waxpby",
            "memory-bandwidth-across-kernels-total",
            "fom-model",
        ],
        "gflops_per_second_waxpby": [
            "random",
            "fom-model",
            "memory-bandwidth-across-kernels-total",
            "gflops-per-second-spmv",
            "gflops-per-second-waxpby",
        ],
        "memory_bandwidth_across_kernels_write": [
            "random",
            "gflops-per-second-spmv",
            "gflops-per-second-waxpby",
            "fom-model",
            "memory-bandwidth-across-kernels-total",
        ],
    }

    # Make a plot for seconds runtime, and each FOM set.
    # We can look at the metric across sizes, colored by experiment
    fig = plt.figure(figsize=(5, 3))
    axes = []
    gs = plt.GridSpec(1, 1)
    for i in range(1):
        axes.append(fig.add_subplot(gs[0, i]))
    i = 0

    for metric in df.metric.unique():
        metric_df = df[df.metric == metric]
        print(metric)
        if metric == "fom":
            # How many more times better?
            fom_mean = metric_df[metric_df.env == 'fom-model'].value.mean()
            random_mean = metric_df[metric_df.env == 'random'].value.mean()
            # fom/random is how many times better? 4.1169396154823765
            print(f"fom/random is how many times better? {fom_mean/random_mean}")
            
        if metric not in [
#            "gflops_per_second_waxpby",
#            "gflops_per_second_spmv",
            "fom",
#            "memory_bandwidth_across_kernels_write",
#            "memory_bandwidth_across_kernels_total",
        ]:
            continue
        sns.set_style("whitegrid")
        func = sns.barplot
        func(
            metric_df,
            ax=axes[i],
            x="problem_size",
            y="value",
            hue="env",
            palette=palette,
            order=orders.get(metric),
            err_kws={"color": "darkred"},
        )
        title = metric.replace("_", " ")
        if metric in ["duration", "waiting_time"]:
            axes[i].set_title(f"HPCG (xhpcg) {metric.capitalize()}", fontsize=12)
        elif metric == "cost":
            axes[i].set_title(f"HPCG (xhpcg) {metric.capitalize()}", fontsize=12)
        elif "memory" in metric and "total" in metric:
            axes[i].set_title(f"HPCG (xhpcg) Memory Bandwidth", fontsize=12)
        elif "memory" in metric and "write" in metric:
            axes[i].set_title(f"HPCG (xhpcg) Memory Bandwidth Write", fontsize=12)
        elif "spmv" in metric:
            axes[i].set_title(f"HPCG (xhpcg) SpmV", fontsize=12)
        elif "waxpby" in metric:
            axes[i].set_title(f"HPCG (xhpcg) Waxpby", fontsize=12)
        elif "fom" in metric:
            axes[i].set_title(f"HPCG (xhpcg) Figure of Merit", fontsize=12)
        if i == 0:
            axes[i].set_ylabel("Total GBytes/Second", fontsize=10)
        else:
            axes[i].set_ylabel("", fontsize=5)
        axes[i].set_xlabel("", fontsize=12)
        # axes[i].tick_params(axis="x", rotation=45)
        #labels = axes[i].get_xticklabels()
        #new_labels = []
        #for label in labels:
        #    parts = label._text.split("-")
        #    label = [x + "\n" if "per" not in x else x + " " for x in parts]
        #    new_labels.append("".join(label))
        #axes[i].set_xticklabels(new_labels, rotation=0, ha="right")
        i += 1
    plt.tight_layout()
    plt.savefig(os.path.join(img_outdir, f"xhpcg-all-metrics.png"))
    plt.savefig(os.path.join(img_outdir, f"xhpcg-all-metrics.svg"))
    plt.clf()

    # Report on waiting time (cost of autoscaling)
    print(df[df.metric=="waiting_time"].groupby(['env']).value.median())
    print(df[df.metric=='instance'].groupby(['env', 'value']).count())


if __name__ == "__main__":
    main()
