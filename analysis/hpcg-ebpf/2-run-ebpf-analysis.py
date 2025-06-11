#!/usr/bin/env python3

import argparse
import os
import sys

import pandas
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
    db_file = os.path.join(outdir, "ebpf-data.sqlite")

    # We absolutely want on premises results here
    if not os.path.exists(db_file):
        sys.exit(f"Database file {db_file} does not exist.")

    db = ps.Database(db_file)
    plot_ebpfs(db, outdir)


def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    if union == 0:
        return 0
    return intersection / union


def comparison_analysis(
    subset,
    result,
    cpu_vs_gpu_difference=None,
    ubuntu_vs_rocky_difference=None,
    mpich_vs_openmpi_difference=None,
):
    # Break into ubuntu-openmpi with and without GPU
    # We need to filter out install prefix of .so files
    with_gpu = subset[subset.experiment == "ubuntu-openmpi-gpu"]
    without_gpu = subset[subset.experiment == "ubuntu-openmpi-cpu"]
    with_gpu_paths, with_gpu_lookup = get_unique_paths(with_gpu)
    without_gpu_paths, without_gpu_lookup = get_unique_paths(without_gpu)

    # Take the set differences
    in_gpu_not_cpu = set(with_gpu_paths) - set(without_gpu_paths)
    in_cpu_not_gpu = set(without_gpu_paths) - set(with_gpu_paths)

    # Calculate jacaard before reconstituting the paths
    if cpu_vs_gpu_difference is not None:
        score_cpu_gpu = jaccard_similarity(set(with_gpu_paths), set(without_gpu_paths))
        cpu_vs_gpu_difference.append(score_cpu_gpu)

    # Assemble back into full paths and save
    in_cpu_not_gpu = {without_gpu_lookup.get(x) or x for x in in_cpu_not_gpu}
    in_gpu_not_cpu = {with_gpu_lookup.get(x) or x for x in in_gpu_not_cpu}
    result.update(
        {
            "in_gpu_not_cpu": sorted(list(in_gpu_not_cpu)),
            "in_cpu_not_gpu": sorted(list(in_cpu_not_gpu)),
        }
    )
    # For those that are the same, look at difference in count of opens?
    # Now look at cpu experiments
    rocky_openmpi = subset[subset.experiment == "rocky-openmpi-cpu"]
    ubuntu_openmpi = subset[subset.experiment == "ubuntu-openmpi-cpu"]
    ubuntu_mpich = subset[subset.experiment == "ubuntu-mpich-cpu"]
    rocky_openmpi_paths, rocky_openmpi_lookup = get_unique_paths(rocky_openmpi)
    ubuntu_openmpi_paths, ubuntu_openmpi_lookup = get_unique_paths(ubuntu_openmpi)
    ubuntu_mpich_paths, ubuntu_mpich_lookup = get_unique_paths(ubuntu_mpich)

    in_ubuntu_not_rocky = set(ubuntu_openmpi_paths) - set(rocky_openmpi_paths)
    in_rocky_not_ubuntu = set(rocky_openmpi_paths) - set(ubuntu_openmpi_paths)
    in_openmpi_not_mpich = set(ubuntu_openmpi_paths) - set(ubuntu_mpich_paths)
    in_mpich_not_openmpi = set(ubuntu_mpich_paths) - set(ubuntu_openmpi_paths)

    # Update scores
    if mpich_vs_openmpi_difference is not None:
        score_openmpi_mpich = jaccard_similarity(
            set(ubuntu_openmpi_paths), set(ubuntu_mpich_paths)
        )
        mpich_vs_openmpi_difference.append(score_openmpi_mpich)

    if ubuntu_vs_rocky_difference is not None:
        score_rocky_ubuntu = jaccard_similarity(
            set(rocky_openmpi_paths), set(ubuntu_openmpi_paths)
        )
        ubuntu_vs_rocky_difference.append(score_rocky_ubuntu)

    in_ubuntu_not_rocky = {
        ubuntu_openmpi_lookup.get(x) or x for x in in_ubuntu_not_rocky
    }
    in_rocky_not_ubuntu = {
        rocky_openmpi_lookup.get(x) or x for x in in_rocky_not_ubuntu
    }
    in_openmpi_not_mpich = {
        ubuntu_openmpi_lookup.get(x) or x for x in in_openmpi_not_mpich
    }
    in_mpich_not_openmpi = {
        ubuntu_mpich_lookup.get(x) or x for x in in_mpich_not_openmpi
    }
    result["in_ubuntu_not_rocky"] = sorted(list(in_ubuntu_not_rocky))
    result["in_rocky_not_ubuntu"] = sorted(list(in_rocky_not_ubuntu))
    result["in_openmpi_not_mpich"] = sorted(list(in_openmpi_not_mpich))
    result["in_mpich_not_openmpi"] = sorted(list(in_mpich_not_openmpi))


def get_unique_paths(pathset):
    """
    Get unique paths and a lookup for a set
    """
    lookup = {}
    _paths = set(pathset.metric_name.unique().tolist())
    paths = set()
    for path in _paths:
        if ".so" not in path:
            paths.add(path)
        else:
            # Normalize for version
            if ".so." in path:
                path = path.split(".so.")[0] + ".so"
            basename = os.path.basename(path)
            if basename in lookup:
                print(lookup[basename])
                print(path)
                print(f"Warning: .so {basename} is found in two places")
                lookup[basename] = basename
            else:
                lookup[basename] = path
            paths.add(basename)
    return paths, lookup


def get_shared_paths(A, B):
    """
    Get shared paths and counts for two sets.

    Note sure what to do with this info.
    """
    _paths_A = set(A.metric_name.unique().tolist())
    _paths_B = set(B.metric_name.unique().tolist())
    shared = _paths_A.intersection(_paths_B)
    A = A[A.metric_name.isin(shared)]
    B = B[B.metric_name.isin(shared)]
    diffs = {}
    for path in shared:
        diffs[path] = (
            A[A.metric_name == path].metric_value.tolist()[0]
            - B[B.metric_name == path].metric_value.tolist()[0]
        )
    return diffs


def get_table_for_analysis(db, name):
    """
    Get data frame of values for an entire analysis group by name
    """
    query = "SELECT * FROM performance_data WHERE analysis_name = ?"
    df = db.query_to_dataframe(query, params=(name,))
    sizes = [int(x) for x in df.nodes.unique()]
    sizes.sort()
    df.nodes = df.nodes.astype(int)

    # Create a column that combines experiment and environment
    df["experiment"] = df["environment"]

    df["optimization"] = [x.split("-")[-1] for x in df.experiment.tolist()]
    df["compatible"] = [
        False if "not-compatible" in x else True for x in df["experiment"].tolist()
    ]
    df["experiment"] = [
        x.replace("-not-compatible", "") for x in df.experiment.tolist()
    ]
    return df


def plot_shmem(db, outdir):
    """
    Look at shared memory as we increase in size.
    """
    analysis = "shmem"
    df = get_table_for_analysis(db, analysis)
    df = df[df.context == "lmp"]

    # Convert all nanoseconds to seconds.
    df["microseconds"] = df.metric_value / 1000
    df["seconds"] = df.metric_value / 1e9

    img_outdir = os.path.join(outdir, "img")
    for metric in df.metric_name.unique():
        fig = plt.figure(figsize=(10, 3.3))
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        sns.set_style("whitegrid")
        view = df[df.metric_name == metric]
        sns.barplot(
            view,
            ax=axes[0],
            x="nodes",
            y="metric_value",
            hue="experiment",
            err_kws={"color": "darkred"},
            order=[4, 8, 16, 32, 64, 128],
            # palette=cloud_colors,
        )
        axes[0].set_title(f"LAMMPS (lmp) CPU Shared Memory {metric}", fontsize=12)
        axes[0].set_ylabel(metric, fontsize=14)
        axes[0].set_xlabel("Nodes", fontsize=14)
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
            if ax is not None:
                ax.get_legend().remove()
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(img_outdir, f"lammps-shmem-{metric}.png"))
        plt.savefig(os.path.join(img_outdir, f"lammps-shmem-{metric}.svg"))
        plt.clf()


def plot_futex(db, outdir):
    """
    Look at tcp calls (and change) as we increase in size.
    """
    analysis = "futex"
    all_df = get_table_for_analysis(db, analysis)
    all_df = all_df[all_df.context == "xhpcg"]
    for instance in all_df.environment_type.unique():
        df = all_df[all_df.environment_type == instance]

        # Convert all nanoseconds to seconds.
        df["microseconds"] = df.metric_value / 1000
        df["optimization"] = [x.split("-")[-1] for x in df.experiment.tolist()]
        img_outdir = os.path.join(outdir, "img")
        fig = plt.figure(figsize=(17, 5))
        gs = plt.GridSpec(1, 2, width_ratios=[7, 1])
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        data_sorted = df.sort_values(by="microseconds", ascending=True)
        order = (
            data_sorted.groupby("experiment")["microseconds"]
            .mean()
            .sort_values(ascending=True)
            .index
        )
        sns.set_style("whitegrid")
        sns.barplot(
            data_sorted,
            ax=axes[0],
            x="experiment",
            y="microseconds",
            hue="optimization",
            err_kws={"color": "darkred"},
            order=order,
        )
        axes[0].set_title(
            f"HPCG (xhpcg) Cumulative Futex Wait Times for {instance}", fontsize=12
        )
        axes[0].set_ylabel("Time (µs)", fontsize=12)
        axes[0].set_xlabel("Micro-architecture and Optimization", fontsize=14)
        axes[0].tick_params(axis="x", rotation=90)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(-0.1, 0.5),
            frameon=False,
        )
        axes[1].axis("off")
        axes[0].get_legend().remove()
        plt.tight_layout()
        plt.savefig(os.path.join(img_outdir, f"xhpcg-median-futex-wait-{instance}.png"))
        plt.savefig(os.path.join(img_outdir, f"xhpcg-median-futex-wait-{instance}.svg"))
        plt.clf()


def plot_cpu(db, outdir):
    """
    Look at tcp calls (and change) as we increase in size.
    """
    analysis = "cpu"
    all_df = get_table_for_analysis(db, analysis)
    all_df = all_df[all_df.context == "xhpcg"]
    for instance in all_df.environment_type.unique():
        df = all_df[all_df.environment_type == instance]
        waiting = df[df.metric_name == "cpu_waiting_ns"]
        running = df[df.metric_name == "cpu_running_ns"]
        print(df)
        one = running.groupby(
            ["environment", "environment_type", "compatible"]
        ).metric_value.median()
        two = waiting.groupby(
            ["environment", "environment_type", "compatible"]
        ).metric_value.median()
        # TODO it would be better if we don't take median and can show variation.
        view = pandas.DataFrame(one / two)
        view.loc[:, "experiment"] = [
            x[0].replace("-not-compatible", "") for x in view.index
        ]
        view.loc[:, "environment"] = [x[1] for x in view.index]
        view.loc[:, "compatible"] = [x[2] for x in view.index]
        data_sorted = view.sort_values(by="metric_value", ascending=True)
        order = (
            data_sorted.groupby("experiment")["metric_value"]
            .mean()
            .sort_values(ascending=True)
            .index
        )

        # These are all cpu
        img_outdir = os.path.join(outdir, "img")
        fig = plt.figure(figsize=(22, 5))
        gs = plt.GridSpec(1, 2, width_ratios=[9, 1])
        axes = []
        axes.append(fig.add_subplot(gs[0, 0]))
        axes.append(fig.add_subplot(gs[0, 1]))
        sns.set_style("whitegrid")
        sns.barplot(
            data_sorted,
            ax=axes[0],
            x="experiment",
            y="metric_value",
            hue="compatible",
            err_kws={"color": "darkred"},
            order=order,
        )
        axes[0].set_title(
            f"HPCG (xhpcg) CPU Running/Waiting ratio for {instance}", fontsize=12
        )
        axes[0].set_ylabel("Ratio", fontsize=14)
        axes[0].set_xlabel("Micro-architecture and Optimization", fontsize=14)
        axes[0].tick_params(axis="x", rotation=90)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[1].legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(-0.1, 0.5),
            frameon=False,
        )
        axes[0].get_legend().remove()
        axes[1].axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(img_outdir, f"xhpcg-running-waiting-cpu-ratio-{instance}.png")
        )
        plt.savefig(
            os.path.join(img_outdir, f"xhpcg-running-waiting-cpu-ratio-{instance}.svg")
        )
        plt.clf()


def plot_tcp(db, outdir):
    """
    Look at tcp calls (and change) as we increase in size.
    """
    analysis = "tcp"

    # We want to calculate each nanoseconds as a percentage of total time
    times_df = pandas.read_csv(os.path.join(outdir, "lammps-results.csv"), index_col=0)

    # Get median time based on size and environment
    # We are only using the single samples for now
    filter_names = [
        "Ubuntu Mpich eBPF Sample",
        "Ubuntu OpenMPI eBPF Sample",
        "Rocky OpenMPI eBPF Sample",
        "Ubuntu OpenMPI eBPF GPU",
    ]
    times_df = times_df[times_df.problem_size.isin(filter_names)]
    times_df = times_df[times_df.metric == "duration"]
    lookup = times_df.groupby(["problem_size", "nodes"]).value.median().to_dict()
    df = get_table_for_analysis(db, analysis)

    # Get rid of mean bytes
    df = df[df.metric_name != "mean_bytes"]

    # Convert all nanoseconds to seconds.
    df["seconds"] = df.metric_value / 1e9

    res = []
    for command in df.context.unique():
        command_df = df[df.context == command]
        for i, row in command_df.iterrows():
            if row.experiment == "Ubuntu Mpich-cpu":
                key = "Ubuntu Mpich eBPF Sample"
            elif row.experiment == "Rocky OpenMPI-cpu":
                key = "Rocky OpenMPI eBPF Sample"
            elif row.experiment == "Ubuntu OpenMPI-cpu":
                key = "Ubuntu OpenMPI eBPF Sample"
            elif row.experiment == "Ubuntu OpenMPI-gpu":
                key = "Ubuntu OpenMPI eBPF GPU"
            else:
                raise ValueError(f"Unexpected environment {row.environment}")
            percentage_time = row.seconds / lookup[(key, row.nodes)]
            res.append(
                [row.nodes, row.experiment, percentage_time, row.metric_name, command]
            )
            # comm_df.loc[idx, :] = [row.nodes, row.experiment, percentage_time, row.metric_name]
            # idx += 1

    # Create DF of percentage values
    comm_df = pandas.DataFrame(res)
    import IPython

    IPython.embed()
    comm_df.columns = ["nodes", "experiment", "percentage", "metric", "command"]
    test = comm_df.groupby(
        ["nodes", "experiment", "metric", "command"]
    ).percentage.median()
    test.to_csv(os.path.join(outdir, "time-percentages-lammps.csv"))


# filter_df = comm_df[comm_df.metric == "duration_ns_0B-1KB"]
# filter_df[filter_df.command == "lmp-read_sock"]
# filter_df = filter_df[filter_df.command == "lmp-read_sock"]
# filter_df.groupby(['experiment', 'nodes']).percentage.median()
# filter_df.groupby(['experiment', 'nodes']).percentage.mean()


def plot_open_close(db, outdir):
    """
    Plot counts of open and close for each application
    """
    df = get_table_for_analysis(db, "open-close")

    img_outdir = os.path.join(outdir, "img", "open-close")
    if not os.path.exists(img_outdir):
        os.makedirs(img_outdir)

    # Let's choose one middle size? No, look across all
    # df = df[df.nodes == 64]

    # Let's look at differences between GPU and CPU for each command
    # and then differences between experiments. Also keep track of differences between sizes
    cpu_vs_gpu_difference = {}
    ubuntu_vs_rocky_difference = {}
    mpich_vs_openmpi_difference = {}

    # Results will combine across node sizes
    results = {}

    # Specific sizes
    matrix = {}
    for command in df.context.unique():
        if command not in matrix:
            matrix[command] = {}
            results[command] = {}
            cpu_vs_gpu_difference[command] = []
            ubuntu_vs_rocky_difference[command] = []
            mpich_vs_openmpi_difference[command] = []

        # This is doing the analysis for all data
        subset = df[df.context == command]
        comparison_analysis(subset, results[command])

        for size in sizes:
            if size not in matrix[command]:
                matrix[command][size] = {}
            # Make sorted histogram of counts > 1
            subset = df[df.context == command]
            subset = subset[subset.nodes == size]
            comparison_analysis(
                subset,
                matrix[command][size],
                cpu_vs_gpu_difference[command],
                ubuntu_vs_rocky_difference[command],
                mpich_vs_openmpi_difference[command],
            )

    for command, diffs in cpu_vs_gpu_difference.items():
        data = {
            "cpu vs gpu": diffs,
            "mpich vs openmpi": mpich_vs_openmpi_difference[command],
            "rocky vs ubuntu": ubuntu_vs_rocky_difference[command],
        }
        diff_df = pandas.DataFrame(data)
        diff_df = diff_df.transpose()
        command = command.replace(os.sep, "-")
        fig = plt.figure(figsize=(9, 4))
        sns.heatmap(diff_df, vmin=0.0, vmax=1.0)
        ax = plt.gca()
        ax.set_title(f"Jaccard Score ({command})", fontsize=12)
        ax.set_xlabel("Nodes", fontsize=12)
        ax.set_xticklabels(sizes)
        plt.tight_layout()
        plt.savefig(os.path.join(img_outdir, f"{command}-jaccard.svg"))
        plt.savefig(os.path.join(img_outdir, f"{command}-jaccard.png"))
        plt.clf()

    ps.write_json(
        results, os.path.join(outdir, "file-access-differences-all-nodes.json")
    )
    ps.write_json(matrix, os.path.join(outdir, "file-access-differences.json"))


def plot_ebpfs(db, outdir):
    """
    Plot ebpf result
    """
    analyses = db.query("Select distinct analysis_name from performance_data;")
    print(f"Found analysis: {analyses}")

    img_outdir = os.path.join(outdir, "img")
    if not os.path.exists(img_outdir):
        os.makedirs(img_outdir)

    # plot open close
    # plot_open_close(db, outdir)

    # Plot tcp
    # plot_tcp(db, outdir)
    plot_cpu(db, outdir)
    plot_futex(db, outdir)
    # plot_shmem(db, outdir)
    sys.exit()


if __name__ == "__main__":
    main()
