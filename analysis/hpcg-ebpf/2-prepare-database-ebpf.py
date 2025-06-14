#!/usr/bin/env python3

import argparse
import sys
import re
import json
import os

here = os.path.dirname(os.path.abspath(__file__))
analysis_root = os.path.dirname(here)
root = os.path.dirname(analysis_root)
sys.path.insert(0, analysis_root)

import performance_study as ps

# Global counts for number of datum
total_counts = {}


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
    global db
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Output images and data
    outdir = os.path.abspath(args.out)
    indir = os.path.abspath(args.root)

    # We absolutely want on premises results here
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Database path
    db_path = os.path.join(outdir, "ebpf-data.sqlite")
    db = ps.Database(db_path)

    # Find input files (skip anything with test)
    files = ps.find_inputs(indir, "cpu.out") + ps.find_inputs(indir, "futex.out")
    parse_data(indir, outdir, files, db_path)


def add_ebpf_result(indir, filename, io_counts):
    """
    Add an ebpf result
    
    # look at hugepages
    # plot would be good to have relative increase/decrease
    """
    global db
    global total_counts
    env_name = filename.split(os.sep)[-2]
    env_name = env_name.replace('-arm', '')
    exp = ps.ExperimentNameParser(filename, indir)

    # We will use env for instance type
    instance = filename.split(os.sep)[-3]
    item = ps.read_file(filename)
    if "PROGRAM" not in item:
        return
    hpcg_run = ps.read_file(os.path.join(os.path.dirname(filename), "hpcg.out"))
    if "exited with exit code 132" in hpcg_run:
        env_name = f"{env_name}-not-compatible"

    analysis = [x for x in item.split("\n") if "PROGRAM" in x][0].split(":")[-1].strip()
    if analysis not in total_counts:
        total_counts[analysis] = 0
    total_counts[analysis] += 1
    elif analysis == "futex":
        for name, value, command in parse_futex(item):
            db.add_result(
                exp.cloud,
                analysis,
                env_name,
                instance,
                exp.size,
                name,
                value,
                command=command,
            )
    elif analysis == "cpu":
        for name, value, command in parse_cpu(item):
            db.add_result(
                exp.cloud,
                analysis,
                env_name,
                instance,
                exp.size,
                name,
                value,
                command=command,
            )
    else:
        raise ValueError(analysis)


def parse_cpu(item):
    """
    Parse TCP
    """
    item = item.split("Cleaning up BPF resources")[-1]
    lines = [x for x in item.split("\n") if x and "Possibly lost" not in x][1:-1]
    item = "\n".join(lines)
    models = json.loads(item.split("Initiating cleanup sequence...")[-1])
    for model_type, model_names in models.items():
        for model in model_names:

            # E.g,. migration/22, kworker/88:2
            command = model["comm"].split("/")[0]

            # Add the median for now
            time_waiting_q = (
                model["runq_latency_stats_ns"]["median_ns"]
                * model["runq_latency_stats_ns"]["count"]
            )
            if time_waiting_q is not None:
                yield "cpu_waiting_ns", time_waiting_q, command
            try:
                time_running = (
                    model["on_cpu_stats_ns"]["median_ns"]
                    * model["on_cpu_stats_ns"]["count"]
                )
            except:
                time_running = None
            if time_running is not None:
                yield "cpu_running_ns", time_running, command


def parse_futex(item):
    """
    Futex parsing
    """
    item = item.split("Cleaning up BPF resources")[-1]
    lines = [x for x in item.split("\n") if x][1:-1]
    item = "\n".join(lines)
    models = json.loads(item)

    # {'tgid': 3592,
    # 'comm': 'containerd',
    # 'cgroup_id': 6520,
    # 'wait_duration_stats_ns': {'count': 1693.0,
    #  'sum': 224436362166.99985,
    #  'mean': 132567254.67631415,
    #  'median': 221633.7996235165,
    #  'min': 812,
    #  'max': 25489918732,
    #  'variance_ns2': 9.336119483708768e+17,
    #  'iqr_ns': 97748409.18476921,
    #  'q1_ns': 65205.02079568948,
    #  'q3_ns': 97813614.2055649},
    # 'futex_op_counts': {'128': 1693},
    # 'first_seen_ts_ns': 1941370732957,
    # 'last_seen_ts_ns': 1975537850634}
    for model_type, model_names in models.items():
        for model in model_names:
            # Add the median for now...
            median = model["wait_duration_stats_ns"]["median"]
            # The number of times
            count = 0
            for futex_id, increment in model["futex_op_counts"].items():
                count += increment
            yield "median_futex_wait", median * count, model["comm"]


def parse_shmem(item):
    """
    Shared memory parsing
    """
    sections = item.split("Shared Memory Stats Summary")[1:]
    for timepoint, section in enumerate(sections):
        # Just get to the last timepoint
        continue

    for line in section.split("\n"):
        if not line.startswith("{"):
            continue
        datum = json.loads(line)

        # The munmap seems to change, the rest don't.
        yield "get_mb", datum["get_mb"], datum["comm"]
        yield "mmap_sh", datum["mmap_sh"], datum["comm"]
        yield "munmap", datum["munmap"], datum["comm"]
        yield "mmap_sh_mb", datum["mmap_sh_mb"], datum["comm"]


# {'tgid': 28139,
# 'comm': 'lmp',
# 'shmget': 1,
# 'shmat': 1,
# 'shmdt': 1,
# 'rmid': 1,
# 'get_mb': 0.00390625,
# 'shmopen': 0,
# 'unlink': 0,
# 'mmap_sh': 88,
# 'munmap': 35,
# 'mmap_sh_mb': 352.00067138671875}


def parse_data(indir, outdir, files, db_path):
    """
    Parse filepaths for environment, etc., and results files for data.
    """
    global db
    global total_counts

    # It's important to just parse raw data once, and then use intermediate
    total = len(files) - 1
    for i, filename in enumerate(files):
        if "test" in filename:
            continue
        print(f"{i}/{total}", end="\r")
        basename = os.path.basename(filename)
        if "hpcg.out" in filename:
    
        else:
            add_ebpf_result(indir, filename, io_counts)

    db.close()
    print(json.dumps(total_counts))
    print("Done parsing lammps eBPF results!")


if __name__ == "__main__":
    main()
