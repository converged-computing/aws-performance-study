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
    if analysis == "shmem":
        for name, value, command in parse_shmem(item):
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
    elif analysis == "tcp-socket":
        return
    elif analysis == "tcp":
        for name, value, context in parse_tcp(item):
            db.add_result(
                exp.cloud,
                analysis,
                env_name,
                instance,
                exp.size,
                name,
                value,
                context=context,
            )
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
    elif analysis == "open_close":
        io_counts = parse_io(item, exp_name, exp, io_counts, filename)
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


def parse_tcp(item):
    """
    Parse TCP
    """
    item = item.split("--- FINAL AGGREGATED STATISTICS (JSON) ---")[-1]
    lines = [x for x in item.split("\n") if x][:-1]
    item = "\n".join(lines)
    models = json.loads(item)

    pattern = r"TGID\((?P<tgid>.*?)\)_COMM\((?P<command>.*?)\)_EVT\((?P<event>.*?)\)"
    bucket_pattern = r"TGID\((?P<tgid>.*?)\)_COMM\((?P<command>.*?)\)_EVT\((?P<event>.*?)\)_BUCKET\((?P<bucket>.*?)\)"

    def update_meta(meta):
        if "lmp" in meta["command"]:
            meta["command"] = "lmp"
        if "flux-broker" in meta["command"]:
            meta["command"] = "flux-broker"
        if " " in meta["command"]:
            meta["command"] = meta["command"].split(" ")[0]
        return meta

    for model_type, model_names in models.items():
        for model_name, model in model_names.items():
            if model_type == "overall_byte_stats":
                meta = re.search(pattern, model_name).groupdict()
                meta = update_meta(meta)
                event = f"{meta['command']}-{meta['event'].lower()}"
                yield "mean_bytes", model["mean"], event
            elif "BUCKET" in model_name:
                meta = re.search(bucket_pattern, model_name).groupdict()
                meta = update_meta(meta)
                bucket = meta["bucket"]
                event = f"{meta['command']}-{meta['event'].lower()}"
                yield f"duration_ns_{bucket}", model["p95"] * model["count"], event
            else:
                meta = re.search(pattern, model_name).groupdict()
                meta = update_meta(meta)
                event = f"{meta['command']}-{meta['event'].lower()}"
                yield "duration_ns", model["duration_stats"]["p95"] * model[
                    "duration_stats"
                ]["count"], event


def parse_io(item, experiment, exp, counts, filename):
    """
    Parse IO. This is more qualitiative.
    """
    if experiment not in counts:
        counts[experiment] = {}

    if exp.size not in counts[experiment]:
        counts[experiment][exp.size] = {}

    if exp.env_type not in counts[experiment][exp.size]:
        counts[experiment][exp.size][exp.env_type] = {}

    # Split on start of list
    try:
        items = json.loads(item.split("proceeding to print.", 1)[1])
    except:
        print(f"Issue parsing {filename}")
        return counts
    for d in items:

        filename = d["filename"]

        if filename.startswith("/var/lib/kubelet/pods"):
            filename = "/var/lib/kubelet/pods"
        elif filename.startswith("/sys/bus/cpu/devices"):
            filename = "/sys/bus/cpu/devices"
        elif "/proc" in filename and "/stat" in filename:
            filename = "/proc/<pid>/stat"
        elif "/proc" in filename and "/net/dev" in filename:
            filename = "/proc/<pid>/net/dev"
        elif "/proc" in filename and "/fd" in filename:
            filename = "/proc/<pid>/fd"
        elif "/proc" in filename and "/limits" in filename:
            filename = "/proc/<pid>/limits"
        elif "/sys/devices/system/cpu" in filename and filename.endswith("id"):
            filename = "/sys/devices/system/cpu/<cpu>/cache/<index>/id"
        elif "/sys/devices/system/cpu" in filename and filename.endswith("level"):
            filename = "/sys/devices/system/cpu/<cpu>/cache/<index>/level"
        elif "/sys/devices/system/cpu" in filename and filename.endswith(
            "shared_cpu_map"
        ):
            filename = "/sys/devices/system/cpu/<cpu>/cache/<index>/shared_cpu_map"
        elif "/sys/devices/system/cpu" in filename and filename.endswith("size"):
            filename = "/sys/devices/system/cpu/<cpu>/cache/<index>/size"
        elif "/sys/devices/system/cpu" in filename and filename.endswith("type"):
            filename = "/sys/devices/system/cpu/<cpu>/cache/<index>/type"
        elif "/sys/devices/system/cpu" in filename:
            filename = "/sys/devices/system/cpu/<cpu>/cache"
        elif "/sys/dev/block" in filename:
            filename = "/sys/dev/block"
        elif filename.startswith("/var/log/pods"):
            filename = "/var/log/pods"
        elif filename.startswith("/etc/kubernetes"):
            filename = "/etc/kubernetes"
        elif filename.startswith("/usr/local/pancakes/lib/openmpi"):
            filename = "/usr/local/pancakes/lib/openmpi"

        # Flux running assets
        elif (
            filename.startswith("/mnt/flux/view/run/flux")
            and "vader_segment" in filename
        ):
            filename = "/mnt/flux/view/run/flux/<vader_segment>"

        elif "/sys/fs/cgroup/kubepods.slice" in filename:
            filename = "/sys/fs/cgroup/kubepods.slice/"
        elif "/proc/self/task" in filename:
            filename = "/proc/self/task"
        elif "/proc" in filename and "oom_score_adj" in filename:
            filename = "/proc/<pid>/oom_score_adj"
        elif "/sys/bus/pci/devices" in filename:
            filename = os.path.dirname(filename)
        elif "/sys/devices/system/node" in filename and "topology/core_id" in filename:
            filename = "/sys/devices/system/node/<node>/<cpu>/topology/core_id"
        elif "/run/containerd/io.containerd.grpc.v1.cri/containers" in filename:
            filename = "/run/containerd/io.containerd.grpc.v1.cri/containers/"
        elif "/var/lib/containerd/io.containerd.grpc.v1.cri/containers" in filename:
            filename = "/var/lib/containerd/io.containerd.grpc.v1.cri/containers"
        elif (
            "/sys/devices/system/node" in filename
            and "/hugepages" in filename
            and "nr_hugepages" in filename
        ):
            hugepage_size = os.path.basename(os.path.dirname(filename))
            filename = f"/sys/devices/system/node/<node>/hugepages/{hugepage_size}/nr_hugepages"

        elif "/var/run/netns/cni" in filename:
            filename = "/var/run/netns/cni"
        elif ".kube/cache/discovery" in filename:
            filename = ".kube/cache/discovery"
        elif ".kube/cache/http/.diskv-temp" in filename:
            filename = ".kube/cache/http/.diskv-temp/"
        elif "/var/log/journal" in filename:
            filename = "/var/log/journal"
        elif "/proc" in filename and filename.endswith("cgroup"):
            filename = "/proc/<pid>/cgroup"
        elif "/proc" in filename and filename.endswith("/cmdline"):
            filename = "/proc/<pid>/cmdline"
        elif "/proc" in filename and filename.endswith("/current"):
            filename = "/proc/<pid>/current"
        elif "/proc" in filename and filename.endswith("/comm"):
            filename = "/proc/<pid>/comm"
        elif "/proc" in filename and filename.endswith("/loginuid"):
            filename = "/proc/<pid>/loginuid"
        elif "/proc" in filename and filename.endswith("/sessionid"):
            filename = "/proc/<pid>/sessionid"
        elif "/mnt/flux/view/run/flux/jobtmp" in filename:
            filename = "/mnt/flux/view/run/flux/jobtmp-XX"
        elif "/run/systemd/journal/streams" in filename:
            filename = "/run/systemd/journal/streams"
        elif "/tmp/runc-process" in filename:
            filename = "/tmp/runc-process"
        elif "/kube-dns-config/" in filename:
            filename = "/kube-dns-config/"
        elif "/etc/k8s/dns/dnsmasq-nanny" in filename:
            filename = "/etc/k8s/dns/dnsmasq-nanny"
        elif "/sys/bus/node/devices" in filename and "distance" in filename:
            filename = "/sys/bus/node/devices/<node>/distance"
        elif "/sys/devices/system/node" in filename and "meminfo" in filename:
            filename = "/sys/devices/system/node/<node>/meminfo"
        elif "/proc" in filename and "/ns/ipc" in filename:
            filename = "/proc/<pid>/ns/ipc"
        elif "/proc" in filename and "/ns/mnt" in filename:
            filename = "/proc/<pid>/ns/mnt"
        elif "/proc" in filename and "/ns/net" in filename:
            filename = "/proc/<pid>/ns/net"
        elif "/proc" in filename and "/ns/pid" in filename:
            filename = "/proc/<pid>/ns/pid"
        elif "/proc" in filename and "/ns/uts" in filename:
            filename = "/proc/<pid>/ns/uts"
        elif "/run/containerd/runc" in filename:
            filename = "/run/containerd/runc"
        elif "/usr/lib/tmpfiles.d" in filename:
            filename = "/usr/lib/tmpfiles.d"
        elif "/tmp/systemd-private" in filename:
            filename = "/tmp/systemd-private"
        elif "/run/systemd/transient/kubepods-besteffort.slice.d" in filename:
            filename = "/run/systemd/transient/kubepods-besteffort.slice.d"
        elif "/run/systemd/transient/kubepods-burstable.slice.d" in filename:
            filename = "/run/systemd/transient/kubepods-burstable.slice.d"
        elif "/sys/bus/node/devices" in filename and "cpumap" in filename:
            filename = "/sys/bus/node/devices/<node>/cpumap"
        elif "/sys/fs/cgroup/system.slice/systemd-tmpfiles-clean.service" in filename:
            filename = "/sys/fs/cgroup/system.slice/systemd-tmpfiles-clean.service"
        elif "/sys/bus/node/devices" in filename and "meminfo" in filename:
            filename = "/sys/bus/node/devices/<node>/meminfo"
        elif "/sys/fs/cgroup/system.slice/kube-logrotate.service" in filename:
            filename = "/sys/fs/cgroup/system.slice/kube-logrotate.service"
        elif "/sys/fs/cgroup/system.slice/motd-news.service" in filename:
            filename = "/sys/fs/cgroup/system.slice/motd-news.service"
        elif "/sys/fs/cgroup/system.slice/gke-node-reg-checker.service" in filename:
            filename = "/sys/fs/cgroup/system.slice/gke-node-reg-checker.service"
        elif "kube-system_metrics-server" in filename:
            filename = "kube-system_metrics-server"
        elif "/var/lib/cni/networks/k8s-pod-network" in filename:
            filename = "/var/lib/cni/networks/k8s-pod-network"
        elif "/prometheus/data" in filename:
            filename = "/prometheus/data"
        elif "/run/udev/data" in filename:
            filename = "/run/udev/data"
        elif "/usr/libexec/flux" in filename:
            filename = "/usr/libexec/flux"
        elif "/usr/lib/python" in filename:
            filename = "/usr/lib/python"
        elif "/usr/local/lib/python" in filename:
            filename = "/usr/local/lib/python"
        elif "/usr/lib/flux/python" in filename:
            filename = "/usr/lib/flux/python"
        elif "/tmp/apt-key-gpghome" in filename:
            filename = "/tmp/apt-key-gpghome"
        elif "default_lammps" in filename:
            filename = "default_lammps"
        elif (
            "/sys/fs/cgroup/system.slice/gce-workload-cert-refresh.service" in filename
        ):
            filename = "/sys/fs/cgroup/system.slice/gce-workload-cert-refresh.service"
        elif (
            "/sys/devices/system/node" in filename
            and "topology/physical_package_id" in filename
        ):
            filename = (
                "/sys/devices/system/node/<node>/<cpu>/topology/physical_package_id"
            )
        elif "/var/lib/apt/lists" in filename:
            filename = "/var/lib/apt/lists"
        elif "/tmp/apt" in filename:
            filename = "/tmp/apt"
        elif "/etc/dpkg/dpkg" in filename:
            filename = "/etc/dpkg/dpkg"
        elif "/etc/apt" in filename:
            filename = "/etc/apt"
        elif "/run/containerd/io.containerd.runtime.v2.task" in filename:
            filename = "/run/containerd/io.containerd.runtime.v2.task"
        elif (
            "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots"
            in filename
        ):
            filename = (
                "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs/snapshots"
            )
        elif "/tmp/clearsigned" in filename:
            filename = "/tmp/clearsigned"
        elif "/sys/devices/system/node" in filename and "distance" in filename:
            filename = "/sys/devices/system/node/<node>/distance"
        elif "/sys/bus/node/devices" in filename and "hugepages" in filename:
            filename = "/sys/bus/node/devices/<node>/hugepages"
        elif "/sys/devices/system/node" in filename and filename.endswith("hugepages/"):
            filename = "/sys/devices/system/node/<node>/hugepages/"
        elif filename.startswith("/tmp/ompi.lammps"):
            filename = "/tmp/ompi.lammps"

        # These are hidden digest directories?
        if len(filename) == 65 and filename.startswith("."):
            print(f"Skipping digest {filename}")
            continue

        elif (len(filename) == 64 or len(filename) == 32) and os.sep not in filename:
            print(f"Skipping digest {filename}")
            continue

        # If we get an integer, don't include it
        try:
            int(filename)
            continue
        except:
            pass
        if d["command"] not in counts[experiment][exp.size][exp.env_type]:
            counts[experiment][exp.size][exp.env_type][d["command"]] = {}
        if filename not in counts[experiment][exp.size][exp.env_type][d["command"]]:
            counts[experiment][exp.size][exp.env_type][d["command"]][filename] = 0
        counts[experiment][exp.size][exp.env_type][d["command"]][filename] += d[
            "open_count"
        ]
    return counts


# {'event': 'OPEN',
# 'command': 'kubelet',
# 'retval': 33,
# 'ts_sec': 1857.259043102,
# 'tgid': 1701734764,
# 'tid': 3792,
# 'ppid': 13411,
# 'cgroup_id': 7237126475108805679,
# 'filename': '/var/log/pods/default_lammps-0-f6vcz_f27a413b-4d60-44d9-b282-c6cd399424e8/bcc-monitor'}


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
    io_counts = {}

    # It's important to just parse raw data once, and then use intermediate
    total = len(files) - 1
    for i, filename in enumerate(files):
        if "test" in filename:
            continue
        print(f"{i}/{total}", end="\r")
        basename = os.path.basename(filename)
        add_ebpf_result(indir, filename, io_counts)

    # save counts (note we didn't run this here)
    for experiment, sizes in io_counts.items():
        for size, env_types in sizes.items():
            for env_type, commands in env_types.items():
                for command, filepaths in commands.items():
                    for filepath, count in filepaths.items():
                        db.add_result(
                            "google",
                            "open-close",
                            experiment,
                            env_type,
                            size,
                            filepath,
                            count,
                            command=command,
                        )

    db.close()
    print(json.dumps(total_counts))
    print("Done parsing lammps eBPF results!")


if __name__ == "__main__":
    main()
