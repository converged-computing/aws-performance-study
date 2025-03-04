# GKE GPU Experiment Size 2

## Experiment

### 1. Setup

```bash
GOOGLE_PROJECT=llnl-flux

# 2 nodes, 1 GPU each
GPUS=1
NODES=2
INSTANCE=n1-standard-8

# OR 1 node, 2 GPU
GPUS=2
NODES=1
INSTANCE=n1-standard-8

time gcloud container clusters create gpu-cluster \
    --threads-per-core=1 \
    --accelerator type=nvidia-tesla-v100,count=$GPUS,gpu-driver-version=latest \
    --num-nodes=$NODES \
    --machine-type=$INSTANCE \
    --enable-gvnic \
    --network=mtu9k \
    --system-config-from-file=./system-config.yaml \
    --region=us-central1-a \
    --project=${GOOGLE_PROJECT} 
```

Sanity check installed on all nodes

```bash
kubectl get nodes -o json | jq .items[].status.allocatable
kubectl get nodes -o json | jq .items[].status.allocatable | grep nvidia | wc -l
```
```
2
```

Install the Flux Operator.

```bash
kubectl apply -f https://raw.githubusercontent.com/flux-framework/flux-operator/refs/heads/main/examples/dist/flux-operator.yaml
```

## Applications

For each application, since this is interactive after apply we can shell in as follows:

```bash
kubectl exec -it flux-sample-0-xxx -- bash
. /mnt/flux/flux-view.sh 

# Connect to lead broker
flux proxy $fluxsocket bash

# See resources
flux resource list

# Load and verify fluxion
bash /mnt/flux/load-fluxion.sh
flux module list
```

### NCCL Tests

```bash
kubectl apply -f ./crd/nccl-tests.yaml
```

Once connected to flux:

```bash
# This is for one node
./nccl-tests/build/all_reduce_perf -b 8 -e 128M -f 2 -g 1
```
```console
# nThread 1 nGpus 1 minBytes 8 maxBytes 134217728 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    126 on flux-sample-0 device  0 [0000:00:04] Tesla V100-SXM2-16GB
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1     4.56    0.00    0.00      0     0.14    0.06    0.00      0
          16             4     float     sum      -1     3.72    0.00    0.00      0     0.13    0.12    0.00      0
          32             8     float     sum      -1     3.68    0.01    0.00      0     0.13    0.24    0.00      0
          64            16     float     sum      -1     4.03    0.02    0.00      0     0.13    0.48    0.00      0
         128            32     float     sum      -1     3.63    0.04    0.00      0     0.13    0.96    0.00      0
         256            64     float     sum      -1     3.60    0.07    0.00      0     0.14    1.87    0.00      0
         512           128     float     sum      -1     4.53    0.11    0.00      0     0.15    3.31    0.00      0
        1024           256     float     sum      -1     4.04    0.25    0.00      0     0.15    6.66    0.00      0
        2048           512     float     sum      -1     3.99    0.51    0.00      0     0.15   13.29    0.00      0
        4096          1024     float     sum      -1     4.05    1.01    0.00      0     0.15   27.20    0.00      0
        8192          2048     float     sum      -1     4.11    2.00    0.00      0     0.15   53.72    0.00      0
       16384          4096     float     sum      -1     3.98    4.12    0.00      0     0.15  107.37    0.00      0
       32768          8192     float     sum      -1     3.94    8.31    0.00      0     0.15  213.40    0.00      0
       65536         16384     float     sum      -1     4.44   14.75    0.00      0     0.13  489.07    0.00      0
      131072         32768     float     sum      -1     3.92   33.47    0.00      0     0.13  974.88    0.00      0
      262144         65536     float     sum      -1     3.68   71.28    0.00      0     0.13  1965.83    0.00      0
      524288        131072     float     sum      -1     4.10  127.84    0.00      0     0.14  3876.44    0.00      0
     1048576        262144     float     sum      -1     5.81  180.33    0.00      0     0.13  7839.82    0.00      0
     2097152        524288     float     sum      -1     8.39  249.95    0.00      0     0.13  15893.54    0.00      0
     4194304       1048576     float     sum      -1    14.12  296.95    0.00      0     0.13  31289.10    0.00      0
     8388608       2097152     float     sum      -1    25.06  334.80    0.00      0     0.13  62578.20    0.00      0
    16777216       4194304     float     sum      -1    46.27  362.61    0.00      0     0.13  125390.25    0.00      0
    33554432       8388608     float     sum      -1    88.58  378.80    0.00      0     0.13  250499.68    0.00      0
    67108864      16777216     float     sum      -1    173.5  386.90    0.00      0     0.15  435489.06    0.00      0
   134217728      33554432     float     sum      -1    342.7  391.63    0.00      0     0.14  972944.75    0.00      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 0 
#
```

Here are options:

```bash
root@flux-sample-0:/opt# ./nccl-tests/build/all_reduce_perf --help                
USAGE: all_reduce_perf 
	[-t,--nthreads <num threads>] 
	[-g,--ngpus <gpus per thread>] 
	[-b,--minbytes <min size in bytes>] 
	[-e,--maxbytes <max size in bytes>] 
	[-i,--stepbytes <increment size>] 
	[-f,--stepfactor <increment factor>] 
	[-n,--iters <iteration count>] 
	[-m,--agg_iters <aggregated iteration count>] 
	[-w,--warmup_iters <warmup iteration count>] 
	[-N,--run_cycles <cycle count> run & print each cycle (default: 1; 0=infinite)] 
	[-p,--parallel_init <0/1>] 
	[-c,--check <check iteration count>] 
	[-o,--op <sum/prod/min/max/avg/mulsum/all>] 
	[-d,--datatype <nccltype/all>] 
	[-r,--root <root>] 
	[-z,--blocking <0/1>] 
	[-y,--stream_null <0/1>] 
	[-T,--timeout <time in seconds>] 
	[-G,--cudagraph <num graph launches>] 
	[-C,--report_cputime <0/1>] 
	[-a,--average <0/1/2/3> report average iteration time <0=RANK0/1=AVG/2=MIN/3=MAX>] 
	[-R,--local_register <1/0> enable local buffer registration on send/recv buffers (default: disable)] 
	[-h,--help]
```

> Run MPI processes on nodes with 1 GPU each, for a total of 2 GPUs spread across 2 nodes : (NB: The nccl-tests binaries must be compiled with MPI=1 for this case)

```bash
flux run -n 2 -N 2 -g 1 ./nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
```
```console
 flux run -n 2 -N 2 -g 1 -o gpu-affinity=per-task -o cpu-affinity=per-task ./nccl-tests/build/all_reduce_perf -b 8 -e 8G -f 2 -g 1
# nThread 1 nGpus 1 minBytes 8 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid    167 on flux-sample-0 device  0 [0000:00:04] Tesla V100-SXM2-16GB
#  Rank  1 Group  0 Pid     91 on flux-sample-1 device  0 [0000:00:04] Tesla V100-SXM2-16GB
#
# Reducing maxBytes to 5284866730 due to memory limitation
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
           8             2     float     sum      -1    105.3    0.00    0.00      0    110.1    0.00    0.00      0
          16             4     float     sum      -1    112.5    0.00    0.00      0    101.0    0.00    0.00      0
          32             8     float     sum      -1    117.8    0.00    0.00      0    125.9    0.00    0.00      0
          64            16     float     sum      -1    122.9    0.00    0.00      0    121.4    0.00    0.00      0
         128            32     float     sum      -1    125.1    0.00    0.00      0    126.4    0.00    0.00      0
         256            64     float     sum      -1    132.5    0.00    0.00      0    128.3    0.00    0.00      0
         512           128     float     sum      -1    125.1    0.00    0.00      0    126.1    0.00    0.00      0
        1024           256     float     sum      -1    125.9    0.01    0.01      0    136.4    0.01    0.01      0
        2048           512     float     sum      -1    143.2    0.01    0.01      0    140.2    0.01    0.01      0
        4096          1024     float     sum      -1    149.2    0.03    0.03      0    141.5    0.03    0.03      0
        8192          2048     float     sum      -1    161.5    0.05    0.05      0    155.4    0.05    0.05      0
       16384          4096     float     sum      -1    270.3    0.06    0.06      0    264.8    0.06    0.06      0
       32768          8192     float     sum      -1    300.5    0.11    0.11      0    304.8    0.11    0.11      0
       65536         16384     float     sum      -1    393.7    0.17    0.17      0    414.1    0.16    0.16      0
      131072         32768     float     sum      -1    488.1    0.27    0.27      0    540.9    0.24    0.24      0
      262144         65536     float     sum      -1   1815.0    0.14    0.14      0    719.4    0.36    0.36      0
      524288        131072     float     sum      -1   1118.8    0.47    0.47      0   1122.2    0.47    0.47      0
     1048576        262144     float     sum      -1   1793.0    0.58    0.58      0   1795.9    0.58    0.58      0
     2097152        524288     float     sum      -1   4159.0    0.50    0.50      0   3161.5    0.66    0.66      0
     4194304       1048576     float     sum      -1   5952.6    0.70    0.70      0   5943.6    0.71    0.71      0
     8388608       2097152     float     sum      -1    12078    0.69    0.69      0    12913    0.65    0.65      0
    16777216       4194304     float     sum      -1    23370    0.72    0.72      0    23078    0.73    0.73      0
    33554432       8388608     float     sum      -1    57453    0.58    0.58      0    56435    0.59    0.59      0
    67108864      16777216     float     sum      -1    90968    0.74    0.74      0   106153    0.63    0.63      0
   134217728      33554432     float     sum      -1   189995    0.71    0.71      0   210266    0.64    0.64      0
   268435456      67108864     float     sum      -1   359514    0.75    0.75      0   377029    0.71    0.71      0
   536870912     134217728     float     sum      -1   722816    0.74    0.74      0   723954    0.74    0.74      0
  1073741824     268435456     float     sum      -1  1447234    0.74    0.74      0  1446217    0.74    0.74      0

  2147483648     536870912     float     sum      -1  2892029    0.74    0.74      0  2899726    0.74    0.74      0
...
```

The above takes a long time on V100. I'm not timing here because the scale is so different, but it's a longer test, generally (minutes).

### DeepGEMM

```bash
kubectl apply -f ./crd/deepgemm.yaml
```

```bash
# Test JIT compilation
python tests/test_jit.py

# Test all GEMM implements (normal, contiguous-grouped and masked-grouped)
python tests/test_core.py
```

The test JIT works, but the test core does not - it segfaults.

### AI Benchmarks

```bash
kubectl apply -f ./crd/ai-benchmarks.yaml
```

```bash
unset PYTHONPATH
```

Here are the options:

```console
usage: ai-benchmark [-h] [-c] [-C CPU_CORES] [-b INTRA_THREADS] [-B INTER_THREADS] [-T {0,1}] [-i {0,1}]
                    [-m {0,1}] [-v {0,1,2,3}] [-p {normal,high,dry}] [-s SEED]
                    [-t {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...]]
                    [-j]

options:
  -h, --help            show this help message and exit
  -c, --use-cpu         Run the tests on CPUs (if tensorflow-gpu is installed)
  -C CPU_CORES, --cpu-cores CPU_CORES
                        Number of CPU cores to use.
  -b INTRA_THREADS, --intra-threads INTRA_THREADS
                        inter_op_parallelism_threads
  -B INTER_THREADS, --inter-threads INTER_THREADS
                        intra_op_parallelism_threads
  -T {0,1}, --run-training {0,1}
                        Run training benchmark
  -i {0,1}, --run-inference {0,1}
                        Run inference benchmark
  -m {0,1}, --run-micro {0,1}
                        Run micro benchmark
  -v {0,1,2,3}, --verbose {0,1,2,3}
                        0: silent, 1: short summary, 2: more info, 3: TF logs
  -p {normal,high,dry}, --precision {normal,high,dry}
                        normal or high, if high is selected, the benchmark will execute 10 times more runs for
                        each test. dry do not run any iterations.
  -s SEED, --seed SEED  Random seed
  -t {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...], --test-ids {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} [{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19} ...]
                        Select test by ID, all by default
  -j, --json            Output results as JSON
```

And the run, normal precision. Note we likely want to output `--json`

<details>

<summary>benchmark results</summary>

```bash
$ ai-benchmark --precision normal
2025-03-04 03:06:59.771053: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1741057619.793003     137 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1741057619.800646     137 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
2025-03-04 03:06:59.823156: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
>>   AI-Benchmark - 0.1.4.cm
>>   Let the AI Games begin
I0000 00:00:1741057622.939179     137 gpu_device.cc:2022] Created device /device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
I0000 00:00:1741057622.940428     137 gpu_device.cc:2022] Created device /device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
I0000 00:00:1741057622.941150     137 gpu_device.cc:2022] Created device /device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
*  TF Version: 2.18.0
*  Platform: Linux-6.6.56+-x86_64-with-glibc2.39
*  CPU: N/A
*  CPU RAM: 29 GB
*  GPU/0: Tesla V100-SXM2-16GB
*  GPU RAM: 14.4 GB
*  CUDA Version: 12.8
*  CUDA Build: V12.8.61
The benchmark is running...
The tests might take up to 20 minutes
Please don't interrupt the script

1/19. MobileNet-V2

I0000 00:00:1741057624.014141     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1741057624.197567     137 mlir_graph_optimization_pass.cc:401] MLIR V1 optimization pass is not enabled
I0000 00:00:1741057625.423800     164 cuda_dnn.cc:529] Loaded cuDNN version 90700
1.1 - inference | batch=50, size=224x224: 60.9 ± 0.7 ms
1.2 - training  | batch=50, size=224x224: 75.3 ± 0.5 ms

2/19. Inception-V3

I0000 00:00:1741057644.368547     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
2.1 - inference | batch=20, size=346x346: 49.2 ± 0.4 ms
2.2 - training  | batch=20, size=346x346: 140 ± 4 ms

3/19. Inception-V4

I0000 00:00:1741057662.866599     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
3.1 - inference | batch=10, size=346x346: 43.3 ± 1.8 ms
3.2 - training  | batch=10, size=346x346: 146.0 ± 0.7 ms

4/19. Inception-ResNet-V2

I0000 00:00:1741057680.236435     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
4.1 - inference | batch=10, size=346x346: 55.2 ± 0.5 ms
4.2 - training  | batch=8, size=346x346: 150.4 ± 0.6 ms

5/19. ResNet-V2-50

I0000 00:00:1741057704.719802     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
5.1 - inference | batch=10, size=346x346: 26.6 ± 0.5 ms
5.2 - training  | batch=10, size=346x346: 74.7 ± 0.5 ms

6/19. ResNet-V2-152

I0000 00:00:1741057715.471755     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
6.1 - inference | batch=10, size=256x256: 35.0 ± 0.5 ms
6.2 - training  | batch=10, size=256x256: 108.6 ± 0.5 ms

7/19. VGG-16

I0000 00:00:1741057734.845076     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
7.1 - inference | batch=20, size=224x224: 50.0 ± 0.6 ms
2025-03-04 03:08:59.778108: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 411041792 exceeds 10% of free system memory.
2025-03-04 03:09:00.108552: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 411041792 exceeds 10% of free system memory.
7.2 - training  | batch=2, size=224x224: 69.6 ± 0.5 ms

8/19. SRCNN 9-5-5

I0000 00:00:1741057743.695934     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
8.1 - inference | batch=10, size=512x512: 93.1 ± 1.1 ms
8.2 - inference | batch=1, size=1536x1536: 80.9 ± 2.8 ms
8.3 - training  | batch=10, size=512x512: 213 ± 7 ms

9/19. VGG-19 Super-Res

I0000 00:00:1741057779.408111     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
9.1 - inference | batch=10, size=256x256: 46.4 ± 1.1 ms
9.2 - inference | batch=1, size=1024x1024: 70.1 ± 0.9 ms
9.3 - training  | batch=10, size=224x224: 144.3 ± 0.5 ms

10/19. ResNet-SRGAN

I0000 00:00:1741057803.269742     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
10.1 - inference | batch=10, size=512x512: 101.0 ± 0.8 ms
10.2 - inference | batch=1, size=1536x1536: 85.5 ± 2.6 ms
10.3 - training  | batch=5, size=512x512: 95.8 ± 0.4 ms

11/19. ResNet-DPED

I0000 00:00:1741057827.454428     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
11.1 - inference | batch=10, size=256x256: 66.6 ± 0.5 ms
11.2 - inference | batch=1, size=1024x1024: 127.3 ± 0.5 ms

11.3 - training  | batch=15, size=128x128: 99.0 ± 0.2 ms

12/19. U-Net

I0000 00:00:1741057857.934857     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
12.1 - inference | batch=4, size=512x512: 122.2 ± 0.8 ms
12.2 - inference | batch=1, size=1024x1024: 122 ± 1 ms
12.3 - training  | batch=4, size=256x256: 112.4 ± 0.5 ms

13/19. Nvidia-SPADE

I0000 00:00:1741057881.003645     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
13.1 - inference | batch=5, size=128x128: 62.7 ± 0.5 ms
13.2 - training  | batch=1, size=128x128: 87.6 ± 0.7 ms

14/19. ICNet

I0000 00:00:1741057897.034466     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
14.1 - inference | batch=5, size=1024x1536: 181 ± 1 ms
2025-03-04 03:11:50.353585: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 188743680 exceeds 10% of free system memory.
2025-03-04 03:11:50.513788: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 188743680 exceeds 10% of free system memory.
2025-03-04 03:11:50.668651: W external/local_xla/xla/tsl/framework/cpu_allocator_impl.cc:83] Allocation of 188743680 exceeds 10% of free system memory.
14.2 - training  | batch=10, size=1024x1536: 627 ± 3 ms

15/19. PSPNet

I0000 00:00:1741057939.418302     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
15.1 - inference | batch=5, size=720x720: 256 ± 5 ms
15.2 - training  | batch=1, size=512x512: 90.8 ± 0.4 ms

16/19. DeepLab

I0000 00:00:1741057960.318575     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
16.1 - inference | batch=2, size=512x512: 76.3 ± 0.5 ms
16.2 - training  | batch=1, size=384x384: 78.9 ± 0.6 ms

17/19. Pixel-RNN

I0000 00:00:1741057975.681329     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
I0000 00:00:1741057976.011151     137 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
2025-03-04 03:13:18.454828: W tensorflow/c/c_api.cc:305] Operation '{name:'conv2d_out_logits/biases/Adam_1/Assign' id:47255 op device:{requested: '', assigned: ''} def:{{{node conv2d_out_logits/biases/Adam_1/Assign}} = AssignVariableOp[_has_manual_control_dependencies=true, dtype=DT_FLOAT, validate_shape=false](conv2d_out_logits/biases/Adam_1, conv2d_out_logits/biases/Adam_1/Initializer/zeros)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.
17.1 - inference | batch=50, size=64x64: 385 ± 19 ms
17.2 - training  | batch=10, size=64x64: 1633 ± 21 ms

# 18/19. LSTM-Sentiment (removed, didn't work)

19/1. GNMT-Translation

I0000 00:00:1741058211.929931     389 gpu_device.cc:2022] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 14785 MB memory:  -> device: 0, name: Tesla V100-SXM2-16GB, pci bus id: 0000:00:04.0, compute capability: 7.0
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1741058212.116317     389 mlir_graph_optimization_pass.cc:401] MLIR V1 optimization pass is not enabled
2025-03-04 03:16:52.541430: W tensorflow/c/c_api.cc:305] Operation '{name:'index_to_string/table_init' id:13 op device:{requested: '', assigned: ''} def:{{{node index_to_string/table_init}} = InitializeTableFromTextFileV2[_has_manual_control_dependencies=true, delimiter="\t", key_index=-1, offset=0, value_index=-2, vocab_size=-1](index_to_string, index_to_string/table_init/asset_filepath)}}' was changed by setting attribute after it was run by a session. This mutation will have no effect, and will trigger an error in the future. Either don't modify nodes after running them or create a new session.
19.1 - inference | batch=1, size=1x20: 112 ± 3 ms
Device Inference Score: 17202
Device Training Score: 0
Device AI Score: 17202
For more information and results, please visit http://ai-benchmark.com/alpha
```

</details>

### FMWork

> Haven't tested yet, but the data is pulled to image. It's big.

```bash
kubectl apply -f ./crd/fmwork.yaml
```

### GPU Fryer

This seems to work, but we don't need flux. We should verify one pod per node and remove flux. This could be run as an indexed job or similar.

```bash
root@flux-sample-0:/# gpu-fryer 120
Detected GPU #0: "Tesla V100-SXM2-16GB" (88ad3b91-09d7-fbcd-2642-036343e9b3f7)
Detected GPU #1: "Tesla V100-SXM2-16GB" (8bf17fe2-b187-79ec-9269-2c0480351b8e)
Creating random matrices
Matrices created
GPU #0: Using 1293 MB out of 1436 MB
GPU #1: Using 1293 MB out of 1436 MB
1023 (1124800 Gflops/s) - 0 (0 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 1026 (1128098 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
5 (5497 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
5 (5497 Gflops/s) - 5 (5497 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 58°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 59°C - 56°C | Throttling: None - None
5 (5497 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 59°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 59°C - 56°C | Throttling: None - None
6 (6597 Gflops/s) - 5 (5497 Gflops/s) | Temperatures: 59°C - 57°C | Throttling: None - None
...
5 (5497 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 61°C - 58°C | Throttling: None - None
6 (6597 Gflops/s) - 6 (6597 Gflops/s) | Temperatures: 61°C - 58°C | Throttling: None - None
GPU #0:   6208 Gflops/s (min: 5497.56, max: 6597.07, dev: 6208.45)
         Temperature: 60.43°C (min: 58.00, max: 61.00)
         Throttling HW: false, Thermal SW: false, Thermal HW: false
GPU #1:   6332 Gflops/s (min: 5497.56, max: 6597.07, dev: 6331.67)
         Temperature: 57.80°C (min: 56.00, max: 59.00)
         Throttling HW: false, Thermal SW: false, Thermal HW: false
All GPUs seem healthy
Freeing GPUs...
```

### GPU Burn

> Doesn't work with V100 

It says the driver build is not compatible. I tried COMPUTE=70 and setting other vars, but it didn't work. I think it might need an older driver (meaning container base).

### FMPerf

```
. /fmperf/environment.sh
python -m fmperf.loadgen.generate-input
```

With or without the model, it requires an [inference server](https://github.com/fmperf-project/fmperf/blob/b2e216c4b63066150d4b1575557713b0295f3f0d/.env.example#L49-L52) to be running, which is a bit complex for what we want.

## Thoughts

For many of these that are intended for single nodes, it's not clear to me why we would need such a large cluster. Especially for Tranium that can't run HPC apps. We might want to consider a smaller Trainium cluster. Maybe just the one MPI benchmark that is multi-node would make sense to scale out.

## Cleanup 

```bash
time gcloud container clusters delete gpu-cluster --region us-central1-a
```
