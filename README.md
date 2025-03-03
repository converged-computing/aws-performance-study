# AWS Performance Study

We are prototyping a performance study (followup) on AWS that has the following environments:

- AWS Trainium EKS
- AWS Trainium Parallel Cluster
- AWS EKS with p5/p5en.48xlarge
- AWS Parallel Cluster with p5/p5en.48xlarge

## Applications

We are dividing the application space in 32/64 bit. We can run 32 bit apps on 64 but not the other way around. Note that Trainium is only 32 bit.

### 64 bit apps

- amg2023
- kripke
- laghos
- lammps-reax
- mixbench
- osu
- pytorch

### 32 bit apps

- pytorch
- [inference-perf](https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/design.md#metrics-to-collect) looks good, but isn't ready yet
- [fmperf](https://github.com/fmperf-project/fmperf)
- [fmwork](https://github.com/IBM/fmwork)
- [ai-benchmark](https://github.com/cloudmercato/ai-benchmark)
- [hugging-face](https://huggingface.co/docs/transformers/v4.39.1/benchmarks) from Angel, note deprecated
- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
- [gpu-fryer](https://github.com/huggingface/gpu-fryer) Already has a container [ghcr.io/huggingface/gpu-fryer:latest](ghcr.io/huggingface/gpu-fryer:latest). We might want to rebuild if a common base is desired.
- [gpu-burn](https://github.com/wilicc/gpu-burn)
- [DualPipe](https://github.com/deepseek-ai/DualPipe)
- [nccl-tests](https://github.com/NVIDIA/nccl-tests)
