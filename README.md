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

...
