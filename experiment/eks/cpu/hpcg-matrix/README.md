# Optimization Test

Let's run hpcg on different optimizations. For instance selection I am aiming for $2-3, in the lower range if allowed.

Create the cluster.

```bash
# Choose an instance config
eksctl create cluster --config-file ./cfg/eks-config-hpc6a.48xlarge.yaml
eksctl create cluster --config-file ./cfg/eks-config-t3-2xlarge.yaml
eksctl create cluster --config-file ./cfg/eks-config-m6a.12xlarge.yaml
eksctl create cluster --config-file ./cfg/eks-config-m6i.12xlarge.yaml
eksctl create cluster --config-file ./cfg/eks-config-t3a-2xlarge.yaml
eksctl create cluster --config-file ./cfg/eks-config-inf2.8xlarge.yaml


aws eks update-kubeconfig --region us-east-2 --name hpcg-test
```

Install node feature discovery (so we get node metadata for the hpc6a instance type):

```bash
kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.17.3
```

View the NFD pods, and features for your node(s):

```bash
/kubectl -n node-feature-discovery get all
kubectl get nodes -o json | jq '.items[].metadata.labels'
```

For the tests below, install the Flux Operator.

```bash
kubectl apply -f https://raw.githubusercontent.com/flux-framework/flux-operator/refs/heads/main/examples/dist/flux-operator.yaml
```

## Applications

Note that you'll need to clone [converged-computing/flux-apps-helm](https://github.com/converged-computing/flux-apps-helm).

## CPU Applications

### Testing hpcg on hpc6a

- For 40 40 40 / 30 seconds
  - 42 seconds without eBPF
  - 40.82 with single eBPF sidecar
  - 45 seconds with 2 eBPF sidecars

- For 32 32 32 / 15 seconds
  - 2 eBPF sidecars (22 seconds)
  - 21.47 seconds without eBPF

We chose a problem size of the above to get an approximate running time of 30 seconds. With ebpf containers and without turned to be similar performance.

### hpcg on hpc6a

These runs should take ~190 minutes, which will be about ~10. We want to see the CPU go up to 100% across processes. We want 3 iterations to get a sample of 3x.

```bash
mkdir -p ./logs/hpcg
helm dependency update hpcg-matrix/
for tag in $(cat ./tags.txt); do
  outdir=./logs/hpcg/$tag
  mkdir -p $outdir
  if [[ -f "$outdir/hpcg.out" ]]; then
    echo "$tag was already run"
    continue
  fi
  helm install \
  --set experiment.nodes=1 \
  --set minicluster.size=1 \
  --set minicluster.tasks=96 \
  --set experiment.tasks=96 \
  --set minicluster.save_logs=true \
  --set monitor.multiple="cpu|futex" \
  --set minicluster.image=ghcr.io/converged-computing/hpcg-matrix:$tag \
  --set monitor.installer="ghcr.io/converged-computing/kernel-header-installer:fedora43" \
  --set experiment.iterations=3 \
  hpcg ./hpcg-matrix
  sleep 3
  time kubectl wait --for=condition=ready pod -l job-name=hpcg --timeout=600s
  pod=$(kubectl get pods -o json | jq  -r .items[0].metadata.name)
  kubectl logs ${pod} -c hpcg -f |& tee $outdir/hpcg.out
  # Let's get the eBPF even if it fails, could be interesting.
  for prog in cpu futex
    do
      kubectl logs ${pod} -c bcc-monitor-$prog -f |& tee $outdir/${prog}.out    
  done
  helm uninstall hpcg
done
```

After this test instance I used the script:

```bash
# bash run-study.sh $instance $tasks
bash run-study.sh t3.2xlarge 4
bash run-study.sh m6a.12xlarge 24
bash run-study.sh m6i.12xlarge 24
bash run-study.sh t3a.2xlarge 4
bash run-study.sh inf2.8xlarge 16
```

## Clean Up

When you are done:

```bash
eksctl delete cluster --config-file ./cfg/eks-config-t3-2xlarge.yaml --wait
eksctl delete cluster --config-file ./cfg/eks-config-m6a.12xlarge.yaml --wait
eksctl delete cluster --config-file ./cfg/eks-config-m6i.12xlarge.yaml --wait
eksctl delete cluster --config-file ./cfg/eks-config-t3a-2xlarge.yaml --wait
```
