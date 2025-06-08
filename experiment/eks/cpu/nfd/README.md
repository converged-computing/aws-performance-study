# Node Feature Discovery Data

Here we want to extract NFD labels across different instance types. We can then design different containers to run lammps on the nodes (or some other app) and determine if we can create compatibility manifests for CPU. Since we want to keep costs down, we are going to stick to small instance types and simple ways to assess. Create the cluster.

```bash
eksctl create cluster --config-file ./eks-config.yaml
aws eks update-kubeconfig --region us-east-1 --name nfd-cluster
```

Install node feature discovery:

```bash
kubectl apply -k https://github.com/kubernetes-sigs/node-feature-discovery/deployment/overlays/default?ref=v0.17.3
```

Export the NFD features, and that's it!

```bash
kubectl get nodes -o json | jq '.items[].metadata.labels' > node-features.json
kubectl get nodes -o json > nodes.json
```

## Clean Up

When you are done:

```bash
eksctl delete cluster --config-file ./eks-config.yaml --wait
```
