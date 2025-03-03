# FMWork

This image has data in it, and I've built in a way so each object is a separate layer and doesn't go over the max. This means the build needs to happen first as the "entire thing" and then a second image built with the data. Here is how to do that:

```
docker build -f Dockerfile.pre -t ghcr.io/converged-computing/metric-fmwork:gpu .

# Bind /tmp/data
docker run -it -v /tmp/data:/tmp/data ghcr.io/converged-computing/metric-fmwork:gpu  bash

# Inside the container, move the data there
mv ./models /tmp/data

# Exit, fix permission and move models directory here
sudo chown -R $(whoami) /tmp/data
mv /tmp/data/models ./models

# Build final container
docker build -t ghcr.io/converged-computing/metric-fmwork:gpu .
docker push ghcr.io/converged-computing/metric-fmwork:gpu
```
