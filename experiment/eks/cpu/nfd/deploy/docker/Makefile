IMAGE_NAME ?= ghcr.io/converged-computing/aws-performance-study
IMAGE_TAG ?= nfd-export
FULL_IMAGE_NAME = $(IMAGE_NAME):$(IMAGE_TAG)
FULL_ARM_NAME = $(IMAGE_NAME):$(IMAGE_TAG)-arm
DOCKERFILE_PATH = Dockerfile
BUILD_CONTEXT = .

# Default target: builds the Docker image
all: build

# Build the Docker image
build:
	@echo "Building Docker image $(FULL_IMAGE_NAME)..."
	docker build \
		-f $(DOCKERFILE_PATH) \
		-t $(FULL_IMAGE_NAME) \
		$(BUILD_CONTEXT)
	@echo "Docker image $(FULL_IMAGE_NAME) built successfully."

build-arm:
	@echo "Building Docker ARM image $(FULL_ARM_NAME)..."
	docker buildx build \
		--platform linux/arm64 \
		-f $(DOCKERFILE_PATH) \
		-t $(FULL_ARM_NAME) \
		$(BUILD_CONTEXT) \
		--load
	@echo "Docker image $(FULL_ARM_NAME) built successfully."

# Push the docker image
push:
	@echo "Pushing image $(FULL_IMAGE_NAME)..."
	docker push $(FULL_IMAGE_NAME)
push-arm:
	@echo "Pushing image $(FULL_ARM_NAME)..."
	docker push $(FULL_ARM_NAME)

# Remove the image (clean with rmi)
clean:
	@echo "Removing Docker image $(FULL_IMAGE_NAME)..."
	docker rmi $(FULL_IMAGE_NAME) || true
	@echo "Docker image $(FULL_IMAGE_NAME) removed (if it existed)."

.PHONY: all build push clean

