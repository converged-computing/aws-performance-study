# benchmark target: {vllm, tgis}
export TARGET=tgis

# minimum number of input tokens
export MIN_INPUT_TOKENS=500

# maximum number of inputs tokens
export MAX_INPUT_TOKENS=500

# minimum number of output tokens
export MIN_OUTPUT_TOKENS=50

# maximum number of output tokens
export MAX_OUTPUT_TOKENS=50

# fraction of greedy requests
export FRAC_GREEDY=1.0

# number of input requests to generate (virtual users will sample from these)
export SAMPLE_SIZE=1

# Default requests directory
export REQUESTS_DIR=/requests

# requests file
export REQUESTS_FILENAME=sample_requests.json

# results file
export RESULTS_FILENAME=results.json

# filename for combined results
export RESULTS_ALL_FILENAME=results_all.json

# number of virtual users
export NUM_USERS=1

# number of virtual users
export SWEEP_USERS=1,2,4

# experiment duration
export DURATION=30s

# if a request fails, we will backoff for some time
export BACKOFF=1s

# we allow some grace period for requests to finish when experiment ends
export GRACE_PERIOD=5s

# URL for inference server endpoint
# for vLLM this should look like: $(IP_ADDRESS):8000
# and for TGIS this should look like $(IP_ADDRESS):8033
export URL=
