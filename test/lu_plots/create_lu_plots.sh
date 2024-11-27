#!/bin/bash

# First argument: path to HMatrices project
# Second argument: path where to store plots

if [ -z "$1" ]; then
  JULIA_PROJECT_PATH="$(dirname "$(realpath "$0")")/../.."
else
  JULIA_PROJECT_PATH="$1"
fi

if [ -z "$2" ]; then
  PLOTS_PATH="$(dirname "$(realpath "$0")")"
else
  PLOTS_PATH="$2"
fi

# Path to the Julia project environment
JULIA_ENVIRONMENT_PATH="$JULIA_PROJECT_PATH/test"

# Paths to the scripts
RUN_LU_PATH="$JULIA_PROJECT_PATH/test/lu_plots/run_lu.jl"
CREATE_PLOTS_PATH="$JULIA_PROJECT_PATH/test/lu_plots/create_plots.jl"

# Space-separated list of thread counts to use
THREAD_COUNTS="1 2 4 6 8"

# Loop over each thread count
for t in $THREAD_COUNTS; do
  echo "Running LU with $t threads..."
  export JULIA_NUM_THREADS=$t
  julia --project="$JULIA_ENVIRONMENT_PATH" "$RUN_LU_PATH" "$PLOTS_PATH"
done

# Create the plots
echo "Creating plots..."
julia --project="$JULIA_ENVIRONMENT_PATH" "$CREATE_PLOTS_PATH" "$PLOTS_PATH"
