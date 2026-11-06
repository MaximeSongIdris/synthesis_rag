#!/usr/bin/env bash

### ENV VARIABLES DEFINITION
export PROJECT_PATH=/lustre/fswork/projects/idris/sos/ssos027/synthesis_RAG  # path to the project on your Jean Zay account

### ACTIVATE ENVIRONMENT
# Check if the job is running on h100 nodes
if echo $SLURM_JOB_ACCOUNT | grep -q "h100"; then
    module load arch/h100
# Check if the job is running on a100 nodes
elif echo $SLURM_JOB_ACCOUNT | grep -q "a100"; then
    module load arch/a100
fi
module load pytorch-gpu/py3/2.13.0

# Add binaries to PATH
export PATH=$PROJECT_PATH/.local/bin:$PATH
# Local install of python packages outside of the conda env
export PYTHONUSERBASE=$PROJECT_PATH/.local
