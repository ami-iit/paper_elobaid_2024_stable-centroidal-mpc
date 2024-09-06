#!/bin/bash

source ~/mambaforge/etc/profile.d/conda.sh  # Adjust this path according to your Conda installation

conda env create -f ../../quadruped-mpc/installation/mamba/integrated_gpu/mamba_environment.yml

git submodule update --init --recursive

conda activate quadruped_pympc_env

cd ../../quadruped-mpc/quadruped_pympc/acados
mkdir -p build && cd build
cmake ..
make install -j4

pip install -e ../interfaces/acados_template



cd ../../gym-quadruped
pip install -e .

cd ../..
pip install -e .

cd ../examples/quadruped

conda install gymnasium

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ACADOS_SOURCE_DIR="$SCRIPT_DIR/../../quadruped-mpc/quadruped_pympc/acados"

# Add the ACADOS library path to the LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$ACADOS_SOURCE_DIR/lib"

echo "ACADOS_SOURCE_DIR: $ACADOS_SOURCE_DIR"


python3 quadruped-test.py