#!/bin/bash

# build bipdal fork with stable mpc
cd ../../humanoid-mpc
mkdir -p build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../../humanoid-mpc/build/install \
      -DCMAKE_BUILD_TYPE=Release \
      -DBIPEDAL_MPC_STABLE=ON \
      ..

make install -j4

# build test application and animate data
cd ../../examples/humanoid

if [ ! -d "build" ]; then
    mkdir build
fi
cd build

if [ ! -f "Makefile" ]; then
    cmake ..
fi

if [ ! -f "bin/humanoid-test" ]; then
    make install
fi

cd .. # Go back to the examples/humanoid directory


# Define the paths to your executable and MATLAB script
APPLICATION_EXECUTABLE="./build/bin/humanoid-test"
MATLAB_SCRIPT="./build/bin/plot_data.m"

echo "Running the application..."
$APPLICATION_EXECUTABLE

# Check if the application ran successfully
if [ $? -eq 0 ]; then
    echo "Application finished successfully."

    # Run the MATLAB script
    echo "Running MATLAB script..."
    matlab -nodisplay -r "run('$MATLAB_SCRIPT'); exit;"

    if [ $? -eq 0 ]; then
        echo "MATLAB script executed successfully."
    else
        echo "MATLAB script encountered an error."
    fi
else
    echo "Application encountered an error."
fi
