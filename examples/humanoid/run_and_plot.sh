#!/bin/bash

# Define the paths to your executable and MATLAB script
APPLICATION_EXECUTABLE="./build/bin/humanoid-test"
MATLAB_SCRIPT="./build/bin/plot_data.m"

# Run the application
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
