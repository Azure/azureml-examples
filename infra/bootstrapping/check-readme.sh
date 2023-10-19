#!/bin/bash

# Get the folder path from the command-line argument
folder_path="$1"

echo "Checking if $folder_path contains a README.md file with all required words..."

# Check if the "README.md" file exists in the folder
if [ -e "$folder_path/README.md" ]; then

    # Check if the required words exist in the README.md file
    if grep -qi "overview:" "$folder_path/README.md" && grep -qi "objective:" "$folder_path/README.md" && grep -qi "programming languages:" "$folder_path/README.md" && grep -qi "estimated runtime:" "$folder_path/README.md"; then
        echo "The folder contains a README.md file with all required words."
    else
        echo "The folder contains a README.md file, but it does not have all required words."
        exit 1
    fi
else
    echo "The folder does not contain a README.md file."
    exit 1
fi
