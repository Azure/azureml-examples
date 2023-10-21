#!/bin/bash

# Get the folder path from the command-line arguments
working_directory="$1"
folder_path="$2"

echo "Checking if $folder_path contains a README.md file with all required words..."

# Check if the "README.md" file exists in the folder
invalid_readme_message="$folder_path does not contain a README.md file with the required sections. See CONTRIBUTING.md."

# Check if folder path is in ./readme_exclusions.txt file
exclusions_file="{working_directory}/infra/bootstrapping/readme_validation_exclusions.txt"

if [ -f "$exclusions_file" ]; then
    while IFS= read -r line; do
        if [ "$line" = "$folder_path" ]; then
            echo "Skipping $folder_path since it is excluded from the README check."
            exit 0
        fi
    done < "$exclusions_file"
fi

if [ -e "$folder_path/README.md" ]; then

    # Define an array of required words
    required_words=("overview:" "objective:" "programming languages:" "estimated runtime:")

    # Iterate through the required words and check for their presence in the README.md file (case-insensitive)
    for word in "${required_words[@]}"; do
        if ! grep -qi "$word" "$folder_path/README.md"; then
            echo $invalid_readme_message
            exit 1
        fi
    done
else
    echo $invalid_readme_message
    exit 1
fi

echo "This sample contains a valid README."