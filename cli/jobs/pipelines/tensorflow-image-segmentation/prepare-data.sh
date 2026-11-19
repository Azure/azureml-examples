#!/usr/bin/env bash
set -euo pipefail

download_dir=$(mktemp -d)
trap 'rm -f "$download_dir/images.tar.gz" "$download_dir/annotations.tar.gz"; rmdir "$download_dir"' EXIT

for archive in images annotations; do
    curl --fail --location --retry 3 --connect-timeout 30 --max-time 1800 \
        "https://www.robots.ox.ac.uk/~vgg/data/pets/data/$archive.tar.gz" \
        --output "$download_dir/$archive.tar.gz"
    tar -tzf "$download_dir/$archive.tar.gz" > /dev/null
done

for archive in images annotations; do
    az ml data create --name "oxford-pets-$archive" --version 1 \
        --type uri_file --path "$download_dir/$archive.tar.gz"
done
