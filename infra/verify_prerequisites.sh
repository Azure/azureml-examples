#!/bin/bash

###################
# validate dependencies if the required utilities are installed
###################

command -v az >/dev/null 2>&1 || { echo_warning "azure cli but it's not installed. See https://docs.microsoft.com/cli/azure/?view=azure-cli-latest/. Aborting."; exit 1; }
command -v jq >/dev/null 2>&1 || { echo_warning "jq is required but not installed. See https://stedolan.github.io/jq/.  Aborting."; exit 1; }
type sed >/dev/null 2>&1 || { echo >&2 "Error: You do not have 'sed' installed."; exit 1; }
# command -v kubectl >/dev/null 2>&1 || { echo >&2 "'kubectl' is required but not installed. Aborting."; exit 1; }