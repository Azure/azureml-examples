#!/bin/bash
# these components are needed for the sample under cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)

pushd "$ROOT_DIR" > /dev/null

az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/train.yml" > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/score.yml" > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/eval.yml" > /dev/null 2>&1

popd > /dev/null
