#!/bin/bash
# these registry components are needed for the samples under cli/jobs/pipelines-with-components/basics

# The filename of this script for help messages
SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"
ROOT_DIR=$(cd "${SCRIPT_DIR}/../" && pwd)
REGISTRY_NAME="${1:-${REGISTRY_NAME:-}}"

pushd "$ROOT_DIR" > /dev/null

az ml component create --file "./cli/jobs/pipelines-with-components/basics/1a_e2e_local_components/eval.yml" --registry-name $REGISTRY_NAME -n eval_1a_e2e_local_components -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/1a_e2e_local_components/score.yml" --registry-name $REGISTRY_NAME -n score_1a_e2e_local_components -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/1a_e2e_local_components/train.yml" --registry-name $REGISTRY_NAME -n train_1a_e2e_local_components -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/train.yml" --registry-name $REGISTRY_NAME -n my_train_1b_e2e_registered_components -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/score.yml" --registry-name $REGISTRY_NAME -n my_score_1b_e2e_registered_components -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/1b_e2e_registered_components/eval.yml" --registry-name $REGISTRY_NAME -n my_eval_1b_e2e_registered_components -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/2a_basic_component/component.yml" --registry-name $REGISTRY_NAME -n component_2a_basic_component -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/2b_component_with_input_output/component.yml" --registry-name $REGISTRY_NAME -n component_2b_component_with_input_output -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/3a_basic_pipeline/componentA.yml" --registry-name $REGISTRY_NAME -n componenta_3a_basic_pipeline -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/3a_basic_pipeline/componentB.yml" --registry-name $REGISTRY_NAME -n componentb_3a_basic_pipeline -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/3a_basic_pipeline/componentC.yml" --registry-name $REGISTRY_NAME -n componentc_3a_basic_pipeline -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/3b_pipeline_with_data/componentA.yml" --registry-name $REGISTRY_NAME -n componenta_3b_pipeline_with_data -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/3b_pipeline_with_data/componentB.yml" --registry-name $REGISTRY_NAME -n componentb_3b_pipeline_with_data -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/3b_pipeline_with_data/componentC.yml" --registry-name $REGISTRY_NAME -n componentc_3b_pipeline_with_data -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/4a_local_data_input/component.yml" --registry-name $REGISTRY_NAME -n component_4a_local_data_input -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/4b_datastore_datapath_uri/component-file.yml" --registry-name $REGISTRY_NAME -n component_file_4b_datastore_datapath_uri -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/4b_datastore_datapath_uri/component-folder.yml" --registry-name $REGISTRY_NAME -n component_folder_4b_datastore_datapath_uri -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/4c_web_url_input/component.yml" --registry-name $REGISTRY_NAME -n component_4c_web_url_input -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/4d_data_input/component-hello.yml" --registry-name $REGISTRY_NAME -n component_hello_4d_data_input -v 1 > /dev/null 2>&1
az ml component create --file "./cli/jobs/pipelines-with-components/basics/4d_data_input/component-world.yml" --registry-name $REGISTRY_NAME -n component_world_4d_data_input -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/5a_env_public_docker_image/component.yml" --registry-name $REGISTRY_NAME -n component_5a_env_public_docker_image -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/5b_env_registered/component.yml" --registry-name $REGISTRY_NAME -n component_5b_env_registered -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/5c_env_conda_file/component.yml" --registry-name $REGISTRY_NAME -n component_5c_env_conda_file -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/6a_tf_hello_world/component.yml" --registry-name $REGISTRY_NAME -n component_6a_tf_hello_world -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/6b_pytorch_hello_world/component.yml" --registry-name $REGISTRY_NAME -n component_6b_pytorch_hello_world -v 1 > /dev/null 2>&1

az ml component create --file "./cli/jobs/pipelines-with-components/basics/6c_r_iris/component.yml" --registry-name $REGISTRY_NAME -n component_6c_r_iris -v 1 > /dev/null 2>&1

popd > /dev/null
