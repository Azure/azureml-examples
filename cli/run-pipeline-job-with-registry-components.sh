pipeline=$1
folder=$2

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
SCRIPT_DIR="$( cd "$( dirname "${SCRIPT_PATH}" )" && pwd )"

if [[ "$folder" == "6c_r_iris" ]]; then
    # Publish this checkout's environment instead of reusing the bootstrap component.
    component_version="${GITHUB_RUN_ID:-$(date -u +%Y%m%d%H%M%S%N)}-${GITHUB_RUN_ATTEMPT:-1}"
    az ml component create --file component.yml --registry-name "$REGISTRY_NAME" \
        --name component_6c_r_iris --version "$component_version" || exit $?
fi

cp $pipeline.yml ${pipeline}-registry-components.yml
sed -i -r "s/component:.+\/(.+).yml|component:\s*azureml:(.+)@latest/component: azureml:\/\/registries\/$REGISTRY_NAME\/components\/\L\1\L\2_$folder\/labels\/latest/g" ${pipeline}-registry-components.yml
sed -i -r "s/\/components\/component-/\/components\/component_/g" ${pipeline}-registry-components.yml

if [[ "$folder" == "6c_r_iris" ]]; then
    sed -i "s|/components/component_6c_r_iris/labels/latest|/components/component_6c_r_iris/versions/$component_version|g" ${pipeline}-registry-components.yml
fi

cat ${pipeline}-registry-components.yml

bash $SCRIPT_DIR/run-job.sh ${pipeline}-registry-components.yml
