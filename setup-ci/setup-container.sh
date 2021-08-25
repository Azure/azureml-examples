#This script allows a docker container to be registered as Jupyter kernel
#This script refers to a sample docker file and a helper file kernel.json

echo $(date -u) "Starting user setup."
RUNAS=azureuser

sudo CUR_PATH=`pwd` -u $RUNAS -i <<'EOF'

ENV_FOLDER=~/customenv
KERNEL_FOLDER=~/.local/share/jupyter/kernels/custom_env

mkdir $ENV_FOLDER
cp $CUR_PATH/Dockerfile $ENV_FOLDER
cd $ENV_FOLDER
echo $(date -u) "Building docker image..."
docker build --tag my-custom-env .
echo $(date -u) "Docker image built."

mkdir -p $KERNEL_FOLDER
cp $CUR_PATH/kernel.json $KERNEL_FOLDER
echo $(date -u) "Installed kernel."

EOF

systemctl restart jupyter
echo $(date -u) "Restarted jupyter."
