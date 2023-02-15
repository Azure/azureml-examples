#!/bin/bash
az_batch_host_list="$AZ_BATCH_HOST_LIST"
RANK="$AZUREML_CR_NODE_RANK"

# Get ssh key from generated-key and add it to the current node.
mkdir -p /root/.ssh
mkdir /var/run/sshd
sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
sed 's@session\\s*required\\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
chmod 700 /root/.ssh/
cat 'generated-key' > ~/.ssh/id_rsa
chmod 0400 ~/.ssh/id_rsa
ssh-keygen -y -f ~/.ssh/id_rsa> ~/.ssh/id_rsa.pub
touch /root/.ssh/config;echo -e "Port 1143\n StrictHostKeyChecking no\n  UserKnownHostsFile=/dev/null" > /root/.ssh/config
echo "Port 1143" >> /etc/ssh/sshd_config
chmod 600 /root/.ssh/config
touch /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys
/usr/sbin/sshd -D -p 1143 &

## Create hostfile. Use num_gpus_per_node to populate slots value.
# parse az_batch_host_list so host_list contains list of host nodes. If it does not exist, then we are only using one node.
if  [[ -z $AZ_BATCH_HOST_LIST ]]
then
    host_list="localhost"
else
    oldIFS=IFS
    IFS=',' read -ra host_list <<< "$az_batch_host_list"
    IFS=$oldIFS
fi

# Create and write hosts to hostfile.
sudo mkdir /job
if [[ $AZUREML_PROCESS_NAME == "rank_0" ]]
then
    for i in "${host_list[@]}"
    do
        echo "$i" slots=$1 >> /job/hostfile
        echo "$i" slots=$1 >> /job/hostfile.txt
    done
fi

# Show hostfile
echo Hostfile generated
echo ------------
cat /job/hostfile
echo ------------

# Create deepspeed call using arguements passed in.
ds_call="deepspeed --hostfile /job/hostfile "
shift # Shift over to remove the first arguement (already used in hostfile above)
for i in "$@"
do
    ds_call+=$i
    ds_call+=" "
done
ls

# Evaluate deepspeed command only in first process.
if [[ $RANK == 0 ]] && [[ $AZUREML_PROCESS_NAME == "rank_0" ]]
then
    echo rank is 0, starting deepspeed
    sleep 60
    echo $ds_call
    eval $ds_call
fi