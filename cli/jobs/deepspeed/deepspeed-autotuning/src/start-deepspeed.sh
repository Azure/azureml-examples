#!/bin/bash
az_batch_host_list="$AZ_BATCH_HOST_LIST"
RANK="$AZUREML_CR_NODE_RANK"

# Start ssh
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

# Create hostfile. Use num_gpus_per_node to populate slots value.
oldIFS=IFS
IFS=',' read -ra host_list <<< "$az_batch_host_list"
IFS=$oldIFS

sudo mkdir /job
if ! [[ -w /job/hostfile ]]
then
    for i in "${host_list[@]}"
    do
        echo "$i" slots=$1 >> /job/hostfile
        echo "$i" slots=$1 >> /job/hostfile.txt
    done
fi

echo Hostfile generated
echo ------------
cat /job/hostfile
echo ------------

# Create deepspeed call
ds_call="deepspeed --hostfile /job/hostfile "
shift
for i in "$@"
do
    ds_call+=$i
    ds_call+=" "
done
ls
if [[ $RANK == 0 ]] && [[ $AZUREML_PROCESS_NAME == "rank_0" ]]
then
    echo rank is 0, starting deepspeed
    sleep 60
    echo $ds_call
    eval $ds_call
fi
