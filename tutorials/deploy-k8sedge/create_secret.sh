#!/bin/bash

# You would need to login to the container registry first, to get config.json with the authentication tokens
# 
# For example,
#  $ docker login <myaccount>.azurecr.io

sudo microk8s.kubectl create secret generic secret4acr2infer \
     --from-file=.dockerconfigjson=/home/azureuser/.docker/config.json \
     --type=kubernetes.io/dockerconfigjson

#
# You can also create a secred using your SPN id and secret:
#
#kubectl create secret docker-registry <secret name> `
#    --docker-server=<crname, with FQDN>`
#    --docker-username=$userSPNID `
#    --docker-password=$userSPNSecret