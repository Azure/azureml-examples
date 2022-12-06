#!/bin/bash

# Please add your own values here
export subscription_id=2fcb5846-b560-4f38-8b32-ed6dedcc0a38                           
export rg_name=aml
export aml_ws=marckvaisman-aml-east2

# Compute Instances names need to be unique across Azure in a region. 
# The name will have a random number. If you run this script multiple times,
# the number will change
rn=$(od -vAn -N4 -tu4 < /dev/urandom | xargs)
export compute_instance_name=computeinstance_$rn

