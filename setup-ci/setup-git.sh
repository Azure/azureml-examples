#!/bin/bash

# This script sets up personal Git credentials for use in a Compute Instance and clones a repository. 
# Adjust as needed.

# parameters
USERNAME="user"
EMAIL_ADDRESS="user@contoso.com"
PAT="XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
REPO="github.com/contoso/repo.git"
LOCAL_DIR="repo"

# clone repository
git clone https://$USERNAME:$PAT@$REPO $LOCAL_DIR

# change into repository
cd $LOCAL_DIR

# Configure global git on the Compute Instance's local disk.
# This is not the shared file system. You can `cd ~` to find this file later.
git config --global user.name "$USERNAME"
git config --global user.email "$EMAIL_ADDRESS"
git config --global credential.helper 'store --file ~/.aml-git-credentials'
