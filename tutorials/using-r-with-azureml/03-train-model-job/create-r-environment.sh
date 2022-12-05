#!/bin/bash

az ml environment create -f r-environment-with-mlflow.yml \
--workspace-name marckvaisman-aml-east2 \
--resource-group aml \
--subscription 2fcb5846-b560-4f38-8b32-ed6dedcc0a38