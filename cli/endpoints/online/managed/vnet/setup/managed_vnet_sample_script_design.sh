#### Setup Script (runs in build agent) ####
# Create via bicep: vnet, workspace, storage, acr, kv, nsg
# Bicep: create and configure VM
    # create vm: (a) no public ip (b) set managed identity (for cli auth)
    # install CLI
    # install ml extension
# Build image with conda dependencies + push to acr

#### MIR deployment create & test script ####
# Use vm run-command invoke:
    # az upgrade -all:  CLI with extension
    # clone sample from git
    # create endpoint + dep
    # score

#### Cleanup script ####
# delete endpoints, VMs, identity