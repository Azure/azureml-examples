#!/bin/bash
 
set -e

# This script configures network proxy settings for Jupyter.

RUNAS=$ADMIN_USERNAME
sudo -u $RUNAS -i <<'EOF'

mkdir -p $HOME/.ipython/profile_default/startup/ && touch $HOME/.ipython/profile_default/startup/00-startup.py

SERVER=http://myproxy
echo "Setting up proxy $SERVER for $RUNAS"
echo "export http_proxy='$SERVER'" | tee -a $HOME/.profile >/dev/null
echo "export https_proxy='$SERVER'" | tee -a $HOME/.profile >/dev/null

echo "Updated shell configuration."

echo "c.NotebookApp.terminado_settings={'shell_command': ['/bin/bash']}" | tee -a $HOME/.jupyter/jupyter_notebook_config.py >/dev/null

echo "import sys,os,os.path" | tee -a $HOME/.ipython/profile_default/startup/00-startup.py >/dev/null
echo "os.environ['HTTP_PROXY']="\""$SERVER"\""" | tee -a $HOME/.ipython/profile_default/startup/00-startup.py >/dev/null
echo "os.environ['HTTPS_PROXY']="\""$SERVER"\""" | tee -a $HOME/.ipython/profile_default/startup/00-startup.py >/dev/null

echo "Updated jupyter configuration."
EOF
