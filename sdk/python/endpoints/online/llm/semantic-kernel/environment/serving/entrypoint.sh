#!/bin/bash

set -x
export 
ENTRYFILE="$AZUREML_MODEL_DIR/$AZUREML_SERVING_ENTRYPOINT"
DIR="$( cd "$( dirname "$ENTRYFILE" )" >/dev/null 2>&1 && pwd )"
FILENAME="$(basename $ENTRYFILE)"
cd $DIR
ls -la
bash $FILENAME