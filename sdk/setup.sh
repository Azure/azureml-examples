#!/bin/bash

# <az_ml_sdk_install>
pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner
# </mldesigner_install>

# <az_ml_sdk_test_install>
# pip install azure-ai-ml==0.0.63075866 --extra-index-url https://azuremlsdktestpypi.azureedge.net/sdk-cli-v2
# pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ azure-ai-ml==0.1.0b1
# </az_ml_sdk_test_install>

pip list