#!/bin/bash

# <az_ml_sdk_install>
pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner==0.1.0b6 --extra-index-url=https://azuremlsdktestpypi.azureedge.net/test-sdk-cli-v2/
# </mldesigner_install>

# <mltable_install>
pip install mltable
# </mltable_install>


# <az_ml_sdk_test_install>
pip install azure-ai-ml==0.1.0.b6
# </az_ml_sdk_test_install>

pip list
