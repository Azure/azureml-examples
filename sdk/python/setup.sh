#!/bin/bash

# <az_ml_sdk_install>
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner
# </mldesigner_install>

# <mltable_install>
pip install mltable
pip install pandas
# </mltable_install>


# <az_ml_sdk_test_install>
pip install azure-ai-ml==1.7.0a20230503013 --extra-index-url https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple/
# </az_ml_sdk_test_install>

pip list
