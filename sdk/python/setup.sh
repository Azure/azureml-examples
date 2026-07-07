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


# https://docsupport.blob.core.windows.net/ml-sample-submissions/1905732/azure_ai_ml-1.0.0-py3-none-any.whl
pip install https://azuresdkartifacts.z5.web.core.windows.net/python/distributions/ml-sample/6521821/azure_ai_ml-1.35.0-py3-none-any.whl || pip install azure-ai-ml
# </az_ml_sdk_test_install>

# protobuf==5.29.0 has IndentationError bug
pip install "protobuf<=5.28.3"

pip list
