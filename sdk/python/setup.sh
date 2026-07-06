#!/bin/bash

# <az_ml_sdk_install>
# pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
@@ -15,9 +14,7 @@ pip install pandas


# <az_ml_sdk_test_install>
# pip install azure-ai-ml==0.1.0.b8
pip install azure-ai-ml
# https://docsupport.blob.core.windows.net/ml-sample-submissions/1905732/azure_ai_ml-1.0.0-py3-none-any.whl
pip install https://azuresdkartifacts.z5.web.core.windows.net/python/distributions/ml-sample/6521821/azure_ai_ml-1.35.0-py3-none-any.whl
# </az_ml_sdk_test_install>

# protobuf==5.29.0 has IndentationError bug
pip install "protobuf<=5.28.3"

pip list
