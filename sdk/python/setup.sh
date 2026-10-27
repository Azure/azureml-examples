#!/bin/bash

# <az_ml_sdk_install>
# pip install --pre azure-ai-ml
# </az_ml_sdk_install>

# <mldesigner_install>
pip install mldesigner
# </mldesigner_install>

# <mltable_install>
pip install mltable
pip install pandas
# </mltable_install>


# <az_ml_sdk_test_install>
# pip install azure-ai-ml==0.1.0.b8
pip install https://shrivastavpwpv5228506759.blob.core.windows.net/wheels/azure_ai_ml-1.27.0-py3-none-any.whl
# https://docsupport.blob.core.windows.net/ml-sample-submissions/1905732/azure_ai_ml-1.0.0-py3-none-any.whl
# </az_ml_sdk_test_install>

# protobuf==5.29.0 has IndentationError bug
pip install "protobuf<=5.28.3"

pip list