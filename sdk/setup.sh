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
# pip install azure-ai-ml==0.1.0.b6
pip install https://docsupport.blob.core.windows.net/ml-sample-submissions/1860282/azure_ai_ml-0.1.0b7-py3-none-any.whl
# </az_ml_sdk_test_install>

pip list
