set -x

# <create_variables>
SUBSCRIPTION_ID=$(az account show --query id | tr -d '\r"')
LOCATION=$(az ml workspace show --query location | tr -d '\r"')
RESOURCE_GROUP=$(az group show --query name | tr -d '\r"')
WORKSPACE=$(az configure -l | jq -r '.[] | select(.name=="workspace") | .value')
schema='$schema'
#</create_variables>

echo -e "Using:\nSUBSCRIPTION_ID=$SUBSCRIPTION_ID\nLOCATION=$LOCATION\nRESOURCE_GROUP=$RESOURCE_GROUP\nWORKSPACE=$WORKSPACE"

# <read_condafile>
CONDA_FILE=$(< cli/endpoints/online/model-1/environment/conda.yml)
# </read_condafile>

#<get_access_token>
TOKEN=$(az account get-access-token --query accessToken -o tsv)
#</get_access_token>

# <set_endpoint_name>
export ENDPOINT_NAME=endpt-`echo $RANDOM`
# </set_endpoint_name>

#<api_version>
API_VERSION="2022-05-01"
#</api_version>

# define how to wait
wait_for_completion () {
  status="unknown"
  operation_id=""

  while [[ $operation_id == "" || -z $operation_id  || $operation_id == "null" ]]
    do
        sleep 5
        response=$($1)
        operation_id=$(echo $response | jq -r '.properties' | jq -r '.properties' | jq -r '.AzureAsyncOperationUri')
    done

  while [[ $status != "Succeeded" && $status != "Failed" ]]
    do
        operation_result=$(curl --location --request GET $operation_id --header "Authorization: Bearer $TOKEN")
        status=$(echo $operation_result | jq -r '.status')
        echo "Current operation status: $status"
        sleep 5
    done

  if [[ $status == "Failed" ]]
  then
      error=$(echo $operation_result | jq -r '.error')
      echo "Error: $error"
  fi
}

# <get_storage_details>
response=$(curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/datastores?api-version=$API_VERSION&isDefault=true" \
--header "Authorization: Bearer $TOKEN")
AZUREML_DEFAULT_DATASTORE=$(echo $response | jq -r '.value[0].name')
AZUREML_DEFAULT_CONTAINER=$(echo $response | jq -r '.value[0].properties.containerName')
export AZURE_STORAGE_ACCOUNT=$(echo $response | jq -r '.value[0].properties.accountName')
# </get_storage_details>

# <upload_code>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/score -s cli/endpoints/online/model-1/onlinescoring --account-name $AZURE_STORAGE_ACCOUNT
# </upload_code>

# <create_code>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Resources/deployments/score-sklearn?api-version=2021-04-01" \
-H "Authorization: Bearer $TOKEN" \
-H 'Content-Type: application/json' \
--data-raw "{
\"properties\": {
    \"mode\": \"Incremental\",
    \"template\": {
        \"$schema\": \"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#\",
        \"contentVersion\": \"1.0.0.0\",
        \"parameters\": {
            \"workspaceName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning Workspace which will contain this compute.\"
                }
            },
            \"codeAssetName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning code asset which will be created or updated.\"
                }
            },
            \"codeAssetVersion\": {
                \"defaultValue\": \"1\",
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the version of the Azure Machine Learning code asset which will be created or updated.\"
                }
            },
            \"codeUri\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the location of the Azure Machine Learning code asset in a storage account.\"
                }
            },
            \"codeAssetDescription\": {
                \"defaultValue\": \"This is a test description for a code asset created by an ARM template\",
                \"type\": \"string\"
            },
            \"isAnonymous\": {
                \"defaultValue\": false,
                \"type\": \"bool\",
                \"metadata\": {
                    \"description\": \"If the name version are system generated (anonymous registration).\"
                }
            }
        },
        \"resources\": 
        [
            {          
                \"type\": \"Microsoft.MachineLearningServices/workspaces/codes/versions\",
                \"apiVersion\": \"$API_VERSION\",
                \"name\": \"[concat(parameters(\'workspaceName\'), \'/\', parameters(\'codeAssetName\'), \'/\', parameters(\'codeAssetVersion\'))]\",
                \"properties\": {
                    \"description\": \"[parameters(\'codeAssetDescription\')]\",
                    \"codeUri\": \"[parameters(\'codeUri\')]\",
                    \"isAnonymous\": \"[parameters(\'isAnonymous\')]\",
                    \"properties\": {},
                    \"tags\": {}
                }
            }
        ]
    },
    \"parameters\": {
        \"workspaceName\": {
            \"value\": \"$WORKSPACE\"
            },
            \"codeAssetName\": {
                \"value\": \"score-sklearn\"
            },
            \"codeAssetVersion\": {
                \"value\": \"1\"
            },
            \"codeUri\": {
                \"value\": \"https://$AZURE_STORAGE_ACCOUNT.blob.core.windows.net/$AZUREML_DEFAULT_CONTAINER/score\"
            }
        }
    }
}"
# </create_code>

# <upload_model>
az storage blob upload-batch -d $AZUREML_DEFAULT_CONTAINER/model -s cli/endpoints/online/model-1/model --account-name $AZURE_STORAGE_ACCOUNT
# </upload_model>

# <create_model>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Resources/deployments/sklearn?api-version=2021-04-01" \
-H "Authorization: Bearer $TOKEN" \
-H 'Content-Type: application/json' \
--data-raw "{
\"properties\": {
    \"mode\": \"Incremental\",
    \"template\": {
        \"$schema\": \"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#\",
        \"contentVersion\": \"1.0.0.0\",
        \"parameters\": {
            \"workspaceName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning Workspace which will contain this compute.\"
                }
            },
            \"modelAssetName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning model asset which will be created or updated.\"
                }
            },
            \"modelAssetVersion\": {
                \"defaultValue\": \"1\",
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the version of the Azure Machine Learning model asset which will be created or updated.\"
                }
            },
            \"modelUri\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the location of the Azure Machine Learning model asset in a storage account.\"
                }
            },
            \"modelAssetDescription\": {
                \"defaultValue\": \"This is a test description for a model asset created by an ARM template\",
                \"type\": \"string\"
            },
            \"isAnonymous\": {
                \"defaultValue\": false,
                \"type\": \"bool\",
                \"metadata\": {
                    \"description\": \"If the name version are system generated (anonymous registration).\"
                }
            }
        },
        \"resources\": 
        [
            {          
                \"type\": \"Microsoft.MachineLearningServices/workspaces/models/versions\",
                \"apiVersion\": \"2021-10-01\",
                \"name\": \"[concat(parameters(\'workspaceName\'), \'/\', parameters(\'modelAssetName\'), \'/\', parameters(\'modelAssetVersion\'))]\",
                \"properties\": {
                    \"description\": \"[parameters(\'modelAssetDescription\')]\",
                    \"modelUri\": \"[parameters(\'modelUri\')]\",
                    \"isAnonymous\": \"[parameters(\'isAnonymous\')]\",
                    \"properties\": {},
                    \"tags\": {}
                }
            }
        ]
    },
    \"parameters\": {
        \"workspaceName\": {
            \"value\": \"$WORKSPACE\"
            },
            \"modelAssetName\": {
                \"value\": \"score-sklearn\"
            },
            \"modelAssetVersion\": {
                \"value\": \"1\"
            },
            \"modelUri\": {
                \"value\": \"azureml://subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/workspaces/$WORKSPACE/datastores/$AZUREML_DEFAULT_DATASTORE/paths/model/sklearn_regression_model.pkl\"
            }
        }
    }
}"
# </create_model>

# <read_condafile>
CONDA_FILE=$(cat cli/endpoints/online/model-1/environment/conda.yml)
# </read_condafile>

# <create_environment>
ENV_VERSION=$RANDOM
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Resources/deployments/sklearn-env?api-version=2021-04-01" \
-H "Authorization: Bearer $TOKEN" \
-H 'Content-Type: application/json' \
--data-raw "{
\"properties\": {
    \"mode\": \"Incremental\",
    \"template\": {
        \"$schema\": \"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#\",
        \"contentVersion\": \"1.0.0.0\",
        \"parameters\": {
            \"workspaceName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning Workspace which will contain this compute.\"
                }
            },
            \"environmentAssetName\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the name of the Azure Machine Learning environment asset which will be created or updated.\"
                }
            },
            \"environmentAssetVersion\": {
                \"defaultValue\": \"1\",
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Specifies the version of the Azure Machine Learning environment asset which will be created or updated.\"
                }
            },
            \"dockerImage\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Docker image path, for example: 'docker.io/tensorflow/serving:latest'.\"
                }
            },
            \"condaFile\": {
                \"type\": \"string\",
                \"metadata\": {
                    \"description\": \"Standard configuration file used by Conda that lets you install any kind of package, including Python, R, and C/C++ packages.\"
                }
            },
            \"environmentAssetDescription\": {
                \"defaultValue\": \"This is a test description for a environment asset created by an ARM template\",
                \"type\": \"string\"
            },
            \"isAnonymous\": {
                \"defaultValue\": false,
                \"type\": \"bool\",
                \"metadata\": {
                    \"description\": \"If the name version are system generated (anonymous registration).\"
                }
            }
        },
        \"resources\": 
        [
            {          
                \"type\": \"Microsoft.MachineLearningServices/workspaces/environments/versions\",
                \"apiVersion\": \"$API_VERSION\",
                \"name\": \"[concat(parameters(\'workspaceName\'), \'/\', parameters(\'environmentAssetName\'), \'/\', parameters(\'environmentAssetVersion\'))]\",
                \"properties\": {
                    \"description\": \"[parameters(\'environmentAssetDescription\')]\",
                    \"image\": \"[parameters(\'dockerImage\')]\",
                    \"condaFile\": \"[parameters(\'condaFile\')]\",
                    \"isAnonymous\": \"[parameters(\'isAnonymous\')]\"
                }
            }
        ]
    },
    \"parameters\": {
        \"workspaceName\": {
            \"value\": \"$WORKSPACE\"
        },
        \"environmentAssetName\": {
            \"value\": \"sklearn-env\"
        },
        \"environmentAssetVersion\": {
            \"value\": \"$ENV_VERSION\"
        },
        \"dockerImage\": {
            \"value\": \"mcr.microsoft.com/azureml/openmpi3.1.2-ubuntu18.04:20210727.v1\"
        },
        \"condaFile\": {
            \"value\": \"$CONDA_FILE\"
        }
    }
  }
}"
# </create_environment>

# <create_endpoint>
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Resources/deployments/$ENDPOINT_NAME?api-version=2021-04-01" \
-H "Authorization: Bearer $TOKEN" \
-H "Content-Type: application/json" \
--data-raw "{
    \"properties\": {
        \"mode\": \"Incremental\",
        \"template\": {
            \"$schema\": \"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#\",
            \"contentVersion\": \"1.0.0.0\",
            \"parameters\": {
                \"workspaceName\": {
                    \"type\": \"string\",
                    \"metadata\": {
                        \"description\": \"Specifies the name of the Azure Machine Learning Workspace which will contain this compute.\"
                    }
                },
                \"location\": {
                    \"type\": \"string\"
                },
                \"identityType\": {
                    \"allowedValues\": [
                      \"SystemAssigned\",
                      \"UserAssigned\",
                      \"SystemAssignedUserAssigned\",
                      \"None\"
                    ],
                    \"type\": \"string\",
                    \"metadata\": {
                        \"description\": \"The MSI Identity that is associated with this resource.\"
                    }
                },
                \"authMode\": {
                  \"defaultValue\": \"Key\",
                  \"allowedValues\": [
                    \"AMLToken\",
                    \"Key\",
                    \"AADToken\"
                  ],
                  \"type\": \"string\"
                },
                \"onlineEndpointName\": {
                    \"type\": \"string\",
                    \"metadata\": {
                        \"description\": \"Specifies the name of the Azure Machine Learning endpoint which will be created or updated.\"
                    }
                },
                \"onlineEndpointDescription\": {
                  \"defaultValue\": \"This is an online endpoint created by an ARM template\",
                  \"type\": \"string\"
                },
                \"onlineEndpointTags\": {
                  \"defaultValue\": {
                    \"tag1\": \"value1\",
                    \"tag2\": \"value2\",
                    \"tag3\": \"value3\"
                  },
                  \"type\": \"object\"
                }
            },
            \"resources\": 
            [
              {
                \"type\": \"Microsoft.MachineLearningServices/workspaces/onlineEndpoints\",
                \"apiVersion\": \"$API_VERSION\",
                \"name\": \"[concat(parameters(\'workspaceName\'), \'/\', parameters(\'onlineEndpointName\'))]\",
                \"location\": \"[parameters(\'location\')]\",
                \"tags\": \"[parameters(\'onlineEndpointTags\')]\",
                \"identity\": {
                  \"type\": \"[parameters(\'identityType\')]\"
                },
                \"properties\": {
                  \"authMode\": \"[parameters(\'authMode\')]\",
                  \"description\": \"[parameters(\'onlineEndpointDescription\')]\",
                }
              }
            ]
        },
        \"parameters\": {
            \"workspaceName\": {
                \"value\": \"$WORKSPACE\"
            },
            \"location\": {
                \"value\": \"$LOCATION\"
            },
            \"onlineEndpointName\": {
                \"value\": \"$ENDPOINT_NAME\"
            },
            \"identityType\": {
                \"value\": \"SystemAssigned\"
            },
            \"authMode\": {
                \"value\": \"AMLToken\"
            }
        }
    }
}"
# </create_endpoint>

# <get_endpoint>
endpoint_cmd() {
  curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TOKEN"
}
wait_for_completion endpoint_cmd
# </get_endpoint>

# <create_deployment>
resourceScope="/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices"
curl --location --request PUT "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Resources/deployments/blue?api-version=2021-04-01" \
-H "Authorization: Bearer $TOKEN" \
-H 'Content-Type: application/json' \
--data-raw "{
    \"properties\": {
      \"mode\": \"Incremental\",
      \"template\": {
        \"$schema\": \"https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#\",
        \"contentVersion\": \"1.0.0.0\",
        \"parameters\": {
          \"workspaceName\": {
            \"type\": \"string\",
            \"metadata\": {
              \"description\": \"Specifies the name of the Azure Machine Learning Workspace which will contain this compute.\"
            }
          },
          \"location\": {
            \"type\": \"string\"
          },
          \"appInsightsEnabled\": {
            \"type\": \"bool\",
            \"defaultValue\": false
          },
          \"onlineEndpointName\": {
            \"type\": \"string\",
            \"metadata\": {
              \"description\": \"Specifies the name of the Azure Machine Learning online endpoint which will be deployed.\"
            }
          },
          \"onlineDeploymentName\": {
            \"defaultValue\": \"blue\",
            \"type\": \"string\",
            \"metadata\": {
              \"description\": \"Specifies the name of the Azure Machine Learning online endpoint which will be deployed.\"
            }
          },
          \"onlineEndpointDescription\": {
            \"defaultValue\": \"This is an online endpoint deployment created by an ARM template\",
            \"type\": \"string\"
          },
          \"onlineDeploymentTags\": {
            \"defaultValue\": {
              \"tag1\": \"value1\",
              \"tag2\": \"value2\",
              \"tag3\": \"value3\"
            },
            \"type\": \"object\"
          },
          \"codeId\": {
            \"type\": \"string\"
          },
          \"scoringScript\": {
            \"type\": \"string\",
            \"defaultValue\": \"score.py\",
            \"metadata\": {
              \"description\": \"The script to execute on startup. eg. 'score.py'\"
            }
          },
          \"environmentId\": {
          \"type\": \"string\"
          },
          \"model\": {
            \"type\": \"string\"
          },
          \"endpointComputeType\": {
            \"type\": \"string\",
            \"allowedValues\": [
              \"Managed\",
              \"Kubernetes\",
              \"AzureMLCompute\"
            ]
          },
          \"skuName\": {
            \"type\": \"string\",
            \"metadata\": {
              \"description\": \"The name of the SKU. Ex - P3. It is typically a letter+number code\"
            }
          },
          \"skuCapacity\": {
            \"type\": \"int\",
            \"defaultValue\": 1,
            \"metadata\": {
              \"description\": \"If the SKU supports scale out/in then the capacity integer should be included. If scale out/in is not possible for the resource this may be omitted.\"
            }
          }
        },
        \"resources\": [
          {
            \"type\": \"Microsoft.MachineLearningServices/workspaces/onlineEndpoints/deployments\",
            \"apiVersion\": \"$API_VERSION\",
            \"name\": \"[concat(parameters(\'workspaceName\'), \'/\', parameters(\'onlineEndpointName\'), \'/\', parameters(\'onlineDeploymentName\'))]\",
            \"location\": \"[parameters(\'location\')]\",
            \"tags\": \"[parameters(\'onlineDeploymentTags\')]\",
            \"sku\": {
              \"name\": \"[parameters(\'skuName\')]\",
              \"capacity\": \"[parameters(\'skuCapacity\')]\"
            },
            \"properties\": {
              \"description\": \"[parameters(\'onlineEndpointDescription\')]\",
              \"codeConfiguration\": {
                \"codeId\": \"[parameters(\'codeId\')]\",
                \"scoringScript\": \"[parameters(\'scoringScript\')]\"
              },
              \"environmentId\": \"[parameters(\'environmentId\')]\",
              \"appInsightsEnabled\": \"[parameters(\'appInsightsEnabled\')]\",
              \"endpointComputeType\": \"[parameters(\'endpointComputeType\')]\",
              \"model\": \"[parameters(\'model\')]\"
            }
          }
        ]
      },
      \"parameters\": {
        \"workspaceName\": {
          \"value\": \"$WORKSPACE\"
        },
        \"location\": {
          \"value\": \"$LOCATION\"
        },
        \"onlineEndpointName\": {
          \"value\": \"$ENDPOINT_NAME\"
        },
        \"onlineDeploymentName\": {
            \"value\": \"blue\"
        },
        \"codeId\": {
            \"value\": \"$resourceScope/workspaces/$WORKSPACE/codes/score-sklearn/versions/1\"
        },
        \"scoringScript\": {
            \"value\": \"score.py\"
        },
        \"environmentId\": {
            \"value\": \"$resourceScope/workspaces/$WORKSPACE/environments/sklearn-env/versions/$ENV_VERSION\"
        },
        \"model\": {
            \"value\": \"$resourceScope/workspaces/$WORKSPACE/models/score-sklearn/versions/1\"
        },
        \"endpointComputeType\": {
            \"value\": \"Managed\"
        },
        \"skuName\": {
            \"value\": \"Standard_F2s_v2\"
        },
        \"skuCapacity\": {
            \"value\": 1
        }
      }
    }
}"
# </create_deployment>

# <get_deployment>
deployment_cmd() {
  curl --location --request GET "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/deployments/blue?api-version=$API_VERSION" \
  --header "Content-Type: application/json" \
  --header "Authorization: Bearer $TOKEN"
}

wait_for_completion deployment_cmd

scoringUri=$(echo deployment_cmd | jq -r '.properties' | jq -r '.scoringUri')
# </get_deployment>

# <get_endpoint_access_token>
response=$(curl -H "Content-Length: 0" --location --request POST "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/token?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN")
accessToken=$(echo $response | jq -r '.accessToken')
# </get_endpoint_access_token>

# <score_endpoint>
curl --location --request POST $scoringUri \
--header "Authorization: Bearer $accessToken" \
--header "Content-Type: application/json" \
--data @cli/endpoints/online/model-1/sample-request.json
# </score_endpoint>

# <get_deployment_logs>
curl --location --request POST "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME/deployments/blue/getLogs?api-version=$API_VERSION" \
--header "Authorization: Bearer $TOKEN" \
--header "Content-Type: application/json" \
--data-raw "{ \"tail\": 100 }"
# </get_deployment_logs>

# <delete_endpoint>
curl --location --request DELETE "https://management.azure.com/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.MachineLearningServices/workspaces/$WORKSPACE/onlineEndpoints/$ENDPOINT_NAME?api-version=$API_VERSION" \
--header "Content-Type: application/json" \
--header "Authorization: Bearer $TOKEN" || true
# </delete_endpoint>
