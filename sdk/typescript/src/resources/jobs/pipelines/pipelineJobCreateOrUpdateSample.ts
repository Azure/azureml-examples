// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import {
    JobBase,
  } from "@azure/arm-machinelearning";
  import { client, getEnvironmentVariable } from "../../../utils";
  
  // Load the .env file if it exists
  import * as dotenv from "dotenv";
  dotenv.config();
  
  const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
  const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");
  
  /**
   * This sample demonstrates how to Create or update pipeline job.
   *
   * @summary Create or update version.
   */
  async function createOrUpdatePipelineJob() {
    const name = "simple_pipeline_job";
    const version = "0.0.1";
    const body: JobBase = {
      properties: {
        description: "This is the basic pipeline job",
        computeId: "cpu-cluster",
        jobs: {
          "node1": {
            "name": "node1",
            "type": "command",
            "inputs": {
                "component_in_number": {
                    "job_input_type": "literal",
                    "value": "${{parent.inputs.job_in_number}}"
                },
                "component_in_path": {
                    "job_input_type": "literal",
                    "value": "${{parent.inputs.job_in_path}}"
                }
            },
            "_source": "YAML.COMPONENT",
            "componentId": "/subscriptions/00000000-0000-0000-0000-000000000/resourceGroups/00000/providers/Microsoft.MachineLearningServices/workspaces/00000/components/azureml_anonymous/versions/af7c8957-aa4a-4d24-bc2e-0e6e53be325a"
          }
        },
        isAnonymous: false,
        properties: {},
        tags: { 'tag': 'tagvalue', 'owner': 'sdkteam' }
      }
    };
    // const credential = new DefaultAzureCredential();
    // const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
    try {
      console.log("Create or update pipeline job ...")
      const pipelineJobCreateOrUpdateResponse = await client.job.createOrUpdate(
        resourceGroupName,
        workspaceName,
        name,
        version,
        body
      );
      console.log(pipelineJobCreateOrUpdateResponse);
      console.log(`Created or update pipeline job ${pipelineJobCreateOrUpdateResponse.name} successfully`);
    } catch (err: any) {
      console.log(
        `errorMessage - ${err.message}\n`
      )
    }
  }
  
  // createOrUpdateComponentVersion().catch(console.error);
  export async function main(): Promise<void> {
    // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
    // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
    // about DefaultAzureCredential and the other credentials that are available for use.
    await createOrUpdatePipelineJob();
  }
  
  main().catch((error: any) => {
    console.error("An error occurred:", error);
    console.log("error code: ", error.code);
    console.log("error message: ", error.message);
    console.log("error stack: ", error.stack);
    process.exit(1);
  });