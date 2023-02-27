// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import {
    JobBase,
  } from "@azure/arm-machinelearning";
import { client, getEnvironmentVariable } from "../../../utils";
import { createOrUpdateComponentVersion } from "../../component/componentVersionsCreateOrUpdateSample";

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
  // create a simple component
  await createOrUpdateComponentVersion();
  const name = "simple_pipeline_job";
  const body: JobBase = {
    properties: {
      description: "This is the basic pipeline job",
      computeId: "cpu-cluster",
      jobType: "Pipeline",
      jobs: {
        "node1": {
          "name": "node1",
          "type": "command",
          "componentId": "command_component_basic:0.0.1"
        }
      },
      properties: {},
      tags: { 'tag': 'tagvalue', 'owner': 'sdkteam' }
    }
  };
  // const credential = new DefaultAzureCredential();
  // const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
  try {
    console.log("Create or update pipeline job ...")
    const pipelineJobCreateOrUpdateResponse = await client.jobs.createOrUpdate(
      resourceGroupName,
      workspaceName,
      name,
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