
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @summary Gets computes in specified workspace.
 */
 import {
  ComputeResource
} from "@azure/arm-machinelearning";
// import { DefaultAzureCredential } from "@azure/identity";
import { client, getEnvironmentVariable } from "../../utils";

// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

// const subscriptionId = getEnvironmentVariable("SUBSCRIPTION_ID");
const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");
const azureLocation = getEnvironmentVariable("LOCATION_NAME");

/**
 * This sample demonstrates how to Creates or updates compute. This call will overwrite a compute if it exists. This is a nonrecoverable operation. If your intent is to create a new compute, do a GET first to verify that it does not exist yet.
 *
 * @summary Creates or updates compute. This call will overwrite a compute if it exists. This is a nonrecoverable operation. If your intent is to create a new compute, do a GET first to verify that it does not exist yet.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/Compute/createOrUpdate/BasicAmlCompute.json
 */
 async function createAmlCompute() {
  const computeName = "compute1234";
  const vmSize = "Standard_DS3_v2";
  const parameters: ComputeResource = {
    location: azureLocation,
    properties: {
      computeType: "AmlCompute",
      properties: {
        enableNodePublicIp: true,
        isolatedNetwork: false,
        osType: "Linux",
        // remoteLoginPortPublicAccess: "NotSpecified",
        scaleSettings: {
          maxNodeCount: 3,
          minNodeCount: 0,
          nodeIdleTimeBeforeScaleDown: "PT5M"
        },
        vmPriority: "Dedicated",
        vmSize: vmSize
      }
    }
  };
  // const credential = new DefaultAzureCredential();
  // const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
  try {
    console.log("Creating an AmlCompute ...");
    let amlComputeOperationsResponse = await client.computeOperations.beginCreateOrUpdateAndWait(
      resourceGroupName,
      workspaceName,
      computeName,
      parameters
    );
    console.log(amlComputeOperationsResponse);
    console.log(`Created AmlCompute ${amlComputeOperationsResponse.name} successfully`);
  } catch (err: any) {
    console.log(
      `errorMessage - ${err.message}\n`
    );
  }
}


/**
 * This sample demonstrates how to Creates or updates compute. This call will overwrite a compute if it exists. This is a nonrecoverable operation. If your intent is to create a new compute, do a GET first to verify that it does not exist yet.
 *
 * @summary Creates or updates compute. This call will overwrite a compute if it exists. This is a nonrecoverable operation. If your intent is to create a new compute, do a GET first to verify that it does not exist yet.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/Compute/createOrUpdate/BasicAKSCompute.json
 */
 async function createAksCompute() {
  const computeName = "akscompute123";
  const parameters: ComputeResource = {
    location: azureLocation,
    properties: { computeType: "AKS" }
  };
  // const credential = new DefaultAzureCredential();
  // const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
  try {
    console.log("Creating an AksCompute ...");
    const aksComputeOperationsResponse = await client.computeOperations.beginCreateOrUpdateAndWait(
      resourceGroupName,
      workspaceName,
      computeName,
      parameters
    );
    console.log(aksComputeOperationsResponse);
    console.log(`Created AksCompute ${aksComputeOperationsResponse.name} successfully`);
  } catch (err: any) {
    console.log(
      `errorMessage - ${err.message}\n`
    );
  }
}

// getComputes().catch(console.error);

export async function main(): Promise<void> {
  // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
  // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
  // about DefaultAzureCredential and the other credentials that are available for use.
  //console.log("Creating an AML Compute");
  await createAmlCompute();
  //console.log("Creating an AKS Compute");
   await createAksCompute();
}

main().catch((error: any) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});
