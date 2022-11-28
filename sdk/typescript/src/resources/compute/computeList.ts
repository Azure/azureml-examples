
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @summary Gets computes in specified workspace.
 */
 import { client, getEnvironmentVariable } from "../../utils";

// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

/**
 * This sample demonstrates how to Gets computes in specified workspace.
 *
 * @summary Gets computes in specified workspace.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/Compute/list.json
 */
async function getComputes() {
// const subscriptionId = getEnvironmentVariable("SUBSCRIPTION_ID");
const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");

  for await (let computeItem of client.computeOperations.list(
    resourceGroupName,
    workspaceName
  )){
    console.log(computeItem);
  }
  const resArray = new Array();
  for await (let item of client.computeOperations.list(
    resourceGroupName,
    workspaceName
  )) {
    resArray.push(item);
  }
  console.log(resArray);
}

// getComputes().catch(console.error);

export async function main(): Promise<void> {
  // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
  // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
  // about DefaultAzureCredential and the other credentials that are available for use.
  console.log("Listing the Compute");
  await getComputes();
}

main().catch((error) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});