
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

/**
 * @summary Lists all the available machine learning workspaces under the specified subscription.
 */
 import { client, getEnvironmentVariable } from "../../utils";

/**
 * This sample demonstrates how to Lists all the available machine learning workspaces under the specified subscription.
 *
 * @summary Lists all the available machine learning workspaces under the specified subscription.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/Workspace/listBySubscription.json
 */
async function getWorkspacesBySubscription() {
  const resArray = new Array();
  for await (let item of client.workspaces.listBySubscription()) {
    resArray.push(item);
  }
  console.log(resArray);
}

// getWorkspacesBySubscription().catch(console.error);

export async function main(): Promise<void> {
  // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
  // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
  // about DefaultAzureCredential and the other credentials that are available for use.
  console.log("Listing the workspaces");
  await getWorkspacesBySubscription();
}

main().catch((error) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});