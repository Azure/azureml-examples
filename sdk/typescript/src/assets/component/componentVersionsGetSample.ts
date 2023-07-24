// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import { client, getEnvironmentVariable } from "../../utils";
// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");

/**
 * This sample demonstrates how to Get version.
 *
 * @summary Get version.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/ComponentVersion/get.json
 */
async function getComponentVersion(): Promise<void> {
  const name = "command_component_basic";
  const version = "0.0.1";
  try {
    console.log("Get component ...")
    const componentVersionsGetResponse = await client.componentVersions.get(
      resourceGroupName,
      workspaceName,
      name,
      version
    );
    console.log(componentVersionsGetResponse);
    console.log(`Get component ${componentVersionsGetResponse.name} successfully`);
  } catch (err: any) {
    console.log(
      `errorMessage - ${err.message}\n`
    )
  }
}

// getComponentVersion().catch(console.error);
export async function main(): Promise<void> {
  // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
  // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
  // about DefaultAzureCredential and the other credentials that are available for use.
  await getComponentVersion();
}

main().catch((error: any) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});