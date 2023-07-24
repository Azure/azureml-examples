// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.
import {
  ComponentVersion,
  ComponentVersionsListOptionalParams,
} from "@azure/arm-machinelearning";

import { client, getEnvironmentVariable } from "../../utils";
// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");

/**
 * This sample demonstrates how to List component versions.
 *
 * @summary List component versions.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/ComponentVersion/list.json
 */
async function listComponentVersion(): Promise<void> {
  const name = "command_component_basic";
  const orderBy = "createdtime desc";
  const top = 1;
  const options: ComponentVersionsListOptionalParams = { orderBy, top };
  const resArray = new Array<ComponentVersion>();
  for await (let componentItem of client.componentVersions.list(
    resourceGroupName,
    workspaceName,
    name,
    options
  )) {
    resArray.push(componentItem);
    console.log(componentItem);
  }
  console.log(resArray);
}

// listComponentVersion().catch(console.error);
export async function main(): Promise<void> {
  // This sample uses DefaultAzureCredential, which supports a number of authentication mechanisms.
  // See https://docs.microsoft.com/javascript/api/overview/azure/identity-readme?view=azure-node-latest for more information
  // about DefaultAzureCredential and the other credentials that are available for use.
  console.log("Listing the Component");
  await listComponentVersion();
}

main().catch((error) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});