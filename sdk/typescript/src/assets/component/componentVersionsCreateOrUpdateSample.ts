// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

import {
  ComponentVersion,
} from "@azure/arm-machinelearning";
import { client, getEnvironmentVariable } from "../../utils";

// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

const resourceGroupName = getEnvironmentVariable("RESOURCEGROUP_NAME");
const workspaceName = getEnvironmentVariable("WORKSPACE_NAME");

/**
 * This sample demonstrates how to Create or update version.
 *
 * @summary Create or update version.
 * x-ms-original-file: specification/machinelearningservices/resource-manager/Microsoft.MachineLearningServices/stable/2022-10-01/examples/ComponentVersion/createOrUpdate.json
 */
export async function createOrUpdateComponentVersion() : Promise<void> {
  const name = "command_component_basic";
  const version = "0.0.1";
  const body: ComponentVersion = {
    properties: {
      description: "This is the basic command component",
      componentSpec: {
        'command': 'echo Hello World & echo $[[${{inputs.component_in_number}}]] & echo $[[${{inputs.component_in_path}}]] & echo ${{outputs.component_out_path}} > ${{outputs.component_out_path}}/component_in_number',
        'environment': 'azureml:AzureML-sklearn-0.24-ubuntu18.04-py37-cpu:1',
        'name': 'command_component_basic',
        'description': 'This is the basic command component',
        'tags': { 'tag': 'tagvalue', 'owner': 'sdkteam' },
        'version': '0.0.1',
        '$schema': 'https://azuremlschemas.azureedge.net/development/commandComponent.schema.json',
        'display_name': 'CommandComponentBasic',
        'is_deterministic': true,
        'inputs': {
          'component_in_number': { 'type': 'number', 'optional': true, 'default': '10.99', 'description': 'A number' },
          'component_in_path': { 'type': 'uri_folder', 'optional': true, 'description': 'A path' }
        },
        'outputs': { 'component_out_path': { 'type': 'uri_folder' } },
        'type': 'command',
        '_source': 'YAML.COMPONENT'
      },
      isAnonymous: false,
      properties: {},
      tags: { 'tag': 'tagvalue', 'owner': 'sdkteam' }
    }
  };
  // const credential = new DefaultAzureCredential();
  // const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
  try {
    console.log("Create or update component version ...")
    const componentVersionsCreateOrUpdateResponse = await client.componentVersions.createOrUpdate(
      resourceGroupName,
      workspaceName,
      name,
      version,
      body
    );
    console.log(componentVersionsCreateOrUpdateResponse);
    console.log(`Created or update component ${componentVersionsCreateOrUpdateResponse.name} successfully`);
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
  await createOrUpdateComponentVersion();
}

main().catch((error: any) => {
  console.error("An error occurred:", error);
  console.log("error code: ", error.code);
  console.log("error message: ", error.message);
  console.log("error stack: ", error.stack);
  process.exit(1);
});