/*
  Copyright (c) Microsoft Corporation.
  Licensed under the MIT license.

  This sample demonstrates how to create and share an `InteractiveBrowserCredential`
  to authenticate client-side requests in a single-page application.

  For more information on the authentication strategies available for 
  client-side applications, please refer to 
  https://github.com/Azure/azure-sdk-for-js/blob/main/sdk/identity/identity/samples/AzureIdentityExamples.md.
*/

import { AzureMachineLearningWorkspaces } from "@azure/arm-machinelearning";
import { DefaultAzureCredential } from "@azure/identity";
import { getEnvironmentVariable } from "./helper";

// Load the .env file if it exists
import * as dotenv from "dotenv";
dotenv.config();

const subscriptionId = getEnvironmentVariable("SUBSCRIPTION_ID");
export const credential = new DefaultAzureCredential();
export const client = new AzureMachineLearningWorkspaces(credential, subscriptionId);
