/*
  Copyright (c) Microsoft Corporation.
  Licensed under the MIT license.

  This sample demonstrates how to create and share an `InteractiveBrowserCredential`
  to authenticate client-side requests in a single-page application.

  For more information on the authentication strategies available for 
  client-side applications, please refer to 
  https://github.com/Azure/azure-sdk-for-js/blob/main/sdk/identity/identity/samples/AzureIdentityExamples.md.
*/

/** @hidden */
export function getEnvironmentVariable(name: string): string {
  const value = process.env[name.toUpperCase()] || process.env[name.toLowerCase()];
  if (!value) {
    throw new Error(`Environment variable ${name} is not defined.`);
  }
  return value;
}

/**
 * @hidden
 */
 export function isStringNullOrEmpty(inputString: string): boolean {
  // checks whether string is null, undefined, empty or only contains space
  return !inputString || /^\s*$/.test(inputString);
}


/**
 * @hidden
 */
 export function sleep(time: number): Promise<void> {
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve();
    }, time);
  });
}