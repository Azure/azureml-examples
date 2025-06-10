# Azure/azureml-examples repository setup scripts

[![license: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](../LICENSE)

These scripts are for setting up the Azure/azureml-examples repository, including Azure resouces, using the Azure CLI and GitHub CLI.

To setup the resources required by this repository:
1. Clone the repo
```bash
git clone https://github.com/Azure/azureml-examples
```

2. Run the `azure-github.sh` script:
```bash
cd azureml-examples/cli
bash -x azure-github.sh
```
This will run the other scripts, in addition to Azure and GitHub setup. Adjust as needed.

Required CLI tools include:

- `gh`
- `az`
- `az ml`
- `azcopy`

Ensure you `az login` and `azcopy login` and have permissions to set secrets via `gh`.

# Setup MCP server

Setting up the GitHub MCP server will expedite the implementation of the samples process from a minimum of 3-4 days to just minutes. This setup is completely optional and is shared here to help you generate your code (samples/workflows) in minutes. Please update below steps if something needs to add or modify.

## Prerequisites
1. Install [Docker desktop](https://docs.docker.com/desktop/setup/install/windows-install/)
2. Install [VScode](https://code.visualstudio.com/download). You can choose your own IDE like cursor.

## Steps
1. Start Docker desktop.
2. Open Azureml-examples repo in VScode.
3. Switch to the Agent mode.
4. Preferred model is Claude Sonnet 4 but you can choose any model of your wish.
5. Create Github personal access token. Make sure you dont push this anywhere in the code.
	1. Visit this link: https://github.com/settings/personal-access-tokens and click generate new token.
	2. Set token name, expiration etc.
	3. In Repository permissions, modify below permissions and Click on "Generate token"
	  * Contents -> Read and write
    * Pull request -> Read and write
    * Metadata -> Read only
6. [Onetime] Update settings (Press Ctrl + Shift + P, type Setting and Click on `Open User Setting (JSON)`) and add [github mcp servers](https://github.com/github/github-mcp-server) details as below.
```
  "chat.mcp.discovery.enabled": true,
  "mcp": {
    "inputs": [
      {
        "type": "promptString",
        "id": "github_token",
        "description": "GitHub Personal Access Token",
        "password": true
      }
    ],
    "servers": {
      "github": {
        "command": "docker",
        "args": [
          "run",
          "-i",
          "--rm",
          "-e",
          "GITHUB_PERSONAL_ACCESS_TOKEN",
          "ghcr.io/github/github-mcp-server"
        ],
        "env": {
          "GITHUB_PERSONAL_ACCESS_TOKEN": "${input:github_token}"
        }
      }
    }
  }
```
7. Click on Start link in Settings window and it will ask you for the github token to enter. Verify that new container should start in Docker desktop after you enter your PAT.
8. Follow below prompts for generating all the boiler plate code. These prompts are for demonstration purpose only. You can use your own prompts also.

## Sample Prompts
1. Create new branch named <use_your_branch_name> and dont push it to github. 
    - Example: Create new branch named mcp_test_3 and dont push it to github
2. Add new file <use_your_sample_notebook_ipynb_file_name> in <provide_your_path> folder and update the readme file and also add readme file in <provide_folder_name> folder. 
    - Example: Add new file sdkmcpsample.ipynb in sdk/python/assets/sdkmcpsample folder and update the readme file and also add readme file in sdkmcpsample folder. 
3. Can you also add workflow for this ipynb file in .github/workflows folder exactly like how other workflows created? 
4. Can you commit and push these changes to the current branch and raise pull request with main branch?

Here is the sample pull request: https://github.com/Azure/azureml-examples/pull/3615

**Note that AI generated output can be incorrect. Before raising/merging pull request, consider to verify the output and make the require changes.**
