# Use Application Insights with Managed Online Endpoints

This example can be run end-to-end with the script ['deploy-moe-appinsights.sh'](../../../../deploy-moe-appinsights.sh).

Learn how to capture logs and metrics from a Managed Online Endpoint using Application Insights and Log Analytics. AzureML Inference Server has extensive logging integrations with Application Insights that can be enabled with minimal changes to existing scoring logic. 

In this example we connect to an existing ['Workspace-based Application Insights Resource'](https://learn.microsoft.com/en-us/azure/azure-monitor/app/create-workspace-resource) explore the logging capabilities of AzureML Inference Server including the following: 
    - Request Logs
    - Model Data Logs
    - Exception Logs
    - Print Hook Logging (Stdout/Stderr)
    - Simple Custom Logging 

## Prerequisites
- You must have a Log Analytics Workspace-based Application Insights to use Python SDKs to query Application Insights logs. If you do not have one or have an Application Insights (Classic) instance, please follow the instructions in the ['Workspace-based Application Insights Resources'](https://learn.microsoft.com/en-us/azure/azure-monitor/app/create-workspace-resource) guide.
- You must have the following Python packages installed:
    - `azure-mgmt-resource`: Access property values of Log Analytics and Application Insights resources.
    - `azure-monitor-query`: Issue queries to Log Analytics workspaces 