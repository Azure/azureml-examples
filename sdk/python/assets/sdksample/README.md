# Azure ML SDK Sample

This folder contains a comprehensive sample notebook demonstrating the basic usage of Azure Machine Learning SDK v2.

## Overview

The `sdksample.ipynb` notebook provides a hands-on introduction to working with Azure ML SDK v2, covering essential concepts and operations for machine learning practitioners.

## What You'll Learn

- **Workspace Connection**: How to authenticate and connect to Azure ML workspace
- **Asset Management**: Working with data, model, environment, and component assets
- **Compute Resources**: Exploring and managing compute resources
- **Data Operations**: Creating, registering, and exploring data assets
- **Job Management**: Viewing and understanding ML jobs and experiments

## Prerequisites

Before running this notebook, ensure you have:

1. **Azure Subscription**: An active Azure subscription with an Azure ML workspace
2. **Python Environment**: Python 3.7+ with the following packages:
   ```bash
   pip install azure-ai-ml azure-identity pandas
   ```
3. **Authentication**: Proper credentials for accessing your Azure ML workspace

## Getting Started

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Azure/azureml-examples
   cd azureml-examples/sdk/python/assets/sdksample
   ```

2. **Open the Notebook**:
   - Open `sdksample.ipynb` in Jupyter Lab, Jupyter Notebook, or VS Code
   - Alternatively, use Azure ML Studio notebooks

3. **Configure Workspace Details**:
   Update the following variables in the notebook with your Azure details:
   ```python
   subscription_id = "<your-subscription-id>"
   resource_group_name = "<your-resource-group>"
   workspace_name = "<your-workspace-name>"
   ```

4. **Run the Notebook**:
   Execute the cells step by step to learn about Azure ML SDK v2 capabilities

## Notebook Structure

### 1. Setup and Installation
- Package installation and imports
- Environment verification

### 2. Workspace Connection
- Authentication methods (DefaultAzureCredential, InteractiveBrowserCredential)
- Workspace client initialization

### 3. Asset Exploration
- Listing data assets
- Viewing model assets
- Exploring environment assets

### 4. Compute Resources
- Discovering available compute targets
- Understanding compute states

### 5. Data Asset Management
- Creating data assets from web URLs
- Registering assets in the workspace
- Adding metadata and tags

### 6. Data Exploration
- Reading data with pandas
- Basic data analysis and visualization

### 7. Job Management
- Viewing recent jobs and experiments
- Understanding job status and types

## Sample Data

The notebook uses the Titanic dataset as a sample for demonstrating data operations. This dataset is publicly available and commonly used for learning machine learning concepts.

## Authentication Options

The notebook supports multiple authentication methods:

- **DefaultAzureCredential**: Works automatically in Azure environments (Azure VMs, Azure ML Compute, etc.)
- **InteractiveBrowserCredential**: Opens a browser for interactive login
- **ServicePrincipalCredential**: For service principal authentication (not shown in this basic sample)

## Related Examples

After completing this sample, explore these related notebooks:

- **[Data Assets](../data/data.ipynb)**: Deep dive into data asset management
- **[Model Assets](../model/model.ipynb)**: Model registration and versioning
- **[Environment Assets](../environment/environment.ipynb)**: Custom environment creation
- **[Component Assets](../component/component.ipynb)**: Building reusable ML components

## Troubleshooting

### Common Issues

1. **Authentication Failures**:
   - Ensure you're logged into Azure CLI: `az login`
   - Check your subscription and workspace names
   - Verify permissions on the Azure ML workspace

2. **Package Import Errors**:
   - Install required packages: `pip install azure-ai-ml azure-identity pandas`
   - Check Python version compatibility (3.7+)

3. **Workspace Connection Issues**:
   - Verify subscription ID, resource group, and workspace name
   - Ensure the workspace exists and you have access

### Getting Help

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [Azure ML SDK v2 Reference](https://docs.microsoft.com/python/api/azure-ai-ml/)
- [GitHub Issues](https://github.com/Azure/azureml-examples/issues)

## Contributing

We welcome contributions! Please see the [CONTRIBUTING.md](../../../../CONTRIBUTING.md) file for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](../../../../LICENSE) file for details.
