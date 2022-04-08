## Workspace Connections

Use workspace connections to set up secure connection to external resources such as Azure DevOps feed or git repository. These connections can then be used as package sources in environments.


For example, to create a connection to GitHub repository with username and password authentication, use: 

```
az ml connection create --file git-user-passwd.yml --credentials username=<username> password=<password>
```

**Important!** When creating connection, in-line any sensitive content such as passwords or tokens. Do not store them in the yaml file in plaintext.

To updata a connection, use:

```
az ml connection show
```

To list connections in workspace, use:

```
az ml connection list
```

To show details of a connection, use:

```
az ml connection show
```

To delete a connection, use

```
az ml connection delete
```