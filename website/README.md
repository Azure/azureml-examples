# Website

![website](https://github.com/Azure/azureml-examples/workflows/website/badge.svg)

Website is available here: https://azure.github.io/azureml-examples/

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator.

##  Contributions

Make PR's against the `website` branch

```bash
git clone git@github.com:Azure/azureml-examples.git
git checkout website
git checkout -b user/contrib
```

- When a PR arrives against this branch GitHub actions will test the build is successful
- When the PR is merged to `website` (or when someone pushes directly to `website`)
the change will be automatically deployed to `gh-pages` branch (and the webpage will be updated automatically)

99% of contributions should only need the following:

- Add markdown files to the `website/docs` folder
- Update the `sidebar.js` file to add a page to the sidebar

If you need to do anything more than adding a new page to the sidebar (e.g.
modify the nav bar) then a) please refer to [Docusaurus 2](https://v2.docusaurus.io/).

## Development (testing locally)

- Install npm and yarn: see [docusaurus2 webpage](https://v2.docusaurus.io/docs/installation)

- First time Docusaurus2 installation
    ```bash
    cd website
    npm install
    ```

- Run local server while developing:
    ```bash
    cd website
    yarn start
    ```

## Deployment

This repo has GitHub actions in place that automate deployment. If you are interested in how
deployment works then read on :-)

### GitHub Actions

We use GitHub actions to automate deployment. Set up was as follows:

- Generated new SSH key
    - NB. Since there was an existing ssh key tied the repo a new key was generated (in a different location) `/tmp/.ssh/id_rsa`
- Add public key to repo's [deploy key](https://developer.github.com/v3/guides/managing-deploy-keys/)
    - NB. Allow write access
- Add private key as [GitHub secret](https://help.github.com/en/actions/configuring-and-managing-workflows/creating-and-storing-encrypted-secrets)
    - We use repo-level (not org level) secret
    - Secret is named `GH_PAGES_DEPLOY`
    - `xclip -sel clip < /tmp/.ssh/id_rsa`

### Manual

It is possible to make manual deployments without use of the GitHub action above.

```console
GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.

