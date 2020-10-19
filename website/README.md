# Website

This website is built using [Docusaurus 2](https://v2.docusaurus.io/), a modern static website generator.

##  Contributions

99% of contributions should only need the following:

- Add markdown files to the `website/docs` folder
- Update the `sidebar.js` file to add a page to the sidebar

If you need to do anything more than adding a new page to the sidebar (e.g.
modify the nav bar) then a) please refer to [Docusaurus 2](https://v2.docusaurus.io/).

## Installation

```console
yarn install
```

## Local Development

```console
yarn start
```

This command starts a local development server and open up a browser window. Most changes are reflected live without having to restart the server.

## Build

```console
yarn build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Deployment

```console
GIT_USER=<Your GitHub username> USE_SSH=true yarn deploy
```

If you are using GitHub pages for hosting, this command is a convenient way to build the website and push to the `gh-pages` branch.
