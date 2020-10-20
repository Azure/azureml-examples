module.exports = {
  title: 'Azure ML Cheat Sheet',
  tagline: '80% of what you need in 20% of the documentation',
  url: 'https://github.com/Azure/',
  baseUrl: '/azureml-examples/',
  onBrokenLinks: 'ignore',
  favicon: 'img/logo.svg',
  organizationName: 'Azure', // Usually your GitHub org/user name.
  projectName: 'azureml-examples', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Azure ML Cheat Sheet',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: 'docs/',
          label: 'Cheat Sheet',
          position: 'left',
        },
        {
          to: 'docs/vs-code-snippets/snippets',
          label: 'Snippets',
          position: 'left'
        },
        {
          to: 'docs/',
          label: 'Documentation',
          position: 'left'
        },
        {
          href: 'https://github.com/Azure/azureml-examples',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {
              label: 'Microsoft Docs',
              href: 'https://docs.microsoft.com/azure/machine-learning',
            },
            {
              label: 'Azure ML Examples (GitHub)',
              href: 'https://github.com/Azure/azureml-examples',
            },
            {
              label: 'AzureML Python SDK API',
              href: 'https://docs.microsoft.com/python/api/overview/azure/ml/?view=azure-ml-py'
            }
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.microsoft.com/questions/tagged/10888',
            }
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Blog',
              to: 'blog',
            },
            {
              label: 'GitHub',
              href: 'https://github.com/Azure/azureml-examples/',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Microsoft // Built with Docusaurus2`,
    },
    algolia: {
      apiKey: 'd4ee9b6c7a8efe0a93f6455726bf8bbe',
      indexName: 'azureml_cheatsheet',
      searchParameters: {},
      placeholder: 'Search cheat sheet'
    },
    googleAnalytics: {
      trackingID: 'G-2DKKZ26VP0',
    }
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://github.com/Azure/azureml-examples/edit/website/website/',
        },
        cookbook: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: {
          showReadingTime: true,
          editUrl:
            'https://github.com/Azure/azureml-examples/edit/website/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
