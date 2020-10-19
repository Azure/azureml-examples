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
        // {position: 'left', type: 'docsVersionDropdown'},
        // {
        //   to: 'docs/cbdocs/cookbook',
        //   label: 'Cookbook',
        //   position: 'left',
        // },
        {to: 'docs/vs-code-snippets/snippets', label: 'Snippets', position: 'left'},
        // {to: 'blog', label: 'Blog', position: 'left'},
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
          title: 'Docs',
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
              href: 'https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py'
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
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/master/website/',
        },
        cookbook: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/facebook/docusaurus/edit/master/website/blog/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
