module.exports = {
  title: 'Azure Machine Learning (AML)',
  tagline: 'this website is under development',
  url: 'https://github.com/Azure/',
  baseUrl: '/azureml-examples/',
  onBrokenLinks: 'ignore',
  favicon: 'img/logo.svg',
  organizationName: 'Azure', // Usually your GitHub org/user name.
  projectName: 'azureml-examples', // Usually your repo name.
  themeConfig: {
    navbar: {
      title: 'Azure Machine Learning',
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
          to: 'docs/userguide/',
          label: 'User Guide',
          position: 'left',
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
              label: 'Examples (GitHub)',
              href: 'https://github.com/Azure/azureml-examples',
            },
            {
              label: 'Python SDK API',
              href: 'https://docs.microsoft.com/en-us/python/api/overview/azure/ml/?view=azure-ml-py'
            }
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/Azure/azureml-examples/issues',
            },
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
            }
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Microsoft`,
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
            'https://github.com/Azure/azureml-examples/tree/main/website/',
        },
        cookbook: {
          sidebarPath: require.resolve('./sidebars.js'),
        },
        blog: {
          showReadingTime: true,
          // Please change this to your repo.
          editUrl:
            'https://github.com/Azure/azureml-examples/tree/main/website/blog',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
