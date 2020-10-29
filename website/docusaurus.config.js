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
          to: 'docs/cheatsheet/',
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
        // {
        //   to: 'docs/userguide/',
        //   label: 'User Guide',
        //   position: 'left',
        // },
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
          title: 'Reference',
          items: [
            {
              label: 'Microsoft Docs',
              href: 'https://docs.microsoft.com/azure/machine-learning',
            },
            {
              label: 'GitHub Examples',
              href: 'https://github.com/Azure/azureml-examples',
            },
            {
              label: 'Python SDK Docs',
              href: 'https://docs.microsoft.com/python/api/overview/azure/ml/?view=azure-ml-py'
            }
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'GitHub (Issues)',
              href: 'https://github.com/Azure/azureml-examples/issues',
            },
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.microsoft.com/questions/tagged/10888',
            }
          ],
        },
        {
          title: 'Coming soon...',
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
      apiKey: 'b12ff2d7b13980e0983244167d1c2450',
      indexName: 'azure',
      searchParameters: {},
      placeholder: 'Search cheat sheet'
    },
    googleAnalytics: {
      trackingID: 'UA-83747202-1',
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
