// module.exports = {
//   mainSidebar: {
//     'Getting Started': ['installation', 'cheatsheet'],
//     'Basic Assets': ['workspace', 'compute-targets', 'environment'],
//     'Common Scenarios': ['train-cloud'],
//     'Advanced Scenarios': ['docker-build', 'distributed-training'],
//   }
// };

module.exports = {
  mainSidebar: {
    'Menu': [
      {
        type: 'doc',
        id: 'cheatsheet'
      },
      {
        type: 'category',
        label: 'Getting Started',
        collapsed: false,
        items: ['installation'],
      },
      {
        type: 'category',
        label: 'Basic Assets',
        collapsed: false,
        items: ['workspace', 'compute-targets', 'environment', 'data'],
      },
      {
        type: 'category',
        label: 'Submitting Code',
        collapsed: false,
        items: ['run', 'script-run-config', 'logging'],
      },
      {
        type: 'category',
        label: 'Advanced Guides',
        collapsed: false,
        items: ['distributed-training', 'docker-build']
      }
    ],
  },
  secondaySidebar: {
    Cookbook: [
      {
        type: 'doc',
        id: 'cbdocs/cookbook',
      },
      {
        type: 'category',
        label: 'Setup',
        items: ['cbdocs/setup-sdk', 'cbdocs/setup-notebook'],
      }
    ]
  }
};
