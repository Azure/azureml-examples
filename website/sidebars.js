module.exports = {
  mainSidebar: {
    'Menu': [
      {
        type: 'doc',
        id: 'cheatsheet/cheatsheet'
      },
      {
        type: 'category',
        label: 'Getting Started',
        collapsed: false,
        items: ['cheatsheet/installation'],
      },
      {
        type: 'category',
        label: 'Basic Assets',
        collapsed: false,
        items: ['cheatsheet/workspace', 'cheatsheet/compute-targets', 'cheatsheet/environment', 'cheatsheet/data'],
      },
      {
        type: 'category',
        label: 'Submitting Code',
        collapsed: false,
        items: ['cheatsheet/script-run-config', 'cheatsheet/logging'],
      },
      {
        type: 'category',
        label: 'Advanced Guides',
        collapsed: false,
        items: ['cheatsheet/distributed-training', 'cheatsheet/docker-build']
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
