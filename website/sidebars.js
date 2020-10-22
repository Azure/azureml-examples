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
  userguide:{
    'User Guide': [
      {
        type: 'doc',
        id: 'userguide/README'
      },
      {
        type: 'category',
        label: 'Overview',
        collapsed: false,
        items: ['userguide/overview', 'userguide/studio', 'userguide/architecture']
      },
      {
        type: 'category',
        label: 'Concepts',
        collapsed: false,
        items: ['userguide/workspace', 'userguide/job', 'userguide/endpoint', 'userguide/data', 'userguide/compute', 'userguide/model']
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
