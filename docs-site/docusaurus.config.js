const config = {
  title: 'FICOS Documentation',
  tagline: 'Freight Intelligence and Chartering Optimization System',
  favicon: 'img/favicon.ico',
  url: 'https://ficos-docs.netlify.app',
  baseUrl: '/',
  organizationName: 'ficos',
  projectName: 'ficos-platform',
  trailingSlash: false,
  onBrokenLinks: 'warn',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      },
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'FICOS',
      logo: {
        alt: 'FICOS',
        src: 'img/logo.svg',
      },
      items: [
        { type: 'docSidebar', sidebarId: 'mainSidebar', position: 'left', label: 'Docs' },
        { href: 'https://ficos-demo.netlify.app', label: 'Frontend Demo', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Project',
          items: [
            { label: 'Architecture', to: '/project-docs/architecture' },
            { label: 'Model Selection', to: '/project-docs/model_selection' },
            { label: 'Security & Deployment', to: '/deployment/security' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} FICOS.`,
    },
    prism: {
      additionalLanguages: ['python', 'bash', 'json'],
    },
  },
};

export default config;
