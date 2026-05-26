// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  site: 'https://kiankyars.github.io',
  base: '/intuit-rl-envs',
  integrations: [
    starlight({
      title: 'Six Axes of an RL Environment',
      description: 'A worklog on how the shape of an RL environment determines what a post-trained policy actually learns.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/kiankyars/intuit-rl-envs' },
      ],
      sidebar: [
        { label: 'Overview', slug: 'index' },
        {
          label: 'Axes',
          items: [
            { label: '1. Gameability', slug: '01-gameability' },
            { label: '2. Verifiability', slug: '02-verifiability' },
            { label: '3. Shape', slug: '03-shape' },
            { label: '4. Horizon', slug: '04-horizon' },
            { label: '5. Escalation', slug: '05-escalation' },
            { label: '6. Reversibility', slug: '06-reversibility' },
          ],
        },
        {
          label: 'Appendix',
          items: [{ label: 'POMDP mapping', slug: 'appendix-pomdp' }],
        },
      ],
      customCss: ['./src/styles/custom.css'],
    }),
    react(),
  ],
});
