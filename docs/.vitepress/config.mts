import { resolve } from 'node:path'
import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'
import { sidebar } from './sidebar.mts'
import { codeLinksPlugin, serveCodeExamples } from './plugins/code-link-rewrite'

const isDev = process.env.NODE_ENV !== 'production'
const projectRoot = resolve(__dirname, '../..')

export default withMermaid(
  defineConfig({
    title: 'AI 知识库',
    description: 'AI 从入门到实战 — 面向后端开发者的 AI 全栈学习知识库',
    lang: 'zh-CN',
    base: '/guide-ai/',

    themeConfig: {
      nav: [
        { text: '首页', link: '/' },
        { text: '学习路径', link: '/learning-paths/fast-track' },
        { text: '面试', link: '/interview/' },
        {
          text: '编码线',
          items: [
            { text: '前提准备', link: '/0-prerequisites/' },
            { text: 'AI/ML 基础', link: '/1-ml-basics/' },
            { text: '大语言模型', link: '/2-llm/' },
            { text: 'AI 应用开发', link: '/3-ai-apps/' },
            { text: '计算机视觉', link: '/4-cv/' },
            { text: 'AI 工程化', link: '/5-ai-engineering/' },
            { text: 'AI 前沿', link: '/6-ai-frontier/' },
          ],
        },
        {
          text: 'AI 工具',
          items: [
            { text: 'AI 效率工具', link: '/7-ai-tools/7.1-efficiency/' },
            { text: 'AIGC 内容创作', link: '/7-ai-tools/7.2-aigc/' },
            { text: 'AI 商业变现', link: '/7-ai-tools/7.3-business/' },
          ],
        },
      ],

      sidebar,

      search: {
        provider: 'local',
      },

      socialLinks: [
        { icon: 'github', link: 'https://github.com/skyhe58/guide-ai' },
      ],

      outline: {
        label: '页面导航',
      },

      docFooter: {
        prev: '上一页',
        next: '下一页',
      },

      lastUpdated: {
        text: '最后更新于',
      },

      returnToTopLabel: '回到顶部',
      sidebarMenuLabel: '菜单',
      darkModeSwitchLabel: '主题',
      lightModeSwitchTitle: '切换到浅色模式',
      darkModeSwitchTitle: '切换到深色模式',
    },

    markdown: {
      lineNumbers: true,
      config: (md) => {
        // 仅 dev 模式下重写代码链接为本地路径
        if (isDev) {
          codeLinksPlugin(md)
        }
      },
    },

    // 渐进式构建：后续模块页面尚未创建，暂时忽略死链接
    ignoreDeadLinks: true,

    lastUpdated: true,

    vite: {
      optimizeDeps: {
        include: ['mermaid', 'dayjs'],
      },
      plugins: [
        // dev 模式下提供本地 code-examples 文件服务
        ...(isDev ? [serveCodeExamples(projectRoot)] : []),
      ],
    },
  })
)
