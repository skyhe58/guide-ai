/**
 * VitePress 代码链接环境切换插件
 *
 * 原理：
 * - 生产构建 (pnpm build)：链接保持 GitHub URL 不变（语法高亮、目录浏览）
 * - 本地开发 (pnpm dev)：markdown-it 插件将 GitHub URL 重写为 /code-examples/...
 *   同时 Vite 中间件提供本地 code-examples 目录的文件服务
 *
 * markdown 中统一写法：
 *   [代码](https://github.com/<owner>/<repo>/tree/main/code-examples/...)
 *
 * 适配其他项目只需修改下方 GITHUB_BASE_URL 常量。
 */
import type MarkdownIt from 'markdown-it'
import type { Plugin } from 'vite'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { extname, resolve } from 'node:path'

// ─── 配置项（适配其他项目时修改这里）────────────────────
export const GITHUB_BASE_URL = 'https://github.com/skyhe58/guide-ai/tree/main/'
export const LOCAL_CODE_PREFIX = '/code-examples/'

// ─── 1. Markdown-it 插件：dev 模式下重写代码链接 ─────────
export function codeLinksPlugin(md: MarkdownIt): void {
  const defaultRender =
    md.renderer.rules.link_open ||
    function (tokens, idx, options, _env, self) {
      return self.renderToken(tokens, idx, options)
    }

  md.renderer.rules.link_open = function (tokens, idx, options, env, self) {
    const token = tokens[idx]
    const hrefIdx = token.attrIndex('href')

    if (hrefIdx >= 0) {
      const href = token.attrs![hrefIdx][1]
      if (href.startsWith(GITHUB_BASE_URL + 'code-examples/')) {
        // GitHub URL → 本地路径
        const relativePath = href.replace(GITHUB_BASE_URL, '')
        token.attrs![hrefIdx][1] = '/' + relativePath
        // 新标签页打开，方便对照文档和代码
        token.attrSet('target', '_blank')
      }
    }
    return defaultRender(tokens, idx, options, env, self)
  }
}

// ─── 2. Vite 插件：dev server 提供 code-examples 文件 ────
const MIME_MAP: Record<string, string> = {
  '.py': 'text/x-python; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.ts': 'text/typescript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.yaml': 'text/yaml; charset=utf-8',
  '.yml': 'text/yaml; charset=utf-8',
  '.toml': 'text/toml; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.txt': 'text/plain; charset=utf-8',
  '.sh': 'text/x-shellscript; charset=utf-8',
}

export function serveCodeExamples(projectRoot: string): Plugin {
  const codeRoot = resolve(projectRoot, 'code-examples')

  return {
    name: 'vitepress-serve-code-examples',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(LOCAL_CODE_PREFIX)) {
          return next()
        }

        const relativePath = decodeURIComponent(
          req.url.replace(LOCAL_CODE_PREFIX, '').split('?')[0]
        )
        const filePath = resolve(codeRoot, relativePath)

        // 安全：防止路径遍历攻击
        if (!filePath.startsWith(codeRoot)) {
          res.statusCode = 403
          res.end('Forbidden')
          return
        }

        if (!existsSync(filePath)) {
          res.statusCode = 404
          res.end(`Not Found: ${relativePath}`)
          return
        }

        const stat = statSync(filePath)

        // 目录：返回文件列表
        if (stat.isDirectory()) {
          const files = readdirSync(filePath)
          const list = files
            .map((f) => {
              const slash = relativePath.endsWith('/') || relativePath === '' ? '' : '/'
              return `<li><a href="${LOCAL_CODE_PREFIX}${relativePath}${slash}${f}">${f}</a></li>`
            })
            .join('\n')
          res.setHeader('Content-Type', 'text/html; charset=utf-8')
          res.end(`<!DOCTYPE html><html><head><meta charset="utf-8">
<title>code-examples/${relativePath}</title>
<style>body{font-family:monospace;padding:2em;line-height:1.8}a{color:#0366d6}</style>
</head><body><h2>code-examples/${relativePath}</h2><ul>${list}</ul></body></html>`)
          return
        }

        // 文件：返回源代码
        const ext = extname(filePath)
        const contentType = MIME_MAP[ext] || 'text/plain; charset=utf-8'
        res.setHeader('Content-Type', contentType)
        res.end(readFileSync(filePath, 'utf-8'))
      })
    },
  }
}
