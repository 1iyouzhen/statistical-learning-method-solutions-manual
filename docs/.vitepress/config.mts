import { defineConfig } from 'vitepress'

// 1. 获取环境变量并判断
// 如果环境变量 EDGEONE 等于 '1'，说明在 EdgeOne 环境，使用根路径 '/'
// 否则默认是 GitHub Pages 环境，使用仓库子路径 '/easy-vecdb/'
const isEdgeOne = process.env.EDGEONE === '1'
const baseConfig = isEdgeOne ? '/' : '/tilly-figue/'

const guideSidebar = [
  { text: '首页', link: '/'},
  {
    text: '第1篇 监督学习',
    items: [
      { text: '第1章 统计学习方法概论', link: '/chapter01/ch01' },
      { text: '第2章 感知机', link: '/chapter02/ch02' },
      { text: '第3章 k近邻法', link: '/chapter03/ch03' },
      { text: '第4章 朴素贝叶斯法', link: '/chapter04/ch04' },
      { text: '第5章 决策树', link: '/chapter05/ch05' },
      { text: '第6章 Logistic回归与最大熵模型', link: '/chapter06/ch06' },
      { text: '第7章 支持向量机', link: '/chapter07/ch07' },
      { text: '第8章 提升方法', link: '/chapter08/ch08' },
      { text: '第9章 EM算法及其推广', link: '/chapter09/ch09' },
      { text: '第10章 隐马尔可夫模型', link: '/chapter10/ch10' },
      { text: '第11章 条件随机场', link: '/chapter11/ch11' }
    ]
  },
  {
    text: '第2篇 无监督学习',
    items: [
      { text: '第14章 聚类方法', link: '/chapter14/ch14' },
      { text: '第15章 奇异值分解', link: '/chapter15/ch15' },
      { text: '第16章 主成分分析', link: '/chapter16/ch16' },
      { text: '第17章 潜在语义分析', link: '/chapter17/ch17' },
      { text: '第18章 概率潜在语义分析', link: '/chapter18/ch18' },
      { text: '第19章 马尔可夫链蒙特卡罗法', link: '/chapter19/ch19' },
      { text: '第20章 潜在狄利克雷分配', link: '/chapter20/ch20' },
      { text: '第21章 PageRank算法', link: '/chapter21/ch21' }
    ]
  },
  {
    text: '第3篇 深度学习',
    items: [
      { text: '第23章 前馈神经网络', link: '/chapter23/ch23' },
      { text: '第24章 卷积神经网络', link: '/chapter24/ch24' },
      { text: '第25章 循环神经网络', link: '/chapter25/ch25' },
      { text: '第26章 序列到序列模型', link: '/chapter26/ch26' },
      { text: '第27章 预训练语言模型', link: '/chapter27/ch27' },
      { text: '第28章 生成对抗网络', link: '/chapter28/ch28' }
    ]
  }
]

export default defineConfig({
  lang: 'zh-CN',
  title: "机器学习方法习题解答",
  description: "机器学习方法习题解答",
  base: baseConfig,
  head: [
    ['link', { rel: 'icon', href: `${baseConfig === '/' ? '' : baseConfig}datawhale-logo.png` }]
  ],
  markdown: {
    math: true
  },
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    logo: '/tilly-figue-logo.png',

    nav: guideSidebar,
    
    sidebar: guideSidebar,

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: '搜索文档',
            buttonAriaLabel: '搜索文档'
          },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换'
            }
          }
        }
      }
    },
    

    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/statistical-learning-method-solutions-manual' }
    ],

    footer: {
      message: '<a href="https://beian.miit.gov.cn/" target="_blank">京ICP备2026002630号-1</a> | <a href="https://beian.mps.gov.cn/#/query/webSearch?code=11010602202215" rel="noreferrer" target="_blank">京公网安备11010602202215号</a>',
      copyright: '本作品采用 <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/" target="_blank">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议（CC BY-NC-SA 4.0）</a> 进行许可'
    },

    returnToTopLabel: '回到顶部',
    docFooter: {
      prev: '上一篇',
      next: '下一篇'
    },
  }
})
