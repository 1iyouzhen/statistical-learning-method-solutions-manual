---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "机器学习方法习题解答"
  text: '李航《机器学习方法》<span class="best-partner">最佳伴侣</span>'
  tagline: 全书习题详解 + Python代码复现 + 深度原理解析，助你从零构建机器学习知识体系。
  image:
    src: /machine-learning-method-book.png
    alt: 机器学习方法习题解答
  actions:
    - theme: brand
      text: 第一阶段：监督学习
      link: /chapter01/ch01
    - theme: alt
      text: 第二阶段：无监督学习
      link: /chapter14/ch14
    - theme: alt
      text: 第三阶段：深度学习
      link: /chapter23/ch23

features:
  - title: 📚 习题详解
    details: 覆盖监督学习、无监督学习与深度学习，提供详尽的课后习题解答。
  - title: 💻 代码实战
    details: 提供 Python/PyTorch 代码实现，理论结合实践，拒绝纸上谈兵。
  - title: 📊 图文并茂
    details: 结合可视化图表与详细推导，直观展示算法原理与运行过程。
---

<div class="learning-path" style="margin-top: 40px; margin-bottom: 40px;">
  <h2 align="center" style="border-bottom: none; margin-bottom: 30px;">🗺️ 自学路径规划</h2>
  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <div class="learning-path-card lp-card-supervised">
      <h3 style="margin-top: 0; display: flex; align-items: center;">🟢 第一阶段：监督学习</h3>
      <p style="color: var(--vp-c-text-2); margin-bottom: 8px;"><strong>⏱️ 建议时长</strong>：4-6 周</p>
      <p style="color: var(--vp-c-text-2); margin-bottom: 16px;"><strong>目标</strong>：掌握机器学习的核心思想与基础算法，建立数学直觉。</p>
      <ul style="padding-left: 20px; margin: 0;">
        <li><strong>核心章节</strong>：第1章（概论）、第2-5章（感知机、KNN、贝叶斯、决策树）、第6-7章（LR、SVM）。</li>
        <li><strong>关键任务</strong>：手动推导 SVM 对偶问题，用 Python 实现决策树构建。</li>
      </ul>
    </div>
    <div class="learning-path-card lp-card-unsupervised">
      <h3 style="margin-top: 0; display: flex; align-items: center;">🔵 第二阶段：无监督</h3>
      <p style="color: var(--vp-c-text-2); margin-bottom: 8px;"><strong>⏱️ 建议时长</strong>：3-5 周</p>
      <p style="color: var(--vp-c-text-2); margin-bottom: 16px;"><strong>目标</strong>：理解数据内在结构，掌握概率图模型与降维方法。</p>
      <ul style="padding-left: 20px; margin: 0;">
        <li><strong>核心章节</strong>：第14章（聚类）、第15-16章（SVD、PCA）、第19-20章（MCMC、LDA）。</li>
        <li><strong>关键任务</strong>：使用 PCA 进行数据降维可视化，手动实现 EM 算法迭代过程。</li>
      </ul>
    </div>
    <div class="learning-path-card lp-card-deeplearning">
      <h3 style="margin-top: 0; display: flex; align-items: center;">🔴 第三阶段：深度学习</h3>
      <p style="color: var(--vp-c-text-2); margin-bottom: 8px;"><strong>⏱️ 建议时长</strong>：3-4 周</p>
      <p style="color: var(--vp-c-text-2); margin-bottom: 16px;"><strong>目标</strong>：衔接现代 AI 技术，掌握神经网络与深度学习框架。</p>
      <ul style="padding-left: 20px; margin: 0;">
        <li><strong>核心章节</strong>：第23章（前馈网络）、第24-25章（CNN、RNN）、第27章（Transformer）。</li>
        <li><strong>关键任务</strong>：基于 PyTorch 构建 CNN 进行图像分类，理解 Attention 机制。</li>
      </ul>
    </div>
  </div>
</div>

<div class="target-audience" style="margin-top: 40px; margin-bottom: 40px;">
  <h2 align="center" style="border-bottom: none; margin-bottom: 30px;">👥 项目受众</h2>
  <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
    <div class="learning-path-card lp-card-supervised">
      <h3 style="margin-top: 0; display: flex; align-items: center;">🌱 机器学习初学者</h3>
      <p style="color: var(--vp-c-text-2);">正在学习李航老师《统计学习方法》或《机器学习方法》的同学，希望通过习题解答辅助理解。</p>
    </div>
    <div class="learning-path-card lp-card-unsupervised">
      <h3 style="margin-top: 0; display: flex; align-items: center;">💻 算法工程师/开发者</h3>
      <p style="color: var(--vp-c-text-2);">希望深入理解机器学习算法原理，并寻找相关算法的 Python 实现代码（如感知机、决策树、SVM、Transformer 等）的开发者。</p>
    </div>
    <div class="learning-path-card lp-card-deeplearning">
      <h3 style="margin-top: 0; display: flex; align-items: center;">🚀 备战考研/求职者</h3>
      <p style="color: var(--vp-c-text-2);">需要系统复习机器学习基础理论和推导细节，巩固知识体系的同学。</p>
    </div>
  </div>
</div>
