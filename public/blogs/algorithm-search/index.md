# Advanced For-Loop for QML algorithm search
- 作者: FuTe Wong
- 提交年份: 2025年6月23日（v1）
- 出处: arXiv preprint, arXiv:2506.18260v1, 分类 cs.AI

## 研究内容
基于大型语言模型的多代理系统(LLMMA)来自动搜索和优化量子机器学习(QML)算法。

## LLMMA QML算法搜索方式
给定经典深度学习算法(如前向算法或反向传播算法)，系统可以促进开发工作流程找到其优化的量子对等体。
![](/blogs/algorithm-search/cb66e213ad066503.png)
<div align="center"> QML搜索流程流程图</div>

实验以三种经典算法为输入，测试系统的量子转换能力：                                                                                                                                                                                                                                                        
1.	多层感知机Multi-Layer Perceptron (MLP) → Quantum MLP：展示模型确实在学习（训练动态为正）
|  |  |
| ---- | ---- |
| ![](/blogs/algorithm-search/ae89d9157a04d208.png) | ![](/blogs/algorithm-search/1065bf47c077423c.png) |