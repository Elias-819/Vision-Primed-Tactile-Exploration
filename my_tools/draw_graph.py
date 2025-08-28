# -*- coding: utf-8 -*-
"""
基础折线图示例
pip 安装：pip install matplotlib
"""

import matplotlib.pyplot as plt

# 若中文标题/轴标签显示为方块，可尝试开启以下字体设置（你的系统需安装这些字体中的任意一个）
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 示例数据：把这两行换成你的 x/y 数据即可
x = list(range(1, 13))                                # X 轴（1~12 月）
y = [12, 15, 13, 18, 20, 22, 19, 25, 28, 26, 24, 30]  # Y 轴（示意值）

plt.figure(figsize=(10, 5))
plt.plot(x, y, marker='o', linewidth=2)  # 折线 + 圆点
plt.title('月度趋势')
plt.xlabel('月份')
plt.ylabel('数值')
plt.xticks(x)  # 保证每个点都有刻度
plt.grid(True, linestyle='--', alpha=0.5)

# 标注峰值（可选）
max_idx = max(range(len(y)), key=lambda i: y[i])
plt.scatter(x[max_idx], y[max_idx], s=64)
plt.text(x[max_idx], y[max_idx],
         f'峰值: {y[max_idx]}',
         ha='left', va='bottom')

plt.tight_layout()
plt.show()
