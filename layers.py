import matplotlib.pyplot as plt

# layers和NYT值
layers = [200, 250, 300, 350, 400]
nyt_values = [0.72, 0.75, 0.78, 0.80, 0.82]

# ARI和FMI值


ari_values = [0.768, 0.773, 0.750, 0.752, 0.740]
fmi_values = [0.775, 0.782, 0.765, 0.761, 0.752]

# 绘制折线图
plt.plot(layers, ari_values, marker='o', label='ARI')
plt.plot(layers, fmi_values, marker='o', label='FMI')

# 设置坐标轴标签
plt.xlabel('Layers')
plt.ylabel('PubMed')

# 设置纵坐标范围
plt.ylim(0.72, 0.82)

# 设置图例
plt.legend()

# 启用网格
plt.grid(True)

# 显示图形
plt.show()