# -*- coding = utf-8 -*-
# @time:14/07/2023 11:05
# Author:Yunbo Long
# @File:Graph_making.py
# @Software:PyCharm
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.pyplot as plt
import numpy as np

# Example data (replace with your actual data)
# metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R-squared','Loss']
# mlp_results = [34.5, 27.82, 2887.35, 54.20, 0.84,2887.35]
# cnn_results = [29.52, 25.8, 2040.42, 45.17,0.89,2040]
# tf_results = [54.18, 45.9, 4940.29, 70.29, 0.7311,4938.75]

# metrics = ['MAE', 'MAPE', 'RMSE', 'R-squared',]
# mlp_results = [34.5, 27.82,  54.20, 0.84]
# cnn_results = [29.52, 25.8,  45.17,0.89]
# tf_results = [54.18, 45.9, 70.29, 0.7311]

metrics = ['MAE', 'RMSE', 'R-squared',]
mlp_results = [34.5,  54.20, 0.84]
cnn_results = [29.52,   45.17,0.89]
tf_results = [54.18,  70.29, 0.7311]

# Set the width of the bars
bar_width = 0.25

# Set the positions of the bars on the x-axis
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]
r3 = [x + bar_width for x in r2]

# Plot the bars
plt.bar(r1, mlp_results, color='skyblue', width=bar_width, edgecolor='black', label='MLP')
plt.bar(r2, cnn_results, color='lightgreen', width=bar_width, edgecolor='black', label='CNN')
plt.bar(r3, tf_results, color='orange', width=bar_width, edgecolor='black', label='TF')

# Add x-axis labels
plt.xlabel('Performance Metrics')
plt.xticks([r + bar_width for r in range(len(metrics))], metrics)

# Add y-axis label
plt.ylabel('Result')

# Add a title
# plt.title('Comparison in Centralised Learning')

# Add a legend
plt.legend()

# Show the plot
plt.show()

