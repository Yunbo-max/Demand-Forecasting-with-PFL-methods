# -*- coding = utf-8 -*-
# @time:17/07/2023 09:07
# Author:Yunbo Long
# @File:local_total_after_K-means.py
# @Software:PyCharm
# -*- coding = utf-8 -*-
# @time:17/07/2023 08:44
# Author:Yunbo Long
# @File:local_total_after.py
# @Software:PyCharm
import numpy as np
import matplotlib.pyplot as plt

# Region names
region_map = {
    # 0: 'Southeast Asia',
    # 1: 'South Asia',
    # 2: 'Oceania',
    # 3: 'Eastern Asia',
    # 5: 'West of USA',
    # 6: 'US Center',
    # 7: 'West Africa',
    # 9: 'North Africa',
    # 10: 'Western Europe',
    12: 'Central America',
    14: 'South America',
    # 16: 'Southern Europe',
    17: 'East of USA',
    22: 'South of USA'
}

# Example data (replace with your actual data)
clients = list(region_map.values())
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R-squared']

# MLP algorithm metrics for each client
mlp_metrics = [
    # [32.66, 3.5, 0.0272, 51.05, 0.87],
    # [35.51, 12.082, 0.2579, 54.28, 0.83],
    # [36.86, 1.701, 0.006045, 54.54, 0.805],
    # [38.30, 1.435, 0.0088234, 55.36, 0.87],
    # [37.90, 1.35, 0.00513, 55.99, 0.749],
    # [38.34, 1.704, 0.008572, 55, 0.75],
    # [41.63, 3.05, 0.01958, 60.67, 0.679],
    # [41.99, 2.105, 0.01402, 60.22, 0.696],
    # [37.79, 0.6459, 0.001202, 56.45, 0.87],
    [40.67, 0.6457, 0.0027, 56.25, 0.74],
    [43.49, 0.8766, 0.002168, 58.86, 0.71],
    # [39.78, 1.012, 0.004945, 57.43, 0.878],
    [56.90, 0.9644, 0.00235, 68.08, 0.62],
    [39.66, 1.315, 0.004101, 58.92, 0.705],
    # ... Add metrics for other clients
]

# CNN algorithm metrics for each client
cnn_metrics = [
    # [37.47, 6.387, 0.094414, 54.54, 0.852],
    # [43.52, 4.707, 0.05722, 61.82, 0.783],
    # [37.42, 4.511, 0.03838, 55.56, 0.80],
    # [38.76, 7.405, 0.07089, 54.96, 0.873],
    # [36.79, 9.188, 0.09265, 53.64, 0.77],
    # [37.11, 7.908, 0.08906, 51.95, 0.78],
    # [46.11, 8.858, 0.08831, 62.45, 0.66],
    # [56.47, 9.059, 0.1173, 75.46, 0.53],
    # [36.14, 5.933, 0.03257, 54.8, 0.882],
    [32.02, 2.557, 0.01554, 47.81, 0.82],
    [38.91, 4.072, 0.03055, 54.27, 0.76],
    # [47.89, 8.484, 0.07461, 63.42, 0.851],
    [40.44, 4.361, 0.04221, 55.7, 0.75],
    [43.16, 5.917, 0.05722, 59.84, 0.70],
    # ... Add metrics for other clients
]

# TF algorithm metrics for each client
tf_metrics = [
    # [55.23, 21.962, 0.1936, 76.47, 0.71],
    # [54.625, 23.733, 0.2404, 74.6516, 0.69],
    # [53.98, 18.095, 0.1562, 70.97, 0.671],
    # [57.98, 20.502, 0.2301, 80.78, 0.725],
    # [54.37, 22.606, 0.2514, 70.74, 0.61],
    # [53.99, 23.109, 0.2802, 69.38, 0.61],
    # [55.69, 21.311, 0.3451, 71.73, 0.56],
    # [56.89, 25.477, 0.3839, 73.67, 0.55],
    # [59.91, 18.096, 0.1187, 78.53, 0.76],
    [55.03, 17.259, 0.1115, 71.53, 0.59],
    [55.51, 18.27, 0.1311, 70.94, 0.60],
    # [62.47, 19.26, 0.1464, 87.34, 0.71],
    [56.02, 23.52, 0.2483, 70.91, 0.59],
    [53.29, 24.247, 0.3021, 69.22, 0.58],
    # ... Add metrics for other clients
]

# MPL-FL algorithm metrics for each client
mlp_fl_metrics = [
    # [55.23, 21.962, 0.1936, 105.662, 0.71],
    # [54.625, 23.733, 0.2404, 98.393, 0.69],
    # [53.98, 18.095, 0.1562, 97.874, 0.671],
    # [57.98, 20.502, 0.2301, 121.892, 0.725],
    # [54.37, 22.606, 0.2514, 181.255, 0.61],
    # [53.99, 23.109, 0.2802, 178.856, 0.61],
    # [55.69, 21.311, 0.3451, 173.906, 0.56],
    # [56.89, 25.477, 0.3839, 179.821, 0.55],
    # [59.91, 18.096, 0.1187, 141.244, 0.76],
    [28.543, 17.259, 0.1115, 43.989, 0.844],
    [28.226, 18.27, 0.1311, 43.765, 0.8429],
    # [62.47, 19.26, 0.1464, 142.694, 0.71],
    [28.123, 23.52, 0.2483, 43.661, 0.8452],
    [27.485, 24.247, 0.3021, 44.145, 0.8346],
    # ... Add metrics for other clients
]

# CNN-FL algorithm metrics for each client
cnn_fl_metrics = [
    # [55.23, 21.962, 0.1936, 98.785, 0.71],
    # [54.625, 23.733, 0.2404, 89.116, 0.69],
    # [53.98, 18.095, 0.1562, 85.69, 0.671],
    # [57.98, 20.502, 0.2301, 99.866, 0.725],
    # [54.37, 22.606, 0.2514, 166.774, 0.61],
    # [53.99, 23.109, 0.2802, 163.885, 0.61],
    # [55.69, 21.311, 0.3451, 171.826, 0.56],
    # [56.89, 25.477, 0.3839, 168.148, 0.55],
    # [59.91, 18.096, 0.1187, 131.988, 0.76],
    [25.973, 17.259, 0.1115, 43.06, 0.852],
    [27.081, 18.27, 0.1311, 44.51, 0.8531],
    # [62.47, 19.26, 0.1464, 130.707, 0.71],
    [28.597, 23.52, 0.2483, 44.495, 0.8369],
    [28.355, 24.247, 0.3021, 44.673, 0.8317],
    # ... Add metrics for other clients
]

# TF-FL algorithm metrics for each client
tf_fl_metrics = [
    # [55.23, 21.962, 0.1936, 86.71, 0.71],
    # [54.625, 23.733, 0.2404, 81.998, 0.69],
    # [53.98, 18.095, 0.1562, 77.162, 0.671],
    # [57.98, 20.502, 0.2301, 91.645, 0.725],
    # [54.37, 22.606, 0.2514, 82.226, 0.61],
    # [53.99, 23.109, 0.2802, 82.177, 0.61],
    # [55.69, 21.311, 0.3451, 85.842, 0.56],
    # [56.89, 25.477, 0.3839, 86.835, 0.55],
    # [59.91, 18.096, 0.1187, 105.467, 0.76],
    [52.87, 17.259, 0.1115, 68.803, 0.6183],
    [53.132, 18.27, 0.1311, 68.41, 0.6161],
    # [62.47, 19.26, 0.1464, 106.355, 0.71],
    [52.795, 23.52, 0.2483, 68.1, 0.6245],
    [52.176, 24.247, 0.3021,67.38, 0.6188],
    # ... Add metrics for other clients
]

# Set the width of the bars
bar_width = 0.12
padding = 0.

r1 = np.arange(len(clients))
r2 = [x + bar_width + padding for x in r1]
r3 = [x + bar_width + padding for x in r2]
r4 = [x + bar_width + padding for x in r3]
r5 = [x + bar_width + padding for x in r4]
r6 = [x + bar_width + padding for x in r5]



patterns = ['/', '\\', '-', 'O', '.', '*']
colors = ['lightgreen', 'orange',  'yellow', 'red']

plt.figure(figsize=(10, 6))  # Set the size of the plot to 10 inches by 6 inches


for i, client in enumerate(clients):
    # MLP and MLP-FL bars with the same pattern
    plt.bar(r1[i], mlp_metrics[i][3], color='lightgrey', hatch=patterns[0], width=bar_width, edgecolor='black',
            label='MLP' if i == 0 else "")
    plt.bar(r2[i], mlp_fl_metrics[i][3], color=colors[0], hatch=patterns[0], width=bar_width, edgecolor='black',
            label='MLP-FL' if i == 0 else "")
    # CNN and CNN-FL bars with the same pattern
    plt.bar(r3[i], cnn_metrics[i][3], color='lightgrey', hatch=patterns[1], width=bar_width, edgecolor='black',
            label='CNN' if i == 0 else "")
    plt.bar(r4[i], cnn_fl_metrics[i][3], color=colors[1], hatch=patterns[1], width=bar_width, edgecolor='black',
            label='CNN-FL' if i == 0 else "")
    # TF and TF-FL bars with the same pattern
    plt.bar(r5[i], tf_metrics[i][3], color='lightgrey', hatch=patterns[2], width=bar_width, edgecolor='black',
            label='TF' if i == 0 else "")
    plt.bar(r6[i], tf_fl_metrics[i][3], color=colors[2], hatch=patterns[2], width=bar_width, edgecolor='black',
            label='TF-FL' if i == 0 else "")

plt.xlabel('Clients', fontsize=14)  # Update fontsize value as needed
plt.xticks([r + bar_width for r in range(len(clients))], clients, fontsize=14)  # Update fontsize value as needed

plt.ylabel('RMSE Metric Value', fontsize=14)  # Update fontsize value as needed

plt.title('Collaboration between Central America, South America, Eastern USA, and South USA', fontsize=14)  # Update fontsize value as needed

plt.legend(fontsize=14)  # Update fontsize value as needed

plt.tight_layout()
plt.show()

