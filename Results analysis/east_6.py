# -*- coding = utf-8 -*-
# @time:17/07/2023 08:44
# Author:Yunbo Long
# @File:local_total_after.py
# @Software:PyCharm
import numpy as np
import matplotlib.pyplot as plt

# Region names
region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    # 2: 'Oceania',
    3: 'Eastern Asia',
    # 5: 'West of USA',
    # 6: 'US Center',
    # 7: 'West Africa',
    # 9: 'North Africa',
    10: 'Western Europe',
    # 12: 'Central America',
    # 14: 'South America',
    16: 'Southern Europe',
    # 17: 'East of USA',
    # 22: 'South of USA'
}



# Example data (replace with your actual data)
clients = list(region_map.values())
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R-squared']

# MLP algorithm metrics for each client
mlp_metrics = [
    [32.66, 3.5, 0.0272, 51.05, 0.87],
    [35.51, 12.082, 0.2579, 54.28, 0.83],
    # [36.86, 1.701, 0.006045, 54.54, 0.805],
    [38.30, 1.435, 0.0088234, 55.36, 0.87],
    # [37.90, 1.35, 0.00513, 55.99, 0.749],
    # [38.34, 1.704, 0.008572, 55, 0.75],
    # [41.63, 3.05, 0.01958, 60.67, 0.679],
    # [41.99, 2.105, 0.01402, 60.22, 0.696],
    [37.79, 0.6459, 0.001202, 56.45, 0.87],
    # [40.67, 0.6457, 0.0027, 56.25, 0.74],
    # [43.49, 0.8766, 0.002168, 58.86, 0.71],
    [39.78, 1.012, 0.004945, 57.43, 0.878],
    # [56.90, 0.9644, 0.00235, 68.08, 0.62],
    # [39.66, 1.315, 0.004101, 58.92, 0.705],
    # # ... Add metrics for other clients
]


# # MPL-FL algorithm metrics for each client
# mlp_fl_metrics6 = [
#     [39.761, 54.54, 0.1936, 54.54, 0.8518],
#     [41.994, 57.344, 0.2404, 57.344, 0.8136],
#     [40.957, 56.375, 0.1562, 56.375, 0.7921],
#     [45.259, 63.049, 0.2301, 63.049, 0.8323],
#     # [54.37, 22.606, 0.2514, 181.255, 0.61],
#     # [53.99, 23.109, 0.2802, 178.856, 0.61],
#     # [55.69, 21.311, 0.3451, 173.906, 0.56],
#     # [56.89, 25.477, 0.3839, 179.821, 0.55],
#     [43.205, 18.096, 0.1187, 62.048, 0.8484],
#     # [55.03, 17.259, 0.1115, 186.88, 0.59],
#     # [55.51, 18.27, 0.1311, 183.141, 0.60],
#     [41.474, 19.26, 0.1464, 60.27, 0.8659],
#     # [56.02, 23.52, 0.2483, 183.138, 0.59],
#     # [53.29, 24.247, 0.3021, 183.005, 0.58],
#     # # ... Add metrics for other clients
# ]


mlp_fl_metrics5 = [
    [44.495, 54.54, 0.1936, 58.398, 0.8301],
    [47.321, 57.344, 0.2404, 61.753, 0.7839],
    # [40.957, 56.375, 0.1562, 56.375, 0.671],
    [48.718, 63.049, 0.2301, 64.923, 0.8222],
    # [54.37, 22.606, 0.2514, 181.255, 0.61],
    # [53.99, 23.109, 0.2802, 178.856, 0.61],
    # [55.69, 21.311, 0.3451, 173.906, 0.56],
    # [56.89, 25.477, 0.3839, 179.821, 0.55],
    [43.279, 18.096, 0.1187, 59.826, 0.8591],
    # [55.03, 17.259, 0.1115, 186.88, 0.59],
    # [55.51, 18.27, 0.1311, 183.141, 0.60],
    [41.08, 19.26, 0.1464, 57.644, 0.8773],
    # [56.02, 23.52, 0.2483, 183.138, 0.59],
    # [53.29, 24.247, 0.3021, 183.005, 0.58],
    # # ... Add metrics for other clients
]
# #
# # MPL-FL algorithm metrics for each client
# mpl_fl_metrics_euro = [
#     [55.23, 21.962, 0.1936, 105.662, 0.71],
#     [54.625, 23.733, 0.2404, 98.393, 0.69],
#     [53.98, 18.095, 0.1562, 97.874, 0.671],
#     [57.98, 20.502, 0.2301, 121.892, 0.725],
#     # [54.37, 22.606, 0.2514, 181.255, 0.61],
#     # [53.99, 23.109, 0.2802, 178.856, 0.61],
#     # [55.69, 21.311, 0.3451, 173.906, 0.56],
#     # [56.89, 25.477, 0.3839, 179.821, 0.55],
#     [37.139, 18.096, 0.1187, 55.202, 0.88],
#     # [55.03, 17.259, 0.1115, 186.88, 0.59],
#     # [55.51, 18.27, 0.1311, 183.141, 0.60],
#     [35.466, 19.26, 0.1464, 52.449, 0.90],
#     # [56.02, 23.52, 0.2483, 183.138, 0.59],
#     # [53.29, 24.247, 0.3021, 183.005, 0.58],
#     # ... Add metrics for other clients
# ]
#
# # MPL-FL algorithm metrics for each client
# mlp_fl_metrics_asia = [
#     [28.936, 21.962, 0.1936, 49.323, 0.88],
#     [30.746, 23.733, 0.2404, 51.475, 0.85],
#     [53.98, 18.095, 0.1562, 97.874, 0.671],
#     [57.98, 20.502, 0.2301, 121.892, 0.725],
#     # [54.37, 22.606, 0.2514, 181.255, 0.61],
#     # [53.99, 23.109, 0.2802, 178.856, 0.61],
#     # [55.69, 21.311, 0.3451, 173.906, 0.56],
#     # [56.89, 25.477, 0.3839, 179.821, 0.55],
#     [59.91, 18.096, 0.1187, 141.244, 0.76],
#     # [55.03, 17.259, 0.1115, 186.88, 0.59],
#     # [55.51, 18.27, 0.1311, 183.141, 0.60],
#     [62.47, 19.26, 0.1464, 142.694, 0.71],
#     # [56.02, 23.52, 0.2483, 183.138, 0.59],
#     # [53.29, 24.247, 0.3021, 183.005, 0.58],
#     # # ... Add metrics for other clients
# ]


# Set the width of the bars
bar_width = 0.1
padding = 0

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
    # plt.bar(r2[i], mlp_fl_metrics6[i][3], color=colors[0], hatch=patterns[0], width=bar_width, edgecolor='black',
    #         label='MLP-FL_6_clients' if i == 0 else "")
    # CNN and CNN-FL bars with the same pattern
    # if i!=2:

    plt.bar(r2[i], mlp_fl_metrics5[i][3], color=colors[0], hatch=patterns[1], width=bar_width, edgecolor='black',
                label='MLP-FL_5clients' if i == 0 else "")

    #
    # if i in [4,5]:
    #
    #     plt.bar(r4[i], mpl_fl_metrics_euro[i][3], color=colors[2], hatch=patterns[2], width=bar_width, edgecolor='black',
    #             label='MLP-FL_2_clients' if i == 0 else "")
    #
    # if i in [0,1]:
    #
    #     plt.bar(r5[i], mlp_fl_metrics_asia[i][3], color=colors[2], hatch=patterns[3], width=bar_width, edgecolor='black',
    #             label='MLP-FL_1_clients' if i == 0 else "")
    # plt.bar(r4[i], cnn_fl_metrics[i][0], color=colors[1], hatch=patterns[1], width=bar_width, edgecolor='black',
    #         label='CNN-FL' if i == 0 else "")
    # # TF and TF-FL bars with the same pattern
    # plt.bar(r5[i], tf_metrics[i][0], color='lightgrey', hatch=patterns[2], width=bar_width, edgecolor='black',
    #         label='TF' if i == 0 else "")
    # plt.bar(r6[i], tf_fl_metrics[i][0], color=colors[2], hatch=patterns[2], width=bar_width, edgecolor='black',
    #         label='TF-FL' if i == 0 else "")

# Add x-axis labels

plt.xlabel('Clients', fontsize=12)  # Update fontsize value as needed
plt.xticks([r + bar_width for r in range(len(clients))], clients, rotation=90, fontsize=10)  # Update fontsize value as needed

plt.ylabel('RMSE Metric Value', fontsize=12)  # Update fontsize value as needed

plt.title('Collaboration between Asia and Europe', fontsize=14)  # Update fontsize value as needed

plt.legend(fontsize=10)  # Update fontsize value as needed

plt.tight_layout()
plt.show()

