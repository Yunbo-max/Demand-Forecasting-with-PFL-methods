import numpy as np
import matplotlib.pyplot as plt

# Region names
region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    5: 'West of USA',
    6: 'US Center',
    7: 'West Africa',
    9: 'North Africa',
    10: 'Western Europe',
    12: 'Central America',
    14: 'South America',
    16: 'Southern Europe',
    17: 'East of USA',
    22: 'South of USA'
}

# Example data (replace with your actual data)
clients = list(region_map.values())
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R-squared']

# MLP algorithm metrics for each client
mlp_metrics = [
    [0.08821, 3.5, 0.0272, 0.1649, 0.986],
    [0.3156, 12.082, 0.2579, 0.5078, 0.8706],
    [0.04114, 1.701, 0.006045, 0.07775, 0.997],
    [0.04151, 1.435, 0.0088234, 0.09074, 0.9958],
    [0.03669, 1.35, 0.00513, 0.07162, 0.9976],
    [0.05502, 1.704, 0.008572, 0.09259, 0.9959],
    [0.07587, 3.05, 0.01958, 0.1399, 0.9912],
    [0.06039, 2.105, 0.01402, 0.1184, 0.9938],
    [0.01201, 0.6459, 0.001202, 0.03467, 0.9994],
    [0.0145, 0.6457, 0.0027, 0.05197, 0.9987],
    [0.02243, 0.8766, 0.002168, 0.04656, 0.999],
    [0.03016, 1.012, 0.004945, 0.07032, 0.9975],
    [0.02428, 0.9644, 0.00235, 0.04848, 0.9989],
    [0.03083, 1.315, 0.004101, 0.06404, 0.998],
    # ... Add metrics for other clients
]

# CNN algorithm metrics for each client
cnn_metrics = [
    [0.16, 6.387, 0.094414, 0.3068, 0.9514],
    [0.1216, 4.707, 0.05722, 0.2392, 0.9713],
    [0.1049, 4.511, 0.03838, 0.1959, 0.9809],
    [0.1545, 7.405, 0.07089, 0.2663, 0.9641],
    [0.1926, 9.188, 0.09265, 0.3044, 0.9565],
    [0.1801, 7.908, 0.08906, 0.2984, 0.9579],
    [0.1889, 8.858, 0.08831, 0.2972, 0.9604],
    [0.2171, 9.059, 0.1173, 0.3426, 0.9478],
    [0.117, 5.933, 0.03257, 0.1805, 0.9842],
    [0.06622, 2.557, 0.01554, 0.1247, 0.9927],
    [0.09947, 4.072, 0.03055, 0.1748, 0.9855],
    [0.1645, 8.484, 0.07461, 0.2731, 0.9626],
    [0.1092, 4.361, 0.04221, 0.2054, 0.9799],
    [0.1383, 5.917, 0.05722, 0.2397, 0.972],
    # ... Add metrics for other clients
]

# TF algorithm metrics for each client
tf_metrics = [
    [0.3324, 21.962, 0.1936, 0.44, 0.9001],
    [0.3679, 23.733, 0.2404, 0.4903, 0.8797],
    [0.2833, 18.095, 0.1562, 0.3952, 0.9223],
    [0.3453, 20.502, 0.2301, 0.4797, 0.8836],
    [0.3739, 22.606, 0.2514, 0.5014, 0.8821],
    [0.3924, 23.109, 0.2802, 0.5293, 0.8675],
    [0.4113, 21.311, 0.3451, 0.5874, 0.846],
    [0.4703, 25.477, 0.3839, 0.6196, 0.8292],
    [0.2667, 18.096, 0.1187, 0.3445, 0.9425],
    [0.2579, 17.259, 0.1115, 0.3339, 0.9477],
    [0.2762, 18.27, 0.1311, 0.3621, 0.938],
    [0.2884, 19.26, 0.1464, 0.3827, 0.9265],
    [0.3763, 23.52, 0.2483, 0.4983, 0.881],
    [0.4064, 24.247, 0.3021, 0.5497, 0.8535],
    # ... Add metrics for other clients
]

# Set the width of the bars
bar_width = 0.12
padding = 0.04

# Set the positions of the bars on the x-axis
# Set the positions of the bars on the x-axis
# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
padding = 0

r1 = np.arange(len(clients))
r2 = [x + bar_width + padding for x in r1]
r3 = [x + bar_width + padding for x in r2]

# Plotting the bars
# Note: you can iterate through your metrics if you want all of them plotted.
# Here we're only plotting the first metric (i.e., MAE)

for i, client in enumerate(clients):
    plt.bar(r1[i], mlp_metrics[i][0], color='skyblue', width=bar_width, edgecolor='black', label='MLP' if i == 0 else "")
    plt.bar(r2[i], cnn_metrics[i][0], color='lightgreen', width=bar_width, edgecolor='black', label='CNN' if i == 0 else "")
    plt.bar(r3[i], tf_metrics[i][0], color='orange', width=bar_width, edgecolor='black', label='TF' if i == 0 else "")

plt.xlabel('Clients')
plt.xticks([r + bar_width for r in range(len(clients))], clients, rotation=90)

plt.ylabel('Metric Value (MAE)')  # update this label for each metric

plt.title('Local Learning Comparison')  # update this title for each metric

plt.legend()

plt.tight_layout()
plt.show()
