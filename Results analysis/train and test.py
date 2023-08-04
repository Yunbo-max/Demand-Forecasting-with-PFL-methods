# -*- coding = utf-8 -*-
# @time:14/07/2023 11:26
# Author:Yunbo Long
# @File:train and test.py
# @Software:PyCharm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Example data (replace with your actual data)
train_actual = np.array([3, 5, 7, 9, 11])
train_predicted = np.array([2.5, 4.7, 6.8, 8.9, 10.2])

test_actual = np.array([2, 4, 6, 8, 10])
test_predicted = np.array([2.2, 4.5, 5.8, 7.9, 10.5])

# Calculate metrics for train data
train_mae = mean_absolute_error(train_actual, train_predicted)
train_mape = np.mean(np.abs((train_actual - train_predicted) / train_actual)) * 100
train_mse = mean_squared_error(train_actual, train_predicted)
train_rmse = np.sqrt(train_mse)
train_r2 = r2_score(train_actual, train_predicted)

# Calculate metrics for test data
test_mae = mean_absolute_error(test_actual, test_predicted)
test_mape = np.mean(np.abs((test_actual - test_predicted) / test_actual)) * 100
test_mse = mean_squared_error(test_actual, test_predicted)
test_rmse = np.sqrt(test_mse)
test_r2 = r2_score(test_actual, test_predicted)

# Metrics labels
metrics = ['MAE', 'MAPE', 'MSE', 'RMSE', 'R-squared']

# Train and test metric values
train_metrics = [train_mae, train_mape, train_mse, train_rmse, train_r2]
test_metrics = [test_mae, test_mape, test_mse, test_rmse, test_r2]

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = np.arange(len(metrics))
r2 = [x + bar_width for x in r1]

# Plot the bars
plt.bar(r1, train_metrics, color='skyblue', width=bar_width, edgecolor='black', label='Train')
plt.bar(r2, test_metrics, color='lightgreen', width=bar_width, edgecolor='black', label='Test')

# Add x-axis labels
plt.xlabel('Metrics')
plt.xticks([r + bar_width/2 for r in range(len(metrics))], metrics)

# Add y-axis label
plt.ylabel('Metric Value')

# Add a title
plt.title('Comparison of Metrics for Train and Test Data')

# Add a legend
plt.legend()

# Show the plot
plt.show()
