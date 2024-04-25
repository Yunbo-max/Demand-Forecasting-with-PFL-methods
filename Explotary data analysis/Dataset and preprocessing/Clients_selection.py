# -*- coding: utf-8 -*-
# @Author: Yunbo
# @Date:   2024-01-24 10:28:46
# @Last Modified by:   Yunbo
# @Last Modified time: 2024-04-24 16:31:52
import h5py
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
# Open the HDF5 file
file = h5py.File('E:\Federated_learning_flower\experiments\Presentation\market_data.h5', 'r')
from tabulate import tabulate
region_map = {
    0: 'Southeast Asia',
    1: 'South Asia',
    2: 'Oceania',
    3: 'Eastern Asia',
    4: 'West Asia',
    5: 'West of USA',
    6: 'US Center',
    7: 'West Africa',
    8: 'Central Africa',
    9: 'North Africa',
    10: 'Western Europe',
    11: 'Northern Europe',
    12: 'Central America',
    13: 'Caribbean',
    14: 'South America',
    15: 'East Africa',
    16: 'Southern Europe',
    17: 'East of USA',
    18: 'Canada',
    19: 'Southern Africa',
    20: 'Central Asia',
    21: 'Eastern Europe',
    22: 'South of USA'
}

sheet_name = [str(i) for i in range(23)]

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBRegressor

model_performance = {}

for region_index, region_name in region_map.items():
    # Select only the data for the current region
    region_data = file[str(region_index)][:]
    region_data = pd.DataFrame(region_data)

    # Read the column names from the attributes
    column_names = file[str(region_index)].attrs['columns']

    # Assign column names to the dataset
    region_data.columns = column_names

    # Drop unwanted columns
    region_data = region_data.drop(columns=['Region Index'])

    # Split into features and target
    X = region_data.drop(['Sales'], axis=1)
    y = region_data['Sales']

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a model (for example, XGBoost regressor)
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Make predictions and evaluate performance
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    # Store the performance
    model_performance[region_name] = (mse, len(region_data))

# Rank the regions by model performance
ranked_regions = sorted(model_performance.items(), key=lambda x: x[1][0])

# Create a table of the ranked regions
table_data = []
for region, (performance, num_rows) in ranked_regions:
    table_data.append([region, performance, num_rows])

# Print the table
headers = ["Region", "Performance", "Number of Rows"]
table = tabulate(table_data, headers=headers, tablefmt="grid")
print(table)

# Print the table with white background and black characters
table = tabulate(table_data, headers=headers, tablefmt="grid")
table_with_format = f"\033[107m\033[30m{table}\033[0m"
print(table_with_format)

import matplotlib.pyplot as plt

# Print the table with white background and black characters
table = tabulate(table_data, headers=headers, tablefmt="grid")
table_with_format = f"\033[107m\033[30m{table}\033[0m"

# Save the table as a PNG image
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
table_image = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
table_image.auto_set_font_size(False)
table_image.set_fontsize(12)
table_image.scale(1.2, 1.2)
table_image.auto_set_column_width([0, 1, 2])

plt.savefig('table.png', dpi=300, bbox_inches='tight')

continent_map = {
    0: 'Asia',
    1: 'Asia',
    2: 'Oceania',
    3: 'Asia',
    4: 'Asia',
    5: 'North America',
    6: 'North America',
    7: 'Africa',
    8: 'Africa',
    9: 'Africa',
    10: 'Europe',
    11: 'Europe',
    12: 'North America',
    13: 'North America',
    14: 'South America',
    15: 'Africa',
    16: 'Europe',
    17: 'North America',
    18: 'North America',
    19: 'Africa',
    20: 'Asia',
    21: 'Europe',
    22: 'North America'
}


# Create a table of the ranked regions
table_data = []
for region, (performance, num_rows) in ranked_regions:
    region_index = int(region.split()[0])
    continent = continent_map.get(region_index, 'Unknown')  # Get the continent from the continent_map, default to 'Unknown' if not found
    table_data.append([region, continent, performance, num_rows])

# Print the table with white background and black characters
table = tabulate(table_data, headers=["Region", "Continent", "Performance", "Number of Rows"], tablefmt="grid")
table_with_format = f"\033[107m\033[30m{table}\033[0m"
print(table_with_format)

# Save the table as a PNG image
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')
table_image = ax.table(cellText=table_data, colLabels=["Region", "Continent", "Performance", "Number of Rows"],
                       cellLoc='center', loc='center')
table_image.auto_set_font_size(False)
table_image.set_fontsize(12)
table_image.scale(1.2, 1.2)
table_image.auto_set_column_width([0, 1, 2, 3])

plt.savefig('table.png', dpi=300, bbox_inches='tight')
