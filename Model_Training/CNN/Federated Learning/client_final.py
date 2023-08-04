import h5py
import pandas as pd
import wandb

# Open the HDF5 file
file = h5py.File('E:\Federated_learning_flower\experiments\Presentation\market_data.h5', 'r')

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

# Specify the region names you want to keep
sheet_name = ['0', '1', '2', '3', '5', '6', '7', '9', '10', '12', '14', '16', '17', '22']

specified_regions = [region_map[int(name)] for name in sheet_name]

# Read the dataset
dataset = pd.read_csv('E:\Federated_learning_flower\experiments\Presentation\integrated_train_data_ISMM.csv')

specified_regions = [region_map[int(name)] for name in sheet_name]
# Filter rows based on specified region names
filtered_dataset = dataset[dataset['index'].isin(specified_regions)]

print(filtered_dataset)
# Initialize Weights and Biases with project name

