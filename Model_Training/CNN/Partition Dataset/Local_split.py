
import pandas as pd
import h5py

# Read the integrated_train_data.csv file
dataset = pd.read_csv('E:\Python\Dataguan\FL_supply_chain\\result_document\integrated_train_data.csv')

# Get unique market values
markets = dataset['Order Region'].unique()

with h5py.File('market_data.h5', 'w') as f:
    for market in markets:
        market_str = str(market)
        market_data = dataset[dataset['Order Region'] == market]
        # Store the data as a dataset
        f.create_dataset(market_str, data=market_data.to_numpy())
        # Store the column names as an attribute
        f[market_str].attrs['columns'] = market_data.columns.tolist()
