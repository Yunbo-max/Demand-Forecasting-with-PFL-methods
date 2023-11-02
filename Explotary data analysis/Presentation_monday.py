import pandas as pd
import networkx as nx
from pyvis.network import Network
import webbrowser
import os

# Create a sample dataframe
df = pd.read_csv("/Users/yunbo-max/Desktop/Personalised_FL/DataCoSupplyChainDataset.csv", header=0, encoding='unicode_escape')

# Initialize a DiGraph
G = nx.DiGraph()

# Create nodes for Customer City and Order City
for city in set(df['Customer City']).union(set(df['Order City'])):
    G.add_node(city)

# Create a color mapping for unique "Order Region" values
unique_order_regions = set(df['Order Region'])
color_mapping = {region: f'#{region}' for region in unique_order_regions}

# Create edges and assign colors based on "Order Region"
for _, row in df.iterrows():
    customer_city = row['Customer City']
    order_city = row['Order City']
    order_region = row['Order Region']

    if G.has_edge(customer_city, order_city):
        G[customer_city][order_city]['color'] = color_mapping[order_region]
    else:
        G.add_edge(customer_city, order_city, color=color_mapping[order_region])

# Initialize the pyvis Network
nt = Network(height='500px', width='800px', directed=True)

# Create edges and assign colors based on "Order Region"
for _, row in df.iterrows():
    customer_city = row['Customer City']
    order_city = row['Order City']
    order_region = row['Order Region']

    if G.has_edge(customer_city, order_city):
        G[customer_city][order_city]['color'] = color_mapping[order_region]
    else:
        G.add_edge(customer_city, order_city, color=color_mapping[order_region])

    # Add nodes with different colors based on "Order Region"
    nt.add_node(customer_city, color=f'#{customer_city}')
    nt.add_node(order_city, color=f'#{order_city}')

# Modify edge colors and set the width to make them thinner
for edge in G.edges:
    nt.add_edge(edge[0], edge[1], color=f'#{edge[0]}-{edge[1]}', width=0.00001)  # Adjust the width value as needed

# Save the HTML file
html_filename = 'customer_order_graph_thin.html'
nt.save_graph(html_filename)

# Display the HTML file in your default web browser
print(f'Saved HTML file to: {os.path.abspath(html_filename)}')
webbrowser.open(html_filename)

