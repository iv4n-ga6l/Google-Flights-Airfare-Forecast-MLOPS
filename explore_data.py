import pandas as pd

# Load and explore the data
df = pd.read_csv('google_flights_airfare_data.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nUnique values:")
print(f"Airlines: {df['airline'].nunique()}")
print(f"Origins: {df['origin'].nunique()}")
print(f"Destinations: {df['destination'].nunique()}")
print(f"Fare classes: {df['fare_class'].unique()}")
print(f"\nPrice statistics:")
print(df['price'].describe())
