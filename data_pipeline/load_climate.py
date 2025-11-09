import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime


current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)


def process_data(input_path, output_path):
    try:
        # Load data
        df = pd.read_csv(input_path)

        # Rename key columns
        df.rename(columns={'T (degC)': 'target', 'Date Time': 'date'}, inplace=True)

        # Convert 'date' to datetime format
        timestamp_format = '%d.%m.%Y %H:%M:%S'
        df['date'] = pd.to_datetime(df['date'], format=timestamp_format)

        # Compute number of days since the first date
        first_date = df['date'].min()
        df['day'] = (df['date'] - first_date).dt.days

        # Group by day and take mean
        df = df.groupby('day').mean()

        # Drop columns with >15% zero values
        zero_threshold = 0.15
        col_lengths = len(df)
        cols_to_drop = [col for col in df.columns if (df[col] == 0).sum() / col_lengths > zero_threshold]

        # Drop columns with >15% NaN values
        cols_to_drop += [col for col in df.columns if df[col].isna().sum() / col_lengths > zero_threshold]
        df.drop(columns=cols_to_drop, inplace=True)

        # Drop rows with any remaining NaNs
        df.dropna(inplace=True)

        # Normalize columns (excluding 'date' and 'target') using MinMax based on the first 60%
        columns_to_normalize = df.columns[~(df.columns == 'date') & ~(df.columns == 'target')]
        scaler = MinMaxScaler()

        for col in columns_to_normalize:
            df[col] = df[col] - df[col].min()

        split_index = int(0.6 * len(df))
        max_values = df.iloc[:split_index][columns_to_normalize].max()
        for col in columns_to_normalize:
            df[col] = df[col] / max_values[col]

        # Ensure 'date' is datetime and sort
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(by='date')
        df.columns = [col.split('(')[0].strip() for col in df.columns]

        # Save processed data
        df.to_csv(output_path, index=False)
        print(df)
        print(f"Processed data saved to {output_path}")

    except Exception as e:
        print(f"Error processing the file: {e}")


# Set input and output paths
input_file = os.path.join(current_script_dir, '..', 'rawdata', 'energydata_complete.csv')
input_file = os.path.abspath(input_file)
output_file = os.path.join(current_script_dir, '..', 'data', 'energy.csv')
output_file = os.path.abspath(output_file)

# Run data processing
process_data(input_file, output_file)
