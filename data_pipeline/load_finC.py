import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)


def process_data(input_path, output_path):
    # Load raw data
    df = pd.read_csv(input_path)

    # Ensure 'date' is datetime; coerce errors to NaT
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    # Rename target column
    df.rename(columns={'Appliances': 'target'}, inplace=True)

    print(df)

    # Define zero-value threshold and get numeric columns
    zero_threshold = 0.15
    col_lengths = len(df)
    numeric_cols = df.select_dtypes(include=['number']).columns

    # Drop predefined unnecessary column
    df.drop(columns='light', inplace=True, errors='ignore')

    # Drop rows with any missing values
    df.dropna(inplace=True)

    # Set datetime index for resampling
    df = df.set_index('date')

    # Resample to 30-minute intervals and average
    df = df.resample('30min').mean().reset_index()

    # Normalize all numeric columns except 'date' and 'target'
    columns_to_normalize = df.columns[~(df.columns == 'date') & ~(df.columns == 'target')]
    scaler = MinMaxScaler()

    # Subtract column-wise minimum
    for col in columns_to_normalize:
        df[col] = df[col] - df[col].min()

    # Use max from first 60% of the data for normalization
    split_index = int(0.6 * len(df))
    max_values = df.iloc[:split_index][columns_to_normalize].max()

    for col in columns_to_normalize:
        df[col] = df[col] / max_values[col]

    # Sort by date before saving
    df = df.sort_values(by='date')

    # Save processed data
    df.to_csv(output_path, index=False)
    print(df)
    print(f"Processed data saved to {output_path}")


# Set input and output paths
input_file = os.path.join(current_script_dir, '..', 'rawdata', 'finc.csv')
input_file = os.path.abspath(input_file)

output_file = os.path.join(current_script_dir, '..', 'data', 'finc.csv')
output_file = os.path.abspath(output_file)

# Run data processing
process_data(input_file, output_file)
