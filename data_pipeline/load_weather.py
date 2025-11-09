import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os


current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)


def process_data(input_path, output_path):
    try:
        # Load raw CSV data
        df = pd.read_csv(input_path)

        # Rename relevant columns
        df.rename(columns={'temperature': 'target', 'time': 'date'}, inplace=True)

        # Convert 'date' to datetime
        df['date'] = pd.to_datetime(df['date'])

        # Calculate number of days since the first record
        first_date = df['date'].min()
        df['day'] = (df['date'] - first_date).dt.days

        # Group by day and compute daily averages
        numeric_cols = df.select_dtypes(include=['number']).columns
        df = df.groupby('day').mean()

        # Drop columns with >15% zeros or NaNs
        zero_threshold = 0.15
        col_lengths = len(df)
        cols_to_drop = [
            col for col in df.columns 
            if (df[col] == 0).sum() / col_lengths > zero_threshold or 
               df[col].isna().sum() / col_lengths > zero_threshold
        ]
        df.drop(columns=cols_to_drop, inplace=True)

        # Drop rows with remaining NaNs
        df.dropna(inplace=True)

        # Normalize all numeric columns except 'target'
        columns_to_normalize = df.columns[~((df.columns == 'date') | (df.columns == 'target'))]
        scaler = MinMaxScaler()

        for col in columns_to_normalize:
            df[col] = df[col] - df[col].min()

        df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])

        # Re-sort and clean up
        df['date'] = pd.to_datetime(df['date'], errors='ignore')  # for safety if present
        df = df.sort_values(by='date')
        df.columns = [col.split('(')[0].strip() for col in df.columns]

        # Save the processed DataFrame
        df.to_csv(output_path, index=False)
        print(df)
        print(f"Processed data saved to {output_path}")

    except Exception as e:
        print(f"Error processing the file: {e}")


# Define input and output file paths
input_file = os.path.join(current_script_dir, '..', 'rawdata', 'weather.csv')
input_file = os.path.abspath(input_file)

output_file = os.path.join(current_script_dir, '..', 'data', 'weather.csv')
output_file = os.path.abspath(output_file)

# Run the processing function
process_data(input_file, output_file)