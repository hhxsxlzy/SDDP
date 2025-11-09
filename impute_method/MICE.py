import argparse
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def run_mice_imputation(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Separate columns that should not be imputed
    non_impute_cols = ['date', 'target']
    impute_cols = [col for col in df.columns if col not in non_impute_cols]

    # Extract the numeric part for imputation
    impute_data = df[impute_cols].copy()

    # Fit the imputer using the first 80% of the data
    n = len(impute_data)
    split_idx = int(n * 0.8)
    imputer = IterativeImputer(random_state=0, max_iter=100)
    imputer.fit(impute_data.iloc[:split_idx])

    # Apply imputation to the entire dataset
    imputed_array = imputer.transform(impute_data)
    imputed_df = pd.DataFrame(imputed_array, columns=impute_cols, index=impute_data.index)

    # Reconstruct the full DataFrame (restore original date + target columns)
    final_df = df[non_impute_cols].copy()
    for col in imputed_df.columns:
        final_df[col] = imputed_df[col]

    # Save the imputed result
    output_file = csv_file.replace('.csv', '_MICE.csv')
    final_df.to_csv(output_file, index=False)
    print(f"Imputed full dataset saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to input CSV dataset')
    args = parser.parse_args()

    run_mice_imputation(args.dataset)
