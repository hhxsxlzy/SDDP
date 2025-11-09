import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer

def run_missforest_imputation(csv_file):
    # Load the dataset
    df = pd.read_csv(csv_file)

    # Separate columns that should not be imputed
    non_impute_cols = ['date', 'target']
    impute_cols = [col for col in df.columns if col not in non_impute_cols]

    # Extract the part to be imputed
    impute_data = df[impute_cols].copy()
    
    # Identify numeric and categorical columns
    num_cols = impute_data.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = impute_data.select_dtypes(exclude=[np.number]).columns.tolist()

    # Initialize imputed DataFrame with mean/mode imputation
    initial_imputer = SimpleImputer(strategy='mean')
    imputed_array = initial_imputer.fit_transform(impute_data[num_cols])
    imputed_df = pd.DataFrame(imputed_array, columns=num_cols, index=impute_data.index)
    
    # Iteratively refine imputation using Random Forest (MissForest logic)
    max_iter = 10
    for _ in range(max_iter):
        for col in num_cols:
            missing_mask = impute_data[col].isnull()
            if missing_mask.sum() == 0:
                continue
            train_X = imputed_df.loc[~missing_mask].drop(columns=col)
            train_y = imputed_df.loc[~missing_mask, col]
            test_X = imputed_df.loc[missing_mask].drop(columns=col)
            model = RandomForestRegressor(n_estimators=100, random_state=0)
            model.fit(train_X, train_y)
            imputed_df.loc[missing_mask, col] = model.predict(test_X)

    # Reconstruct the full DataFrame (restore original date + target columns)
    final_df = df[non_impute_cols].copy()
    for col in imputed_df.columns:
        final_df[col] = imputed_df[col]

    # Save the imputed result
    output_file = csv_file.replace('.csv', '_MissForest.csv')
    final_df.to_csv(output_file, index=False)
    print(f"[✓] Imputed full dataset saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to input CSV dataset')
    args = parser.parse_args()

    run_missforest_imputation(args.dataset)
