import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class CustomMissForest:
    """
    A simple MissForest-style imputer with an explicit fit/transform interface.
    - fit(X_train): learn RF models for each column with missing values.
    - transform(X): use learned models to impute missing values in new data.
    """

    def __init__(self, max_iter=5, tol=1e-3, random_state=0, verbose=True):
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.is_fitted = False

    def _initial_impute(self, X):
        """Initial mean imputation for numeric columns."""
        X_filled = X.copy()
        for col in X_filled.columns:
            mean_val = X_filled[col].mean()
            if np.isnan(mean_val):
                mean_val = 0.0
            # Avoid chained assignment warnings
            X_filled[col] = X_filled[col].fillna(mean_val)
        return X_filled

    def fit(self, X):
        """
        Fit MissForest-style models using only the training subset.
        X: pandas DataFrame (numeric), training portion.
        """
        X = X.copy()
        self.columns_ = list(X.columns)

        # Missing-value mask in the training data
        missing_mask = X.isnull()

        # Column update order: from least missing to most missing
        col_order = missing_mask.sum().sort_values().index

        # Initial mean imputation
        X_imp = self._initial_impute(X)

        # Store column means for later use in transform
        self.col_means_ = {col: X_imp[col].mean() for col in self.columns_}

        # Dictionary to store fitted RF models for each column
        self.models_ = {col: None for col in self.columns_}

        for it in range(self.max_iter):
            if self.verbose:
                print(f"\n[MissForest Fit] Iteration {it + 1}")
            X_old = X_imp.copy()

            for col in col_order:
                col_missing = missing_mask[col]
                if not col_missing.any():
                    continue

                not_missing = ~col_missing

                # Use all other columns as predictors
                feature_cols = [c for c in self.columns_ if c != col]

                X_train = X_imp.loc[not_missing, feature_cols]
                y_train = X_imp.loc[not_missing, col]

                if X_train.shape[0] == 0:
                    continue

                rf = RandomForestRegressor(
                    n_estimators=100,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                rf.fit(X_train, y_train)
                self.models_[col] = rf

                # Update missing entries in training data for this column
                X_pred = X_imp.loc[col_missing, feature_cols]
                if X_pred.shape[0] > 0:
                    y_pred = rf.predict(X_pred)
                    X_imp.loc[col_missing, col] = y_pred

            # Convergence check on imputed entries (only where there were NaNs)
            diff = 0.0
            count = 0
            for col in self.columns_:
                col_missing = missing_mask[col]
                if not col_missing.any():
                    continue
                old_vals = X_old.loc[col_missing, col]
                new_vals = X_imp.loc[col_missing, col]
                diff += ((old_vals - new_vals) ** 2).sum()
                count += len(old_vals)

            rmse = np.sqrt(diff / count) if count > 0 else 0.0
            if self.verbose:
                print(f"[MissForest Fit] Iter RMSE on imputed entries: {rmse:.6f}")

            if rmse < self.tol:
                if self.verbose:
                    print(f"[MissForest Fit] Converged at iteration {it + 1}.")
                break

        self.is_fitted = True
        return self

    def transform(self, X):
        """
        Impute missing values in a new dataset using models learned in fit().
        X: pandas DataFrame with the same numeric columns as in training.
        """
        if not self.is_fitted:
            raise RuntimeError("CustomMissForest must be fitted before calling transform().")

        X = X.copy()

        # Ensure all training columns exist
        missing_cols = [c for c in self.columns_ if c not in X.columns]
        if missing_cols:
            raise ValueError(f"Input data is missing columns used in training: {missing_cols}")

        # Align column order
        X = X[self.columns_]

        # Missing mask in the new data
        missing_mask = X.isnull()

        # Initial fill using training means
        X_imp = X.copy()
        for col in self.columns_:
            mean_val = self.col_means_.get(col, 0.0)
            X_imp[col] = X_imp[col].fillna(mean_val)

        col_order = missing_mask.sum().sort_values().index

        for it in range(self.max_iter):
            if self.verbose:
                print(f"\n[MissForest Transform] Iteration {it + 1}")
            X_old = X_imp.copy()

            for col in col_order:
                col_missing = missing_mask[col]
                if not col_missing.any():
                    continue

                model = self.models_.get(col, None)
                if model is None:
                    # No model fitted for this column (no missing in train),
                    # keep mean-imputed values.
                    continue

                feature_cols = [c for c in self.columns_ if c != col]
                X_pred = X_imp.loc[col_missing, feature_cols]
                if X_pred.shape[0] == 0:
                    continue

                y_pred = model.predict(X_pred)
                X_imp.loc[col_missing, col] = y_pred

            # Convergence check on entries that were missing in this dataset
            diff = 0.0
            count = 0
            for col in self.columns_:
                col_missing = missing_mask[col]
                if not col_missing.any():
                    continue
                old_vals = X_old.loc[col_missing, col]
                new_vals = X_imp.loc[col_missing, col]
                diff += ((old_vals - new_vals) ** 2).sum()
                count += len(old_vals)

            rmse = np.sqrt(diff / count) if count > 0 else 0.0
            if self.verbose:
                print(f"[MissForest Transform] Iter RMSE on imputed entries: {rmse:.6f}")

            if rmse < self.tol:
                if self.verbose:
                    print(f"[MissForest Transform] Converged at iteration {it + 1}.")
                break

        return X_imp


def run_missforest_imputation(csv_file):
    # Load full dataset
    df = pd.read_csv(csv_file)

    # Columns that should not be imputed
    non_impute_cols = ['date', 'target']
    impute_cols = [col for col in df.columns if col not in non_impute_cols]

    # Use only numeric columns for MissForest
    impute_data = df[impute_cols].select_dtypes(include=[np.number]).copy()

    # 80% / 20% split by row index for fitting vs. evaluation
    n = len(impute_data)
    split_idx = int(n * 0.8)

    train_data = impute_data.iloc[:split_idx]
    full_data = impute_data  # we will impute on the full dataset

    print(f"[Info] Total rows: {n}, using first {split_idx} rows (~80%) to fit MissForest.")

    # Fit on the first 80% of the data
    imputer = CustomMissForest(max_iter=3, tol=1e-3, random_state=0, verbose=True)
    imputer.fit(train_data)

    # Apply the learned imputer to the full dataset
    full_imputed = imputer.transform(full_data)

    # Reconstruct final DataFrame: keep original date/target, replace imputed numeric columns
    result_df = df[non_impute_cols].copy()
    for col in full_imputed.columns:
        result_df[col] = full_imputed[col]

    # Save result
    output_file = csv_file.replace('.csv', '_MISSF.csv')
    result_df.to_csv(output_file, index=False)
    print(f"[✓] MissForest-imputed data saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        help='Path to input CSV dataset')
    args = parser.parse_args()

    run_missforest_imputation(args.dataset)
