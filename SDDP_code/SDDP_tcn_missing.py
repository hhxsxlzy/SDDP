import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import argparse

current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='Dataset name without .csv extension')
parser.add_argument('--an', type=str, required=False, default='1', help='an parameter to distinct parallel')
args = parser.parse_args()
dataset_name = args.dataset
an = args.an


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNPRED(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCNPRED, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                        dilation=dilation, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNTimeSeriesModel(nn.Module):
    def __init__(self, input_channels, tcn_channels, output_size, kernel_size=2, dropout=0.2):
        super(TCNTimeSeriesModel, self).__init__()
        self.tcn = TCNPRED(num_inputs=input_channels, num_channels=tcn_channels,
                           kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(tcn_channels[-1], output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.linear(y)


def masked_mse_loss(reconstructed, target):
    mask = ~torch.isnan(target)
    if mask.sum() == 0:
        # Return a constant tensor that requires gradient, ensuring the loss is connected to the computation graph
        return torch.tensor(0.0, device=target.device, dtype=reconstructed.dtype, requires_grad=True) + 0.0 * torch.sum(reconstructed)
    loss = torch.sum((reconstructed[mask] - target[mask]) ** 2) / mask.sum()
    return loss


def use_tcn1(df, h, retrain):
    """
    Apply the TCN model to predict each variable (except 'target') in the input DataFrame.

    Parameters:
        df (pd.DataFrame): Input data containing the target column 'target' and other variables.
        h (int): Forecasting horizon.
        retrain (bool): Whether to retrain the model for each column.

    Returns:
        pd.DataFrame: DataFrame containing the prediction results along with the original 'target' column.
    """
    if retrain:
        # Construct folder path
        folder_path_del = os.path.join(current_script_dir, '..', 'checkpoints', f'SDDPtcnmissing{an}_{args.dataset}_temp')
        folder_path_del = os.path.abspath(folder_path_del)
        os.makedirs(folder_path_del, exist_ok=True)

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path_del):
            file_path = os.path.join(folder_path_del, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"All files in the folder '{folder_path_del}' have been deleted.")

    # Initialize DataFrame to store predictions
    target_series = df['target'].copy()
    df_filtered = df.drop(columns=['target'], inplace=False, errors='ignore')
    predictions_df = pd.DataFrame(index=df_filtered.index)

    # Iterate over each variable and make predictions
    for column in df_filtered.columns:
        print(f"\nTraining and predicting for variable: {column}")
        predictions_series = train_and_predict_tcn_for_column(
            h=h,
            data_column=df_filtered[column],
            time_index=df_filtered.index,
            retrain=retrain,
            column_name=column,
            target_column=df['target']
        )
        predictions_df[column] = predictions_series

    # Append the `target` column back to the prediction results
    predictions_df['target'] = target_series
    predictions_df['target_lag'] = target_series
    predictions_df = filter_and_clean_dataframe(predictions_df, target_column="target", nan_threshold=0.1)

    return predictions_df


def train_and_predict_tcn_for_column(h, data_column, time_index, retrain, column_name, target_column,
                                     num_channels=[16, 32], kernel_size=2, dropout=0.2,
                                     epochs=100, batch_size=32, learning_rate=0.001,
                                     early_stopping_patience=4):
    """
    Train or load a TCN model for a single variable and perform prediction for time T+h for the extraction task.

    Modified version: includes 8:2 train/validation split.
    """

    predictions = []
    column_name = str(column_name)
    data_column = data_column.values.astype(np.float32).reshape(-1, 1)

    # Define model saving path
    model_save_dir = os.path.join(current_script_dir, '..', 'checkpoints', f'SDDPtcnmissing{an}_{args.dataset}_temp')
    model_save_dir = os.path.abspath(model_save_dir)
    os.makedirs(model_save_dir, exist_ok=True)
    model_path = os.path.join(model_save_dir, f"SDDPtcnmissing_{column_name}.pth")

    # Define TCN model
    input_size = 1
    model = TCNTimeSeriesModel(input_channels=input_size, tcn_channels=num_channels,
                               output_size=1, kernel_size=kernel_size, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load or train the model
    if not retrain and os.path.exists(model_path):
        print(f"Loading pre-trained model for column: {column_name}")
        model.load_state_dict(torch.load(model_path))
        lenth_tr1 = round(min(len(data_column) * 0.05, 200))
    else:
        print(f"Training model for column: {column_name}")

        # Prepare training data
        input_data = []
        target = []
        lenth_tr1 = round(min(len(data_column) * 0.05, 200))
        for i in range(lenth_tr1 - 1, len(data_column) - h):
            window = data_column[i - lenth_tr1 + 1: i + 1]
            window_filled = np.where(np.isnan(window), 0, window)
            input_data.append(window_filled)
            target.append(data_column[i + h])
        input_data = np.array(input_data, dtype=np.float32).reshape(-1, lenth_tr1, 1)
        target = np.array(target, dtype=np.float32).reshape(-1, 1)

        # 8:2 split
        split_idx = int(len(input_data) * 0.8)
        train_X, val_X = input_data[:split_idx], input_data[split_idx:]
        train_Y, val_Y = target[:split_idx], target[split_idx:]

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_X), torch.tensor(train_Y))
        val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_X), torch.tensor(val_Y))

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        # Train the model
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for batch_data, batch_target in train_loader:
                optimizer.zero_grad()
                output = model(batch_data).squeeze()
                batch_target = batch_target.squeeze()
                loss = masked_mse_loss(output, batch_target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            # Validation loss
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_data, batch_target in val_loader:
                    output = model(batch_data).squeeze()
                    batch_target = batch_target.squeeze()
                    loss = masked_mse_loss(output, batch_target)
                    val_loss += loss.item()
                val_loss /= len(val_loader)

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {epoch_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state_dict = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}, best val_loss={best_val_loss:.4f}")
                    break

        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        # Save model parameters
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Prediction phase
    model.eval()
    base_data = data_column.ravel().copy()
    predictions = []

    for i in range(lenth_tr1 - 1, len(base_data) - 1):
        window = base_data[i - lenth_tr1 + 1: i + 1]
        window_filled = np.where(np.isnan(window), 0, window)
        input_window = window_filled.reshape(1, lenth_tr1, 1)
        input_tensor = torch.tensor(input_window, dtype=torch.float32)

        with torch.no_grad():
            prediction = model(input_tensor).item()
        if np.isnan(base_data[i + 1]):
            base_data[i + 1] = prediction
        predictions.append(prediction)

    aligned_index = time_index[lenth_tr1:]
    predictions_series = pd.Series(predictions, index=aligned_index)
    return predictions_series


def filter_and_clean_dataframe(df, target_column, nan_threshold=0.1):
    """
    Filter out columns in the DataFrame with NaN ratio exceeding the threshold,
    and drop all rows containing any NaN values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        target_column (str): Target column that should not be removed.
        nan_threshold (float): Threshold for NaN ratio (between 0 and 1).

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Calculate NaN ratio for each column
    column_nan_ratios = df.isna().mean()

    # Identify columns to remove (excluding the target column)
    columns_to_remove = column_nan_ratios[column_nan_ratios > nan_threshold].index.tolist()
    if target_column in columns_to_remove:
        columns_to_remove.remove(target_column)

    # Drop columns with high NaN ratio and drop all rows with any NaNs
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
    df_cleaned.dropna(inplace=True)
    return df_cleaned


def compute_loadings_tcn(df):
    """
    Perform PCA on the input DataFrame and retain common factors with eigenvalues > 1.
    The loading matrix B is computed using the first 60% of the data.

    Parameters:
        df (pd.DataFrame): Input data containing a 'target' column and index named 'date'.

    Returns:
        np.ndarray: Loading matrix B.
    """
    if 'date' not in df.index.names:
        raise ValueError("Input DataFrame must have an index named 'date'.")
    if 'target' not in df.columns:
        raise ValueError("Input DataFrame must contain a column named 'target'.")
    if 'target_lag' not in df.columns:
        raise ValueError("Input DataFrame must contain a column named 'target_lag'.")

    feature_df = df.drop(columns=['target', 'target_lag'])

    # Use the first 60% of the data for PCA
    split_index = int(len(feature_df) * 0.6)
    feature_df = feature_df.iloc[:split_index]

    # Handle missing values
    if feature_df.isnull().any().any():
        print("Warning: Missing values detected. Filling with column mean.")
        feature_df = feature_df.fillna(feature_df.mean())

    # Remove constant columns (e.g., zero variance)
    constant_columns = feature_df.columns[feature_df.nunique() <= 1]
    if not constant_columns.empty:
        print(f"Warning: Removing constant columns: {list(constant_columns)}")
        feature_df = feature_df.drop(columns=constant_columns)

    # Compute correlation matrix for PCA
    correlation_matrix = feature_df.corr()
    if np.isnan(correlation_matrix).any().any() or np.isinf(correlation_matrix).any().any():
        raise ValueError("Correlation matrix contains NaN or inf values after preprocessing.")

    # Count number of common factors (eigenvalues > 1)
    eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
    num_common_factors = np.sum(eigenvalues > 1)
    print(f"Number of common factors (eigenvalues > 1): {num_common_factors}")

    # Perform PCA and compute loading matrix B
    pca = PCA(n_components=min(num_common_factors, 7))
    pca.fit(feature_df)
    B = pca.components_.T @ np.diag(np.sqrt(pca.explained_variance_))
    return B


def apply_loadings_tcn(df, B):
    """
    Apply the loading matrix B to the input DataFrame for dimensionality reduction,
    and retain the "target" and "target_lag" columns.

    Parameters:
        df (pd.DataFrame): Input data with index named "date", and columns including "target".
        B (np.ndarray): Loading matrix B.

    Returns:
        pd.DataFrame: DataFrame containing the reduced matrix F and the "target" columns.
    """
    if 'date' not in df.index.names:
        raise ValueError("Input DataFrame must have an index named 'date'.")
    if 'target' not in df.columns:
        raise ValueError("Input DataFrame must contain a column named 'target'.")
    if 'target_lag' not in df.columns:
        raise ValueError("Input DataFrame must contain a column named 'target_lag'.")

    target_col = df['target']
    target_lag_col = df['target_lag']
    feature_df = df.drop(columns=['target', 'target_lag'])

    # Check for missing values and fill with column means
    if feature_df.isnull().any().any():
        print("Warning: Missing values detected. Filling with column mean.")
        feature_df = feature_df.fillna(feature_df.mean())

    # Remove constant columns (which have undefined correlation)
    constant_columns = feature_df.columns[feature_df.nunique() <= 1]
    if not constant_columns.empty:
        print(f"Warning: Removing constant columns: {list(constant_columns)}")
        feature_df = feature_df.drop(columns=constant_columns)

    # Compute the reduced matrix F
    F = feature_df @ B
    factor_df = F.copy()
    factor_df.columns = [f'pca_{i + 1}' for i in range(F.shape[1])]

    # Add back the "target" and "target_lag" columns
    factor_df['target'] = target_col
    factor_df['target_lag'] = target_lag_col
    return factor_df


def tcn_predict_tplush(df, h, B, num_channels=[16, 32], kernel_size=2, dropout=0.2,
                       epochs=100, batch_size=18, learning_rate=0.001, patience=4):
    """
    Forecast future target values using TCN based on target-aware predictors.

    This function applies dimensionality reduction via loading matrix B, constructs
    sliding windows, and trains a TCN model to predict the target variable at time T+h.

    Input:
        df (pd.DataFrame): Input DataFrame with index 'date' and columns including 'target'.
        h (int): Forecasting horizon (number of steps ahead to predict).
        B (np.ndarray): Loading matrix for dimensionality reduction.

    Returns:
        tuple: RMSFE and MAE on the validation set.
    """
    exclude_cols = ['date', 'target']
    target_col = 'target'
    predictors = [col for col in df.columns if col not in exclude_cols]

    # Construct dataset using sliding window
    start_idx = round(len(df) * 0.6)
    lenth_tr2 = round(min(len(df) * 0.05, 200))

    X_new, Y_new = [], []
    for i in range(start_idx + 1, len(df) - h - 1):
        df_window = df.iloc[:i + h + 1]
        df_pca = apply_loadings_tcn(df_window, B)

        pca_columns = [col for col in df_pca.columns if col != "target"]

        scaler = MinMaxScaler()
        df_pca[pca_columns] = scaler.fit_transform(df_pca[pca_columns])
        predictors = [col for col in df_pca.columns if col.startswith('pca_')] + ['target_lag']

        window = df_pca.iloc[i - lenth_tr2 + 2:i + 1]
        target = df_pca.iloc[i + h][target_col]
        X_new.append(window[predictors].values)
        Y_new.append(target)

    X_new = np.array(X_new)
    Y_new = np.array(Y_new).reshape(-1, 1)

    # Train/validation split (3/5 * 1/3 = 1/5 as X_new X * 3/5)
    split_idx = int(len(X_new) * 2 / 3)
    X_train, X_val = X_new[:split_idx], X_new[split_idx:]
    Y_train, Y_val = Y_new[:split_idx], Y_new[split_idx:]

    # Convert to Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize TCN model
    input_size = X_train.shape[2]
    model = TCNTimeSeriesModel(
        input_channels=input_size,
        tcn_channels=num_channels,
        output_size=1,
        kernel_size=kernel_size,
        dropout=dropout
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ----- Early Stopping -----
    patience = patience
    best_val_loss = float('inf')
    no_improve = 0
    best_state_dict = None

    # Train model with early stopping
    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze(-1)
            batch_Y = batch_Y.squeeze()
            loss = criterion(outputs, batch_Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / max(len(train_loader), 1)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X).squeeze(-1)
                batch_Y = batch_Y.squeeze()
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
        val_loss = val_loss / max(len(val_loader), 1)

        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # ---- Early Stopping Check ----
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            no_improve = 0
            best_state_dict = model.state_dict()
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}. "
                      f"Best Val Loss: {best_val_loss:.4f}")
                break

    # Use best paraeters
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Evaluate on validation set (using best model)
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            outputs = model(batch_X).squeeze(-1)
            predictions.extend(outputs.numpy())
            actuals.extend(batch_Y.numpy())

    predictions = np.array(predictions).ravel()
    actuals = np.array(actuals).ravel()
    rmsfe = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    print(f"RMSFE on validation set (T+{h}): {rmsfe:.4f}")
    print(f"MAE on validation set (T+{h}): {mae:.4f}")

    return rmsfe, mae


def main():
    # Load data
    data_path = os.path.join(current_script_dir, '..', 'data', f'{dataset_name}.csv')
    df_filtered = pd.read_csv(data_path)
    print(df_filtered)
    df_filtered.set_index('date', inplace=True)
    rmsfe_list = []
    mae_list = []
    h = 3  # Forecasting horizon
    k = 1  # Number of iterations

    for i in range(k):
        print(f"Iteration {i+1}:")

        # Apply TCN model for prediction
        predictions_df = use_tcn1(df_filtered, h, retrain=True)

        # Perform PCA for dimensionality reduction
        print("\nPerforming PCA for dimensionality reduction...")
        B = compute_loadings_tcn(predictions_df)

        # Run prediction and compute RMSFE and MAE
        rmsfe, mae = tcn_predict_tplush(predictions_df, h=h, B=B)
        rmsfe_list.append(rmsfe)
        mae_list.append(mae)
        print(f"Iteration {i+1} - RMSFE: {rmsfe}, MAE: {mae}")

    # Compute mean RMSFE and MAE across iterations
    mean_rmsfe = np.mean(rmsfe_list)
    mean_mae = np.mean(mae_list)

    print("\nFinal Results:")
    print(f"Mean RMSFE over the iterations: {mean_rmsfe}")
    print(f"Mean MAE over the iterations: {mean_mae}")

    # Save RMSFE and MAE results to Excel
    results_df = pd.DataFrame({
        "Iteration": range(1, k+1),
        "RMSFE": rmsfe_list,
        "MAE": mae_list
    })
    results_df.loc["Mean"] = ["Mean", mean_rmsfe, mean_mae]

    results_dir = os.path.join(current_script_dir, '..', 'results_missing', dataset_name)
    os.makedirs(results_dir, exist_ok=True)
    output_excel_path = os.path.join(results_dir, f'results_SDDPtcn_{an}_{args.dataset}.xlsx')
    results_df.to_excel(output_excel_path, index=False)
    print(f"Results saved to {output_excel_path}")


if __name__ == "__main__":
    main()