import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import sys

# ==========================================
# 1. Model Definition (must exactly match training structure)
# ==========================================
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        
    def forward(self, x):
        return self.layers(x)

# ==========================================
# 2. Main Prediction Logic
# ==========================================
def main():
    # --- Command line argument parsing ---
    parser = argparse.ArgumentParser(description="Make predictions using fine-tuned MLP model")
    parser.add_argument('input_file', type=str, help="Input data file path (.tsv)")
    parser.add_argument('output_file', type=str, help="Output result file path (.tsv)")
    args = parser.parse_args()

    input_path = args.input_file
    output_path = args.output_file

    # --- Check file existence ---
    # Prioritize fine-tuned model
    model_path = 'final_model_tuned.pth'
    if not os.path.exists(model_path):
        print(f"Note: '{model_path}' not found, searching for 'final_model.pth'...")
        model_path = 'final_model.pth'
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        sys.exit(1)

    if not os.path.exists('scaler.pkl'):
        print("Error: Standardization file 'scaler.pkl' not found")
        sys.exit(1)
        
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ==========================================
    # Key modification 1: Set feature dimension to 72
    # ==========================================
    FEATURE_DIM = 72 

    # --- Load resources ---
    print(f"Loading model ({model_path}) and Scaler...")
    try:
        scaler = joblib.load('scaler.pkl')
        
        # Initialize model
        model = MLPRegressor(input_dim=FEATURE_DIM).to(device)
        
        # Load weights
        # Key modification 2: Add weights_only=True to suppress warnings
        # (Remove this parameter if PyTorch version is too old and throws error)
        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            # Compatible with older PyTorch versions
            state_dict = torch.load(model_path, map_location=device)
            
        model.load_state_dict(state_dict)
        model.eval() 
    except Exception as e:
        print(f"Failed to load model: {e}")
        print(f"Please check if the model file's input dimension is {FEATURE_DIM}.")
        sys.exit(1)

    # --- Read data ---
    print(f"Reading input file: {input_path}")
    try:
        # Assume no header, first two columns are IDs, followed by features
        # File structure for prediction: [ID1, ID2, Feat1, Feat2, ..., Feat72] (total 74 columns)
        df = pd.read_csv(input_path, sep='\t', header=0)
        
        # Extract IDs (first two columns)
        ids_df = df.iloc[:, 0:2]
        
        # ==========================================
        # Key modification 3: Data slicing logic
        # ==========================================
        # We need to read 72 feature columns
        # Starting column index: 2 (skip ID1, ID2)
        # Ending column index: 2 + 72 = 74
        start_col = 2
        end_col = start_col + FEATURE_DIM 
        
        # Check if there are enough columns
        if df.shape[1] < end_col:
            print(f"Error: Input data has insufficient columns! Model needs {FEATURE_DIM} features (at least {end_col} columns), but file only has {df.shape[1]} columns.")
            sys.exit(1)
            
        # Extract feature data (iloc uses left-closed right-open interval, i.e., [2, 74))
        X_raw = df.iloc[:, start_col:end_col].values
        
        print(f"Data loaded successfully: samples={X_raw.shape[0]}, features={X_raw.shape[1]}")
        
        if X_raw.shape[1] != FEATURE_DIM:
            print(f"Error: Extracted feature count ({X_raw.shape[1]}) doesn't match model definition ({FEATURE_DIM}).")
            sys.exit(1)

    except Exception as e:
        print(f"Error reading data: {e}")
        sys.exit(1)

    # --- Prediction ---
    print("Standardizing and making predictions...")
    try:
        # Standardization
        X_scaled = scaler.transform(X_raw.astype(np.float32))
        
        inputs = torch.from_numpy(X_scaled).to(device)
        
        with torch.no_grad():
            outputs = model(inputs)
            predictions = outputs.cpu().numpy()

    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Hint: Possibly the Scaler dimension doesn't match the current data feature count.")
        sys.exit(1)

    # --- Save results ---
    print(f"Saving results to: {output_path}")
    try:
        # Construct result DataFrame
        result_df = pd.DataFrame(ids_df.values, columns=['ID_1', 'ID_2'])
        result_df['Predicted_Score'] = predictions
        
        result_df.to_csv(output_path, sep='\t', index=False)
        print("Completed!")
        
    except Exception as e:
        print(f"Failed to save file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()