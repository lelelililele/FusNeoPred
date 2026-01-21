import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import copy
import os
import joblib

# ==========================================
# Configuration and Logging
# ==========================================
LOG_FILE = 'training_log.txt'

def log_to_file(message, mode='a'):
    """Write message to log file."""
    with open(LOG_FILE, mode) as f:
        f.write(message + '\n')

# Initialize log file (overwrite old file)
header = "Phase\tEpoch\tTrain_Loss\tVal_Loss\tTest_Loss\tInfo"
log_to_file(header, mode='w')
print(f"Log file initialized: {LOG_FILE}")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Data Loading and Preprocessing
# ==========================================
def load_data(file_path, has_header=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    header_param = 0 if has_header else None
    df = pd.read_csv(file_path, sep='\t', header=header_param) 
    
    if df.shape[1] < 75:
        print(f"[Warning] File {file_path} has {df.shape[1]} columns, expected at least 75.")
    
    # Explicit slicing: indices 2-73 are features (72 columns), 74 is Score
    X = df.iloc[:, 2:74].values.astype(np.float32)
    y = df.iloc[:, 74].values.astype(np.float32).reshape(-1, 1)
    return X, y

# --- Read Main File (Pre-training) ---
main_file = 'S5_combined_complete_score.tsv'
print(f"Reading main file: {main_file} (with header) ...")

try:
    X_raw, y_raw = load_data(main_file, has_header=True)
    print(f"Data loaded successfully | Shape: {X_raw.shape}")
    
    if X_raw.shape[1] != 72:
        raise ValueError(f"Feature dimension error: {X_raw.shape[1]}")
        
except Exception as e:
    print(f"Critical error reading data: {e}")
    exit()

# Split dataset
X_train_raw, X_temp, y_train, y_temp = train_test_split(X_raw, y_raw, test_size=0.30, random_state=42)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw) 

# DataLoader
BATCH_SIZE = 64
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# ==========================================
# 2. Model Definition
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

model = MLPRegressor(input_dim=72).to(device)
criterion_mse = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# ==========================================
# 3. Pre-training Function
# ==========================================
def evaluate_mse(model, loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            total_loss += criterion_mse(outputs, targets).item() * inputs.size(0)
    return total_loss / len(loader.dataset)

def train_pretraining(model, epochs=100, patience=10):
    history = {'train_loss': [], 'val_loss': [], 'test_loss': []}
    best_val_loss = float('inf')
    best_wts = copy.deepcopy(model.state_dict())
    no_improve = 0
    
    print(f"\n=== Phase 1: Pre-training (MSE) ===")
    print(f"{'Epoch':^5} | {'Train MSE':^10} | {'Val MSE':^10} | {'Test MSE':^10} | {'Info'}")
    print("-" * 55)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion_mse(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        avg_train = train_loss / len(train_loader.dataset)
        avg_val = evaluate_mse(model, val_loader)
        avg_test = evaluate_mse(model, test_loader)
        
        history['train_loss'].append(avg_train)
        history['val_loss'].append(avg_val)
        history['test_loss'].append(avg_test)
        
        # Checkpoint
        status = ""
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_wts = copy.deepcopy(model.state_dict())
            no_improve = 0
            status = "*"
        else:
            no_improve += 1
            
        # Logging
        log_line = f"PreTrain\t{epoch+1}\t{avg_train:.6f}\t{avg_val:.6f}\t{avg_test:.6f}\t{status}"
        log_to_file(log_line)
        
        if (epoch+1) % 5 == 0 or status == "*":
            print(f"{epoch+1:^5} | {avg_train:^10.4f} | {avg_val:^10.4f} | {avg_test:^10.4f} | {status}")
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
    model.load_state_dict(best_wts)
    return model, history

model, history_pre = train_pretraining(model)

# Plot 1: Pre-training
def plot_pretrain(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train MSE', color='blue')
    plt.plot(history['val_loss'], label='Val MSE', color='red', linestyle='--')
    plt.plot(history['test_loss'], label='Test MSE', color='green', linestyle=':')
    plt.title('Phase 1: Pre-training Loss (MSE)')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('pretrain_curve.pdf', format='pdf')
    print("Pre-training curve saved: pretrain_curve.pdf")

plot_pretrain(history_pre)

# ==========================================
# 4. Fine-tuning Phase (Full-Batch Pairwise + Increased Epochs)
# ==========================================
fine_tune_file = 'Peplist_Score'

if os.path.exists(fine_tune_file):
    print(f"\n=== Phase 2: Fine-tuning (Ranking) ===")
    log_to_file("\n=== Phase 2: Fine-tuning ===")
    
    try:
        # Load data (No Header)
        X_ft, y_ft = load_data(fine_tune_file, has_header=False)
        
        if X_ft.shape[1] != 72:
            print("Fine-tuning feature dimension error, skipping.")
        else:
            X_ft_scaled = scaler.transform(X_ft)
            
            # Separate positive and negative sample indices
            pos_indices_raw = np.where(y_ft > 0.5)[0]
            neg_indices_raw = np.where(y_ft <= 0.5)[0]
            
            if len(pos_indices_raw) > 0 and len(neg_indices_raw) > 0:
                print(f"Samples: {len(pos_indices_raw)} Positive, {len(neg_indices_raw)} Negative")
                
                # Convert to Tensor
                X_pos = torch.from_numpy(X_ft_scaled[pos_indices_raw]).to(device)
                X_neg = torch.from_numpy(X_ft_scaled[neg_indices_raw]).to(device)
                
                # --- [Core Modification 1]: Construct Full Pairwise Indices (Full Pair Construction) ---
                # Only 3 positive x 2 negative = 6 pairs. Instead of random sampling, we generate all combinations.
                # This ensures gradients are based on all data every epoch, smoothing the loss curve.
                pos_idx_list = []
                neg_idx_list = []
                for p in range(len(X_pos)):
                    for n in range(len(X_neg)):
                        pos_idx_list.append(p)
                        neg_idx_list.append(n)
                
                # Fixed Index Tensors
                batch_p_fixed = X_pos[torch.tensor(pos_idx_list).to(device)]
                batch_n_fixed = X_neg[torch.tensor(neg_idx_list).to(device)]
                
                print(f"Fine-tuning mode: Full Batch | Total {len(pos_idx_list)} positive-negative pairs")

                # Config
                rank_criterion = nn.MarginRankingLoss(margin=0.1)
                
                # --- [Core Modification 2]: Hyperparameter Tuning ---
                # Since we use full batch, gradients are stable, so we can increase epochs.
                FT_LR = 2e-5         # Slightly increased learning rate (originally 1e-5)
                FT_EPOCHS = 20       # Increased epochs to allow Ranking Loss to descend (originally 5)
                
                ft_optimizer = optim.SGD(model.parameters(), lr=FT_LR, momentum=0.9)
                
                history_ft = {
                    'ft_rank_loss': [], 
                    'orig_train_mse': [],
                    'orig_val_mse': [], 
                    'orig_test_mse': []
                }
                
                print(f"{'Epoch':^5} | {'FT RankLoss':^12} | {'Orig Val MSE':^12} | {'Orig Test MSE':^12}")
                print("-" * 60)
                
                model.train() 
                
                for epoch in range(FT_EPOCHS):
                    ft_optimizer.zero_grad()
                    
                    # --- Directly use all fixed pairs ---
                    score_pos = model(batch_p_fixed)
                    score_neg = model(batch_n_fixed)
                    
                    target = torch.ones_like(score_pos).to(device)
                    
                    loss_rank = rank_criterion(score_pos, score_neg, target)
                    loss_rank.backward()
                    ft_optimizer.step()
                    
                    # --- Evaluation ---
                    mse_train = evaluate_mse(model, train_loader)
                    mse_val = evaluate_mse(model, val_loader)
                    mse_test = evaluate_mse(model, test_loader)
                    
                    history_ft['ft_rank_loss'].append(loss_rank.item())
                    history_ft['orig_train_mse'].append(mse_train)
                    history_ft['orig_val_mse'].append(mse_val)
                    history_ft['orig_test_mse'].append(mse_test)
                    
                    print(f"{epoch+1:^5} | {loss_rank.item():^12.6f} | {mse_val:^12.6f} | {mse_test:^12.6f}")
                    log_line = f"FineTune\t{epoch+1}\t{mse_train:.6f}\t{mse_val:.6f}\t{mse_test:.6f}\tRankLoss:{loss_rank.item():.6f}"
                    log_to_file(log_line)

                # Save model
                torch.save(model.state_dict(), 'final_model_tuned.pth')
                joblib.dump(scaler, 'scaler.pkl')
                print("\nModel saved: final_model_tuned.pth")
                
                # --- Plotting (Dual-axis) ---
                def plot_finetune_dynamics(history):
                    fig, ax1 = plt.subplots(figsize=(10, 6))
                    
                    # Left Axis: Ranking Loss (Red)
                    color = 'tab:red'
                    ax1.set_xlabel('Epochs')
                    ax1.set_ylabel('Ranking Loss (New Data)', color=color, fontweight='bold')
                    ax1.plot(history['ft_rank_loss'], color=color, marker='o', linewidth=2, label='Ranking Loss')
                    ax1.tick_params(axis='y', labelcolor=color)
                    ax1.grid(True, linestyle=':', alpha=0.6)
                    
                    # Right Axis: Original MSE (Blue)
                    ax2 = ax1.twinx() 
                    color = 'tab:blue'
                    ax2.set_ylabel('MSE Loss (Original Val Data)', color=color, fontweight='bold') 
                    ax2.plot(history['orig_val_mse'], color=color, linestyle='--', marker='x', label='Original Val MSE')
                    ax2.tick_params(axis='y', labelcolor=color)
                    
                    plt.title('Phase 2: Fine-tuning Dynamics (Full Batch)')
                    fig.tight_layout() 
                    plt.savefig('finetune_dynamics.pdf', format='pdf')
                    print("Fine-tuning curve saved: finetune_dynamics.pdf")

                plot_finetune_dynamics(history_ft)

            else:
                print("Error: Fine-tuning dataset must contain both positive and negative samples.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Fine-tuning error: {e}")
else:
    print("Fine-tuning file 'Peplist_Score' not found.")