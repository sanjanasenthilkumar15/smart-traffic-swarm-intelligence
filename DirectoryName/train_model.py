"""
NEURAL NETWORK TRAINER - Traffic Signal Control AI

Trains a feed-forward neural network to predict optimal traffic signal phases.
Fixed version with proper feature dimension handling.
"""

# Standard library imports
import os
import sys
import warnings
import pickle

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Third-party imports
import numpy as np
import pandas as pd

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Matplotlib imports
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================
DATASET_FILE = "traffic_training_data.csv"
MODEL_OUTPUT = "traffic_model.pth"
SCALER_OUTPUT = "traffic_scaler.pkl"
CONFIG_OUTPUT = "model_config.pkl"

BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Expected number of input features (FIXED)
EXPECTED_INPUT_FEATURES = 9

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================
class TrafficSignalNN(nn.Module):
    """
    Feed-forward neural network for traffic signal control.
    
    Architecture:
        Input: 9 features (4 queues + 4 speeds + 1 emergency flag)
        Hidden Layer 1: 64 neurons + ReLU + Dropout
        Hidden Layer 2: 32 neurons + ReLU + Dropout
        Output: N neurons (number of traffic light phases)
    """
    
    def __init__(self, input_size=9, hidden1=64, hidden2=32, num_phases=4):
        super(TrafficSignalNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, num_phases)
        )
        
    def forward(self, x):
        return self.network(x)

# ============================================================================
# PYTORCH DATASET CLASS
# ============================================================================
class TrafficDataset(Dataset):
    """Custom PyTorch Dataset for traffic control data"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================
def load_and_preprocess_data(dataset_file):
    """
    Load CSV dataset and prepare for training.
    
    CRITICAL FIX: Handles CSV files with 10 or 11 columns properly.
    """
    
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    # Load CSV file
    try:
        data = pd.read_csv(dataset_file)
        print(f"✓ Loaded {len(data)} samples from {dataset_file}")
    except FileNotFoundError:
        print(f"✗ Dataset not found: {dataset_file}")
        print("  Run data_logger.py first to generate training data")
        return None, None, None, None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return None, None, None, None
    
    # Minimum sample check
    if len(data) < 100:
        print(f"⚠ Warning: Only {len(data)} samples. Recommend 1000+ for good training")
    
    # Show dataset info
    print(f"✓ Dataset shape: {data.shape}")
    print(f"✓ Columns ({len(data.columns)}): {list(data.columns)}")
    
    # Separate features and labels
    # Last column is always the label (next_phase)
    X = data.iloc[:, :-1].values  # All columns except last
    y = data.iloc[:, -1].values   # Last column only
    
    print(f"\n[DEBUG] Initial X shape: {X.shape}")
    print(f"[DEBUG] Initial y shape: {y.shape}")
    
    # CRITICAL FIX: Remove current_phase column if present
    # Expected columns in CSV:
    # queue_0, queue_1, queue_2, queue_3,          (4 columns)
    # speed_0, speed_1, speed_2, speed_3,          (4 columns)
    # emergency_flag,                               (1 column)
    # current_phase,                                (1 column - REMOVE THIS)
    # next_phase                                    (1 column - label)
    
    if X.shape[1] == 10:
        print(f"\n⚠ DETECTED 10 FEATURE COLUMNS (includes current_phase)")
        print(f"  Removing current_phase column (index 9)...")
        # Remove column at index 9 (current_phase)
        X = np.delete(X, 9, axis=1)
        print(f"✓ Feature columns reduced from 10 to 9")
    elif X.shape[1] == 9:
        print(f"✓ Correct number of features: 9")
    else:
        print(f"⚠ WARNING: Unexpected number of features: {X.shape[1]}")
        print(f"  Expected 9 or 10, got {X.shape[1]}")
        print(f"  Proceeding anyway, but results may be incorrect")
    
    print(f"\n✓ Final feature shape: {X.shape}")
    print(f"✓ Label shape: {y.shape}")
    
    # Verify final dimensions
    if X.shape[1] != EXPECTED_INPUT_FEATURES:
        print(f"\n✗ ERROR: Feature count mismatch!")
        print(f"  Expected: {EXPECTED_INPUT_FEATURES}")
        print(f"  Got: {X.shape[1]}")
        print(f"  This will cause training to fail!")
        return None, None, None, None
    
    # Determine number of unique phases
    num_phases = len(np.unique(y))
    print(f"✓ Number of traffic phases: {num_phases}")
    
    # Print phase distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"✓ Phase distribution:")
    for phase, count in zip(unique, counts):
        percentage = count / len(y) * 100
        print(f"    Phase {phase}: {count} samples ({percentage:.1f}%)")
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"✓ Features normalized (mean=0, std=1)")
    
    # Train/Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_SEED,
        stratify=y  # Maintain phase distribution in both sets
    )
    
    print(f"✓ Train set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Create PyTorch datasets
    train_dataset = TrafficDataset(X_train, y_train)
    test_dataset = TrafficDataset(X_test, y_test)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )
    
    print("✓ Data loaders created\n")
    
    return train_loader, test_loader, scaler, num_phases

# ============================================================================
# TRAINING LOOP
# ============================================================================
def train_model(model, train_loader, test_loader, epochs=EPOCHS):
    """Train the neural network"""
    
    print("=" * 80)
    print("TRAINING NEURAL NETWORK")
    print("=" * 80)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # TRAINING PHASE
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # VALIDATION PHASE
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        
        # Average test metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            print(f"  Test Loss:  {avg_test_loss:.4f} | Test Acc:  {test_accuracy:.2f}%")
    
    print(f"\n✓ Training completed!")
    print(f"✓ Final Test Accuracy: {test_accuracy:.2f}%\n")
    
    return model, history

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_training_history(history):
    """Plot training metrics"""
    
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Test Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
        ax2.plot(history['test_acc'], label='Test Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print(f"✓ Training plots saved to: training_history.png")
        
    except Exception as e:
        print(f"⚠ Could not save plot: {e}")

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================
def main():
    """Main training pipeline"""
    
    print("\n" + "=" * 80)
    print("EDGE AI TRAFFIC SIGNAL TRAINER")
    print("=" * 80 + "\n")
    
    # Step 1: Load and preprocess data
    train_loader, test_loader, scaler, num_phases = load_and_preprocess_data(DATASET_FILE)
    
    if train_loader is None:
        print("\n✗ Data loading failed. Exiting.")
        return
    
    # Step 2: Initialize model
    input_size = EXPECTED_INPUT_FEATURES  # Always 9
    model = TrafficSignalNN(
        input_size=input_size,
        hidden1=64,
        hidden2=32,
        num_phases=num_phases
    )
    
    print("=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)
    print()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}\n")
    
    # Step 3: Train model
    model, history = train_model(model, train_loader, test_loader, epochs=EPOCHS)
    
    # Step 4: Save model and config
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    try:
        # Save model weights
        torch.save(model.state_dict(), MODEL_OUTPUT)
        print(f"✓ Model weights saved to: {MODEL_OUTPUT}")
        
        # Save scaler
        with open(SCALER_OUTPUT, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✓ Feature scaler saved to: {SCALER_OUTPUT}")
        
        # Save model configuration
        config = {
            'input_size': input_size,
            'hidden1': 64,
            'hidden2': 32,
            'num_phases': num_phases
        }
        
        with open(CONFIG_OUTPUT, 'wb') as f:
            pickle.dump(config, f)
        print(f"✓ Model config saved to: {CONFIG_OUTPUT}")
        
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    # Step 5: Plot training history
    plot_training_history(history)
    
    # Final summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"✓ Model trained for {EPOCHS} epochs")
    print(f"✓ Final test accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"✓ Model size: {total_params:,} parameters")
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Run: python export_to_onnx.py")
    print("   (Converts model to edge-ready ONNX format)")
    print("\n2. Run: python adaptive_signal.py")
    print("   (Deploy AI controller in SUMO simulation)")
    print("=" * 80 + "\n")

# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
