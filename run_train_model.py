import torch
import pandas as pd
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils.soccerDatasets import SoccerPredictionDataset, SoccerSequenceDataset
from utils.multiAgentTransformer import MultiAgentTransformer, SoccerTrainer

FROM_SCRATCH = False

def load_trained_model(model_path: str, device: str = None):
    """
    Load the trained MultiAgentTransformer model with exact training configuration
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with EXACT same parameters as training
    model = MultiAgentTransformer(
        player_input_dim=22,  # Match your training config
        ball_input_dim=4,     # Match your training config
        d_model=128,          # Match your training config
        nhead=4,              # Match your training config
        num_layers=3,         # Match your training config
        forecast_horizon=1,   # Match your training config (you used 1, not 10)
        dropout=0.1           # Match your training config
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'Unknown')}")
    
    return model

if FROM_SCRATCH:
    # Initialize model
    model = MultiAgentTransformer(
        player_input_dim=22,  # Your 22 features per player
        ball_input_dim=4,     # Ball features
        d_model=128,#256,
        nhead=4,#8,
        num_layers=3,#6,
        forecast_horizon=1,
        dropout=0.1
    )
else:
    model_path = '/models/best_soccer_model.pth'
    model = load_trained_model(model_path)

# Dummy initialisation
dataset = SoccerPredictionDataset(seq_len=100, forecast_horizon=1)
trainer = SoccerTrainer(model, dataset, learning_rate=1e-4)

# Show model parameters
print(f"\nModel initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

# Initialise global training supervisor
global_metric_history = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'learning_rate': [],
    'player_mse': [],
    'player_mae': [],
    'player_rmse': [],
}

# Collect files available
files = glob.glob('./training/*.parquet')

batch_file = 15 # temp change
for file_idx in range(570, min(855, len(files)), batch_file): #len(files)
    # Initialize dataset
    # dataset = SoccerPredictionDataset(seq_len=100, forecast_horizon=1)
    
    # Load the specific data
    sublist_files = files[file_idx:file_idx+batch_file]
    dataset.load_specific(sublist_files)

    # Process features (now includes all tactical features)
    dataset.process_features()

    # Get feature information
    feature_info = dataset.get_feature_info()
    print("Features per player:", feature_info)

    # Create sequences
    X_player, X_ball, y_player, y_ball = dataset.create_sequences()

    # Split into train/val
    split_idx = int(0.5 * len(X_player))
        
    train_dataset = SoccerSequenceDataset(
        X_player[:split_idx], X_ball[:split_idx], 
        y_player[:split_idx], y_ball[:split_idx]
    )
    val_dataset = SoccerSequenceDataset(
        X_player[split_idx:], X_ball[split_idx:], 
        y_player[split_idx:], y_ball[split_idx:]
    )
        
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # Initialize trainer
    # trainer = SoccerTrainer(model, dataset, learning_rate=1e-4)
    trainer.update_dataset(dataset)
    trainer.reset_best_val_loss() # other wise a batch of data can be skipped and model is stuck in overfitting

    # Train the model
    trainer.train(train_loader, val_loader, epochs=100, visualize_every=10, 
                player_weight=1.0, ball_weight=2.0)
    
    # Get training evaluation
    current_metrics = trainer.get_training_history()
    
    # Extend global training evaluators
    for key in current_metrics:
        global_metric_history[key].extend(current_metrics[key])

model.eval()

# Plot 
def plot_global_metrics(global_metric_history):
        epochs_range = range(len(global_metric_history['epoch']))

        # 1. Loss Plot
        plt.figure()
        plt.plot(epochs_range, global_metric_history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, global_metric_history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("./analysis/training/Training_and_Validation_Loss.png")
        plt.close()

        # 2. Learning Rate Plot
        plt.figure()
        plt.plot(epochs_range, global_metric_history['learning_rate'], label='Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('LR')
        plt.title('Learning Rate over Epochs')
        plt.grid(True)
        plt.savefig("./analysis/training/Learning_Rate_over_Epochs.png")
        plt.close()

        # 2. MSE
        plt.figure()
        plt.plot(epochs_range, global_metric_history['player_mse'], label='MSE')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE over Epochs')
        plt.grid(True)
        plt.savefig("MSE_over_Epochs.png")
        plt.close()

        # 2. MAE
        plt.figure()
        plt.plot(epochs_range, global_metric_history['player_mae'], label='MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('MAE over Epochs')
        plt.grid(True)
        plt.savefig("./analysis/training/MAE_over_Epochs.png")
        plt.close()

        # 2. RMSE
        plt.figure()
        plt.plot(epochs_range, global_metric_history['player_rmse'], label='RMSE')
        plt.xlabel('Epoch')
        plt.ylabel('RMSE')
        plt.title('RMSE over Epochs')
        plt.grid(True)
        plt.savefig("./analysis/training/RMSE_over_Epochs.png")
        plt.close()

plot_global_metrics(global_metric_history)

training_df = pd.DataFrame({
    'Epoch': range(len(global_metric_history['epoch'])),
    'train_loss': global_metric_history['train_loss'],
    'val_loss': global_metric_history['val_loss'],
    'learning_rate': global_metric_history['learning_rate'],
    'player_mse': global_metric_history['player_mse'],
    'player_mae': global_metric_history['player_mae'],
    'player_rmse': global_metric_history['player_rmse'],
})

training_df.to_csv('./analysis/training/training_570_855.csv', index=False)

# Evaluate
metrics = trainer.evaluate_metrics(val_loader)
print("Evaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
