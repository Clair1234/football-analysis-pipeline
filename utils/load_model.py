import torch
from torch.utils.data import DataLoader
from .soccerDatasets import SoccerPredictionDataset, SoccerSequenceDataset
from .multiAgentTransformer import MultiAgentTransformer

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
        forecast_horizon=1,   # Match your training config 
        dropout=0.1           # Match your training config
    )
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    
    return model

def setup_inference_data(data_path: str, seq_len: int = 100, max_files: int = None):
    """
    Setup data for inference - should match your training data processing exactly
    """
    # Initialize dataset with same parameters as training
    dataset = SoccerPredictionDataset(seq_len=seq_len, forecast_horizon=1)
    
    # Load data
    if max_files:
        dataset.load_data_limited(data_path, max_files=max_files)
    else:
        dataset.load_data(data_path)
    
    # Process features (same as training)
    dataset.process_features()
    
    # Get feature info
    feature_info = dataset.get_feature_info()
    print("Features per player:", feature_info)
    
    return dataset

def run_inference_on_sequences(model, dataset, 
                               device='cpu', 
                               batch_size=16, 
                               filter_sequences=False):
    """
    Run inference on all sequences in the dataset
    """
    # Create sequences
    if filter_sequences:
        X_player, X_ball, y_player, y_ball = dataset.create_sequences_filtered()
    else:
        X_player, X_ball, y_player, y_ball = dataset.create_sequences()
    
    # Create PyTorch dataset
    inference_dataset = SoccerSequenceDataset(X_player, X_ball, y_player, y_ball)
    inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    all_player_preds = []
    all_ball_preds = []
    all_player_targets = []
    all_ball_targets = []
    
    with torch.no_grad():
        for batch in inference_loader:
            # Move to device
            player_states = batch['player_states'].to(device)
            ball_states = batch['ball_states'].to(device)
            target_players = batch['target_players'].to(device)
            target_ball = batch['target_ball'].to(device)
            
            # Get team assignments if available
            team_ids = batch.get('team_ids', None)
            #print("### DEBUG ###")
            #print(team_ids)
            if team_ids is not None:
                team_ids = team_ids.to(device)
            
            # Make predictions
            player_preds, ball_preds = model(player_states, ball_states, team_ids)
            
            # Store predictions and targets (keep on same device for now)
            all_player_preds.append(player_preds)
            all_ball_preds.append(ball_preds)
            all_player_targets.append(target_players)
            all_ball_targets.append(target_ball)
    
    # Concatenate all results (keep on same device)
    player_predictions = torch.cat(all_player_preds, dim=0)
    ball_predictions = torch.cat(all_ball_preds, dim=0)
    player_targets = torch.cat(all_player_targets, dim=0)
    ball_targets = torch.cat(all_ball_targets, dim=0)
    
    return {
        'player_predictions': player_predictions,
        'ball_predictions': ball_predictions,
        'player_targets': player_targets,
        'ball_targets': ball_targets
    }

def run_inference_on_single_sequence(model, player_sequence, ball_sequence, device='cpu'):
    """
    Run inference on a single sequence
    
    Args:
        model: Trained model
        player_sequence: (seq_len, num_players, features) - Raw sequence data
        ball_sequence: (seq_len, ball_features) - Raw ball sequence
        device: Device to run inference on
    
    Returns:
        player_prediction: (num_players, 1, 2) - Predicted positions
        ball_prediction: (1, 2) - Predicted ball position
    """
    model.eval()
    
    # Add batch dimension
    player_states = torch.FloatTensor(player_sequence).unsqueeze(0).to(device)  # (1, seq_len, players, features)
    ball_states = torch.FloatTensor(ball_sequence).unsqueeze(0).to(device)      # (1, seq_len, features)
    
    with torch.no_grad():
        player_pred, ball_pred = model(player_states, ball_states)
    
    return player_pred.squeeze(0).cpu().numpy(), ball_pred.squeeze(0).cpu().numpy()

def ensure_device_compatibility(dataset_processor, device):
    """
    Ensure all scaler tensors in the dataset processor are on the correct device
    """
    if hasattr(dataset_processor, 'position_scaler_torch'):
        for key in dataset_processor.position_scaler_torch:
            if isinstance(dataset_processor.position_scaler_torch[key], torch.Tensor):
                dataset_processor.position_scaler_torch[key] = dataset_processor.position_scaler_torch[key].to(device)
    
    if hasattr(dataset_processor, 'velocity_scaler_torch'):
        for key in dataset_processor.velocity_scaler_torch:
            if isinstance(dataset_processor.velocity_scaler_torch[key], torch.Tensor):
                dataset_processor.velocity_scaler_torch[key] = dataset_processor.velocity_scaler_torch[key].to(device)

def evaluate_predictions(results, dataset_processor=None, device='cpu'):
    """
    Evaluate prediction quality with metrics
    """
    player_preds = results['player_predictions']
    ball_preds = results['ball_predictions']
    player_targets = results['player_targets']
    ball_targets = results['ball_targets']
    
    # If we have a dataset processor with inverse transform, use it
    if hasattr(dataset_processor, 'inverse_transform_predictions_tensor'):
        # Ensure dataset processor scalers are on the same device as tensors
        ensure_device_compatibility(dataset_processor, player_preds.device)
        
        # Apply inverse transforms (keep on same device)
        player_preds = dataset_processor.inverse_transform_predictions_tensor(player_preds, kind="position")
        player_targets = dataset_processor.inverse_transform_predictions_tensor(player_targets, kind="position")
        ball_preds = dataset_processor.inverse_transform_predictions_tensor(ball_preds, kind="position")
        ball_targets = dataset_processor.inverse_transform_predictions_tensor(ball_targets, kind="position")
    
    # Move to CPU for metric calculations if needed
    if player_preds.device.type != 'cpu':
        player_preds_cpu = player_preds.cpu()
        player_targets_cpu = player_targets.cpu()
        ball_preds_cpu = ball_preds.cpu()
        ball_targets_cpu = ball_targets.cpu()
    else:
        player_preds_cpu = player_preds
        player_targets_cpu = player_targets
        ball_preds_cpu = ball_preds
        ball_targets_cpu = ball_targets
    
    # Calculate metrics
    player_mse = torch.nn.functional.mse_loss(player_preds_cpu, player_targets_cpu.transpose(1, 2)).item()
    ball_mse = torch.nn.functional.mse_loss(ball_preds_cpu, ball_targets_cpu).item()
    
    player_mae = torch.nn.functional.l1_loss(player_preds_cpu, player_targets_cpu.transpose(1, 2)).item()
    ball_mae = torch.nn.functional.l1_loss(ball_preds_cpu, ball_targets_cpu).item()
    
    print(f"Player MSE: {player_mse:.4f}")
    print(f"Player MAE: {player_mae:.4f}")
    print(f"Ball MSE: {ball_mse:.4f}")
    print(f"Ball MAE: {ball_mae:.4f}")
    
    return {
        'player_mse': player_mse,
        'player_mae': player_mae,
        'ball_mse': ball_mse,
        'ball_mae': ball_mae
    }

# Alternative approach: Move tensors to CPU before evaluation
def evaluate_predictions_cpu_only(results, dataset_processor=None):
    """
    Evaluate prediction quality with metrics - CPU only version
    """
    # Move all tensors to CPU first
    player_preds = results['player_predictions'].cpu()
    ball_preds = results['ball_predictions'].cpu()
    player_targets = results['player_targets'].cpu()
    ball_targets = results['ball_targets'].cpu()
    
    # If we have a dataset processor with inverse transform, use it
    if hasattr(dataset_processor, 'inverse_transform_predictions_tensor'):
        # Ensure dataset processor scalers are on CPU
        ensure_device_compatibility(dataset_processor, 'cpu')
        
        # Apply inverse transforms
        player_preds = dataset_processor.inverse_transform_predictions_tensor(player_preds, kind="position")
        player_targets = dataset_processor.inverse_transform_predictions_tensor(player_targets, kind="position")
        ball_preds = dataset_processor.inverse_transform_predictions_tensor(ball_preds, kind="position")
        ball_targets = dataset_processor.inverse_transform_predictions_tensor(ball_targets, kind="position")
    
    # Calculate metrics
    player_mse = torch.nn.functional.mse_loss(player_preds, player_targets.transpose(1, 2)).item()
    ball_mse = torch.nn.functional.mse_loss(ball_preds, ball_targets).item()
    
    player_mae = torch.nn.functional.l1_loss(player_preds, player_targets.transpose(1, 2)).item()
    ball_mae = torch.nn.functional.l1_loss(ball_preds, ball_targets).item()
    
    print(f"Player MSE: {player_mse:.4f}")
    print(f"Player MAE: {player_mae:.4f}")
    print(f"Ball MSE: {ball_mse:.4f}")
    print(f"Ball MAE: {ball_mae:.4f}")
    
    return {
        'player_mse': player_mse,
        'player_mae': player_mae,
        'ball_mse': ball_mse,
        'ball_mae': ball_mae
    }

def main_inference_pipeline(): #need to fix that
    """
    Complete inference pipeline matching your training setup
    """
    
    # 1. Load the trained model
    model_path = "models/best_soccer_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(model_path, device)
    
    # 2. Setup inference data (same as training)
    data_path = 'data/processed/testing'  
    dataset = setup_inference_data(data_path, seq_len=100, max_files=15)
    
    # 3. Run inference on all sequences
    print("Running inference on all sequences...")
    results = run_inference_on_sequences(model, dataset, device, batch_size=16)
    
    # 4. Evaluate results (using CPU-only version to avoid device issues)
    print("\nEvaluation Results:")
    metrics = evaluate_predictions_cpu_only(results, dataset)

    return model, dataset, None #results