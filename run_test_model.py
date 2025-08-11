import torch
from utils import load_trained_model, \
                    setup_inference_data, \
                    run_inference_on_sequences, \
                    evaluate_predictions_cpu_only


def main():
    """
    Complete inference pipeline matching your training setup
    """
    
    # 1. Load the trained model
    model_path = "./models/best_soccer_model.pth"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_trained_model(model_path, device)
    
    # 2. Setup inference data (same as training)
    data_path = './data/processed/testing'  
    dataset = setup_inference_data(data_path, seq_len=100, max_files=15)
    
    # 3. Run inference on all sequences
    print("Running inference on all sequences...")
    results = run_inference_on_sequences(model, dataset, device, batch_size=16)
    
    # 4. Evaluate results (using CPU-only version to avoid device issues)
    print("\nEvaluation Results:")
    metrics = evaluate_predictions_cpu_only(results, dataset)

if __name__ == "__main__":
    main()