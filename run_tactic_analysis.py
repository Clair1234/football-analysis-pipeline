# imports
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

from .utils import load_trained_model, setup_inference_data
from .utils import SoccerTrainer
from .utils import SoccerSequenceDataset
from .utils import TacticalPatternAnalyzer   
from .utils import analyze_embedded_clusters_with_positional_data_enhanced  
from .utils import run_complete_advanced_analysis  
from .utils import TacticalExplainabilityLayer  

EXTRACT_PER_TEAM=True

# 0
print("0. Setting up data and model...")
# Run the complete inference pipeline
# 0.1. Load the trained model
model_path = "./models/best_soccer_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_trained_model(model_path, device)
# 0.2. Setup inference data (same as training)
data_path = './data/processed/testing'  
dataset = setup_inference_data(data_path, seq_len=100, max_files=100)
    
# Create a trainer just for visualization (you can also extract this functionality)
trainer = SoccerTrainer(model, dataset)
    
# Create a small validation loader for visualization
X_player, X_ball, y_player, y_ball = dataset.create_sequences() #here filtered

vis_dataset = SoccerSequenceDataset(
    X_player, X_ball, y_player, y_ball
)
vis_loader = DataLoader(vis_dataset, batch_size=4, shuffle=False)
trainer.visualize_attention(vis_loader, num_samples=1)

#1
print("1. Extracting embeddings and patterns from trained model...")
analysis_data = trainer.analyze_tactical_patterns_post_training(
    vis_loader, save_analysis=True
)

#2
print("2. Initializing tactical pattern analyzer...")
analyzer = TacticalPatternAnalyzer(analysis_data)

#3
print("3. Performing multi-dimensional clustering...")
clustering_results = analyzer.perform_clustering(
    layer='layer_-1',  # Use final layer embeddings
    feature_types=['hybrid', 'combined', 'positional', 'per_agent'],
    extract_per_team=EXTRACT_PER_TEAM
)

#4
print("4. Analyse the embedded clusters...")
results_embeddings = analyze_embedded_clusters_with_positional_data_enhanced(
     clustering_results, analyzer.metadata, 
     feature_type='combined', team_ids=[0,0,0,0,0,0,0,0,0,0,0,2], #team_ids=[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2]
     extract_per_team=EXTRACT_PER_TEAM
)

#5
print("5. Analyse the hybrid embedded clusters...")
results_embeddings = analyze_embedded_clusters_with_positional_data_enhanced(
     clustering_results, analyzer.metadata, 
     feature_type='hybrid', team_ids=[0,0,0,0,0,0,0,0,0,0,0,2], #team_ids=[0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2]
     extract_per_team=EXTRACT_PER_TEAM
)

#6
print("6. Cluster interpretation")
# Test different feature_type
advanced_results = run_complete_advanced_analysis(clustering_results, analyzer.metadata, layer='layer_-1', feature_type='combined', extract_per_team=EXTRACT_PER_TEAM)
advanced_results = run_complete_advanced_analysis(clustering_results, analyzer.metadata, layer='layer_-1', feature_type='hybrid', extract_per_team=EXTRACT_PER_TEAM)

#7
print("7. Initializing explainability layer...")
explainer = TacticalExplainabilityLayer(
    model=trainer.model,  # Your trained transformer model
    analysis_data=analysis_data,  # Same data used for clustering
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


#8
print("8. Performing explainability analysis...")
preferred_clustering = clustering_results['combined']['clustering']['kmeans']
cluster_labels = preferred_clustering['labels']
n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

print(f"Found {n_clusters} clusters for explainability analysis")

# Analyze each cluster
for cluster_id in range(n_clusters):
    print(f"\n--- Analyzing Cluster {cluster_id} ---")
    
    # 8a. Feature attribution analysis
    print(f"Computing feature attributions for cluster {cluster_id}...")
    attribution_results = explainer.compute_feature_attributions(
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
        layer='layer_-1',
        method= 'integrated_gradients'
    )
    
    # 8b. Attention pattern visualization
    print(f"Visualizing attention patterns for cluster {cluster_id}...")
    explainer.visualize_attention_patterns_with_average(
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
        layer='layer_-1',
        head=0,  # First attention head
        max_samples=3,  # Show 3 representative samples
        cluster_average = True
    )
    explainer.visualize_attention_patterns_with_average(
        cluster_labels=cluster_labels,
        cluster_id=cluster_id,
        layer='layer_-1',
        head=0,  # First attention head
        max_samples=3,  # Show 3 representative samples
        cluster_average = False
    )
    
#9
print("9. Advanced explainability analysis...")
# Try different attribution methods for comparison
attribution_methods = ['integrated_gradients']
feature_type = 'combined'
folder = 'analysis'
save_path = 'dimension_impact.txt'

report = []

preferred_clustering = clustering_results[feature_type]['clustering']['kmeans']
cluster_labels = preferred_clustering['labels']
n_clusters = len(np.unique(cluster_labels[cluster_labels >= 0]))

for method in attribution_methods:
    print(f"\nTesting attribution method: {method} for {feature_type}")
    report.append(f"Testing attribution method: {method} for {feature_type}")
    
    # Focus on the first cluster for method comparison
    test_cluster_id = 0
    for test_cluster_id in range(n_clusters):
        try:
            method_results = explainer.compute_feature_attributions(
                cluster_labels=cluster_labels,
                cluster_id=test_cluster_id,
                layer='layer_-1',
                method=method
            )
            
            if 'most_important_features' in method_results:
                top_features = method_results['most_important_features']
                print(f"  Top features ({method}): {top_features['names'][:5]}")
                print(f"  Feature scores: {[f'{s:.3f}' for s in top_features['importance_scores'][:5]]}")
                report.append(f"  Top features ({method}): {top_features['names'][:5]}")
                report.append(f"  Feature scores: {[f'{s:.3f}' for s in top_features['importance_scores'][:5]]}")
        
        except Exception as e:
            print(f"  Error with {method}: {e}")

report_text = "\n".join(report)

# Save report
os.makedirs(folder, exist_ok=True) 
report_text = "\n".join(report)
save_path = folder + "/" + save_path.split(".")[0] + "_" + feature_type + "." + save_path.split(".")[1]
with open(save_path, 'w') as f:
    f.write(report_text)
        
print(f"Tactical report saved to {save_path}")

