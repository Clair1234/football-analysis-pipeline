from scipy.stats import ttest_ind
from collections import defaultdict
from collections import Counter
from scipy.spatial import ConvexHull
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict, Counter
import os
from scipy.stats import entropy
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

class TacticalPatternAnalyzer:
    """
    Comprehensive tactical pattern discovery using embeddings and attention patterns
    """
    
    def __init__(self, analysis_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize with data from SoccerTrainer.analyze_tactical_patterns_post_training()
        
        Args:
            analysis_data: Dict containing embeddings, metadata, attention patterns
        """
        self.device = device
        self.embeddings = analysis_data['embeddings']
        self.metadata = analysis_data['metadata'] 
        self.attention_patterns = analysis_data['attention_patterns']
        self.feature_importance = analysis_data['feature_importance']
        self.model_config = analysis_data['model_config']
        
        # Clustering results storage
        self.clustering_results = {}
        self.tactical_patterns = {}

    def prepare_clustering_features(self, layer='layer_-1', feature_type='per_agent', hybrid_weights={'embedding': 0.7, 'spatial': 0.3}): #used
        """ 
        Prepare different types of features for clustering.

        Args:
            layer: Which transformer layer embeddings to use.
            feature_type: One of:
                - 'per_agent': Each agent embedding is treated as a sample (N × num_agents, d_model)
                - 'player_only': Only players (excluding ball) flattened per sample (N, num_players * d_model)
                - 'ball_only': Only the ball embedding (N, d_model)
                - 'combined': All agent embeddings flattened (N, num_agents * d_model)
                - 'positional': Derived spatial features from metadata (e.g., centroids, zones)
                - 'hybrid': Agent embeddings flattened (N, num_agents * d_model) with spatial features from metadata
            hybrid_weights: Only for 'hybrid' feature_type, balance influence of embeddings vs spatial data
        """
        embeddings = self.embeddings[layer]  # Shape: (N, num_agents, d_model)
        metadata = self.metadata[layer]

        N, num_agents, d_model = embeddings.shape

        if feature_type == 'per_agent':
            # Treat each agent (player or ball) as a separate data point
            per_agent_embeddings = embeddings.view(-1, d_model)  # Shape: (N × num_agents, d_model)
            per_agent_embeddings = embeddings.reshape(-1, d_model) #here
            #print(embeddings.shape)
            #agent_first = np.transpose(embeddings, (1, 0, 2))
            #per_agent_embeddings = agent_first.reshape(23, -1)

            return per_agent_embeddings.numpy(), "Per-Agent Embeddings"

        elif feature_type == 'player_only':
            # Exclude the ball (assumes it's the last agent)
            player_embeddings = embeddings[:, :-1, :].reshape(N, -1)  # (N, num_players * d_model)
            return player_embeddings.numpy(), "Player Embeddings Only"

        elif feature_type == 'ball_only':
            # Use only the ball embedding
            ball_embeddings = embeddings[:, -1, :]  # (N, d_model)
            return ball_embeddings.numpy(), "Ball Embeddings Only"

        elif feature_type == 'combined':
            # Flatten all agents into one vector per sample
            combined_embeddings = embeddings.reshape(N, -1)  # (N, num_agents * d_model)
            return combined_embeddings.numpy(), "Combined Embeddings"

        elif feature_type == 'positional':
            # For clusterng purposes - creation of new positional features
            positional_features = self._create_positional_features(metadata)
            return positional_features, "Positional Features"
        
        elif feature_type == 'hybrid':
            combined_embeddings = embeddings.reshape(N, -1).numpy()
            hybrid_features = self._create_hybrid_features(combined_embeddings, metadata, hybrid_weights)
            return hybrid_features, "Hybrid Combined Embeddings and Positional"

        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")
        
    def _create_hybrid_features(self, embedding_features, metadata, feature_weights,extract_per_team:bool=True): #used
        """Create hybrid feature representations"""
        # Check parameters
        print(f"Embedding weight: {feature_weights['embedding']}")
        print(f"Spatial weight: {feature_weights['spatial']}")
        
        # Get raw positions and extract spatial features
        # Handle player_states
        if 'player_states' in metadata:
            if metadata['player_states'] is not None:
                raw_positions = metadata['player_states']
            else:
                raw_positions = np.random.randn(len(metadata.get('sequence_ids', [])), 20)
                print("Warning: 'player_states' key is missing or empty in metadata.")
        else:
            raw_positions = np.random.randn(len(metadata.get('sequence_ids', [])), 20)  
            print("Warning: 'player_states' key is missing or empty in metadata.")

        if extract_per_team:
            print("Extracting per team for raw_positions")

            # Split into team 0 (players 0-10) + ball (index 22)
            team_0_indices = list(range(11)) + [22]  # [0,1,...,10,22]
            team_0 = raw_positions[:, team_0_indices, :]  # Shape: [batch, 12, 2 or 3]

            # Split into team 1 (players 11-21) + ball (index 22)
            team_1_indices = list(range(11, 22)) + [22]  # [11,...,21,22]
            team_1 = raw_positions[:, team_1_indices, :]  # Shape: [batch, 12, 2 or 3]

            # Concatenate along batch dimension
            raw_positions = np.concatenate([team_0, team_1], axis=0)  # Shape: [2 * batch, 12, 2 or 3]

            print(f"Raw positions reshaped to: {raw_positions.shape}")


        spatial_features = extract_transformer_relevant_features(raw_positions)
        
        # Create spatial feature matrix
        spatial_matrix = np.column_stack([spatial_features[name] for name in spatial_features.keys()])
        spatial_matrix = np.nan_to_num(spatial_matrix)
        
        # Standardize both feature sets
        embedding_scaler = StandardScaler()
        spatial_scaler = StandardScaler()
        
        embedding_scaled = embedding_scaler.fit_transform(embedding_features)
        spatial_scaled = spatial_scaler.fit_transform(spatial_matrix)
        
        # Create hybrid feature matrix
        hybrid_features = np.concatenate([
            embedding_scaled * feature_weights['embedding'],
            spatial_scaled * feature_weights['spatial']
        ], axis=1)
        
        return hybrid_features
    
    def _create_positional_features(self, metadata): #used
        """Create tactical features based on spatial positions and relationships from player_states and ball_states"""
        # Extract player_states and ball_states from metadata
        player_states = metadata.get('player_states', None)  # Shape: (N, seq_len, 22, 22)
        ball_states = metadata.get('ball_states', None)      # Shape: (N, seq_len, 4)
        
        if player_states is None:
            print("No player_states data available in metadata")
            return np.random.randn(len(metadata.get('sequence_ids', [])), 20)
        
        # Convert to numpy if needed
        if isinstance(player_states, torch.Tensor):
            player_states = player_states.numpy()
        if isinstance(ball_states, torch.Tensor):
            ball_states = ball_states.numpy()
        
        N, seq_len, num_players, num_features = player_states.shape
        
        # Use the last timestep of each sequence for feature extraction
        # This represents the most recent state before prediction
        current_states = player_states[:, -1, :, :]  # (N, 22, 22)
        current_ball = ball_states[:, -1, :] if ball_states is not None else None  # (N, 4)
        
        features = []
        
        for i in range(N):
            states = current_states[i]  # (22, 22) - 22 players with 22 features each
            ball = current_ball[i] if current_ball is not None else None  # (4,) - ball state
            
            # Extract positions (first 2 features are position_x, position_y)
            positions = states[:, :2]  # (22, 2)
            
            # Split into teams (assuming first 11 are left team, last 11 are right team)
            left_team_pos = positions[:11]   # (11, 2)
            right_team_pos = positions[11:]  # (11, 2)
            
            feature_vec = []

            # Easier to re-calculate info - not very RAM efficient
            
            # 1. Team centroids (from positions)
            left_centroid = np.mean(left_team_pos, axis=0)
            right_centroid = np.mean(right_team_pos, axis=0)
            feature_vec.extend(left_centroid)   # 2 features
            feature_vec.extend(right_centroid)  # 2 features
            
            # 2. Team spreads/compactness
            left_distances = np.linalg.norm(left_team_pos - left_centroid, axis=1)
            right_distances = np.linalg.norm(right_team_pos - right_centroid, axis=1)
            
            feature_vec.append(np.mean(left_distances))   # Left team compactness
            feature_vec.append(np.std(left_distances))    # Left team spread
            feature_vec.append(np.mean(right_distances))  # Right team compactness
            feature_vec.append(np.std(right_distances))   # Right team spread
            
            # 3. Formation width and depth for each team
            # Left team
            left_x, left_y = left_team_pos[:, 0], left_team_pos[:, 1]
            left_width = np.max(left_x) - np.min(left_x)
            left_depth = np.max(left_y) - np.min(left_y)
            feature_vec.extend([left_width, left_depth])
            
            # Right team
            right_x, right_y = right_team_pos[:, 0], right_team_pos[:, 1]
            right_width = np.max(right_x) - np.min(right_x)
            right_depth = np.max(right_y) - np.min(right_y)
            feature_vec.extend([right_width, right_depth])
            
            # 4. Defensive and offensive lines for each team
            left_defensive = np.min(left_y)
            left_offensive = np.max(left_y)
            right_defensive = np.min(right_y)
            right_offensive = np.max(right_y)
            feature_vec.extend([left_defensive, left_offensive, right_defensive, right_offensive])
            
            # 5. Ball-related features (if available)
            if ball is not None:
                ball_pos = ball[:2]  # Ball position (x, y)
                ball_vel = ball[2:4] if len(ball) >= 4 else [0, 0]  # Ball velocity
                
                # Distance from ball to team centroids
                ball_to_left_dist = np.linalg.norm(ball_pos - left_centroid)
                ball_to_right_dist = np.linalg.norm(ball_pos - right_centroid)
                feature_vec.extend([ball_to_left_dist, ball_to_right_dist])
                
                # Ball velocity magnitude
                ball_speed = np.linalg.norm(ball_vel)
                feature_vec.append(ball_speed)
            else:
                # Add zeros if no ball data
                feature_vec.extend([0, 0, 0])
            
            # 6. Extract pre-computed tactical features from player states
            # Assuming features 8-18 contain tactical information based on your dataset structure
            # (own_team_density, own_team_compactness, own_team_spread, global_density, etc.)
            
            # Average tactical features across all players for global view
            tactical_features = states[:, 8:19]  # Features 8-18 (11 tactical features)
            avg_tactical = np.mean(tactical_features, axis=0)
            feature_vec.extend(avg_tactical)  # 11 features
            
            # 7. Score context (features 19-21: own_score, opp_score, score_difference)
            # Take from first player (should be same for all players)
            score_features = states[0, 19:22]  # 3 features
            feature_vec.extend(score_features)
            
            # 8. Velocities analysis
            velocities = states[:, 2:4]  # Features 2-3 are velocity_x, velocity_y
            
            # Average velocity magnitude for each team
            left_velocities = velocities[:11]
            right_velocities = velocities[11:]
            
            left_avg_speed = np.mean(np.linalg.norm(left_velocities, axis=1))
            right_avg_speed = np.mean(np.linalg.norm(right_velocities, axis=1))
            feature_vec.extend([left_avg_speed, right_avg_speed])
            
            # 9. Formation balance - distance between team centroids
            team_separation = np.linalg.norm(left_centroid - right_centroid)
            feature_vec.append(team_separation)
            
            features.append(feature_vec)
        
        features_array = np.array(features)
        
        # Print feature information for debugging
        print(f"Created positional features shape: {features_array.shape}")
        print(f"Feature vector length per sample: {len(feature_vec)}")
        
        return features_array
    
    def find_optimal_clusters(self, features, feature_name, max_clusters=40,folder: str = "analysis/visualisation"): #used
        """Find optimal number of clusters using multiple metrics"""
        if len(features) < 4:
            print(f"Too few samples ({len(features)}) for clustering analysis")
            return 3
            
        silhouette_scores = []
        calinski_scores = []
        inertias = []
        
        K_range = range(2, min(max_clusters + 1, len(features)))
        
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            if len(np.unique(labels)) > 1:  # Need at least 2 clusters for silhouette
                sil_score = silhouette_score(features, labels)
                cal_score = calinski_harabasz_score(features, labels)
                
                silhouette_scores.append(sil_score)
                calinski_scores.append(cal_score)
                inertias.append(kmeans.inertia_)
            else:
                silhouette_scores.append(0)
                calinski_scores.append(0)
                inertias.append(float('inf'))
        
        # Plot metrics
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(K_range, silhouette_scores, 'bo-')
        axes[0].set_title(f'Silhouette Score - {feature_name}')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Silhouette Score')
        
        axes[1].plot(K_range, calinski_scores, 'ro-')
        axes[1].set_title(f'Calinski-Harabasz Score - {feature_name}')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('CH Score')
        
        axes[2].plot(K_range, inertias, 'go-')
        axes[2].set_title(f'Inertia (Elbow Method) - {feature_name}')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Inertia')
        
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/cluster_metrics_{feature_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Choose optimal k based on silhouette score
        if silhouette_scores:
            optimal_k = K_range[np.argmax(silhouette_scores)]
            silhouette = max(silhouette_scores)
            print(f"Optimal clusters for {feature_name}: {optimal_k} (silhouette: {silhouette:.3f})")
            return optimal_k, silhouette
        else:
            return 3
    
    def perform_clustering(self, layer='layer_-1', feature_types=['combined', 'per_agent']): #used
        """
        Perform clustering with multiple algorithms and feature types
        """
        results = {}
        
        for feature_type in feature_types:
            print(f"\n=== Clustering with {feature_type} features ===")
            
            # Prepare features
            features, feature_name = self.prepare_clustering_features(layer, feature_type)
            
            if len(features) < 4:
                print(f"Skipping {feature_type} - insufficient data")
                continue
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Find optimal number of clusters
            optimal_k, silhouette = self.find_optimal_clusters(features_scaled, feature_name)
            
            # Perform different clustering algorithms
            clustering_results = {}
            
            # 1. K-Means
            kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(features_scaled)
            clustering_results['kmeans'] = {
                'labels': kmeans_labels,
                'centroids': kmeans.cluster_centers_,
                'inertia': kmeans.inertia_
            }

            agent_indices = np.tile(np.arange(23), 1000)  # size: N * 23

            
            
            # 2. DBSCAN
            eps_values = [0.3, 0.5, 0.8, 1.0, 1.5]
            best_dbscan = None
            best_score = -1
            
            for eps in eps_values:
                dbscan = DBSCAN(eps=eps, min_samples=3)
                dbscan_labels = dbscan.fit_predict(features_scaled)
                
                if len(np.unique(dbscan_labels)) > 1 and -1 not in dbscan_labels:
                    score = silhouette_score(features_scaled, dbscan_labels)
                    if score > best_score:
                        best_score = score
                        best_dbscan = {
                            'labels': dbscan_labels,
                            'eps': eps,
                            'n_clusters': len(np.unique(dbscan_labels)),
                            'silhouette': score
                        }
            
            if best_dbscan:
                clustering_results['dbscan'] = best_dbscan
            
            # 3. Hierarchical Clustering
            hierarchical = AgglomerativeClustering(n_clusters=optimal_k)
            hierarchical_labels = hierarchical.fit_predict(features_scaled)
            clustering_results['hierarchical'] = {
                'labels': hierarchical_labels,
                'n_clusters': optimal_k
            }
            
            # Store results
            results[feature_type] = {
                'features': features,
                'features_scaled': features_scaled,
                'scaler': scaler,
                'clustering': clustering_results,
                'optimal_k': optimal_k,
                'silhouette': silhouette
            }
            
            # Visualize results
            self.visualize_clustering_results(features_scaled, clustering_results, feature_name)
        
        self.clustering_results = results
        return results
    
    def visualize_clustering_results(self, features, clustering_results, feature_name,folder: str = "analysis/visualisation"): #used
        """Visualize clustering results using dimensionality reduction"""
        # Reduce dimensionality for visualization
        if features.shape[1] > 2:
            # Try both PCA and t-SNE
            pca = PCA(n_components=2, random_state=42)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features)-1))
            
            features_pca = pca.fit_transform(features)
            features_tsne = tsne.fit_transform(features)
        else:
            features_pca = features
            features_tsne = features
        
        n_methods = len(clustering_results)
        fig, axes = plt.subplots(2, n_methods, figsize=(5*n_methods, 10))
        if n_methods == 1:
            axes = axes.reshape(2, 1)
        
        for i, (method, results) in enumerate(clustering_results.items()):
            labels = results['labels']
            n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1) for DBSCAN
            
            # PCA plot
            scatter = axes[0, i].scatter(features_pca[:, 0], features_pca[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7)
            axes[0, i].set_title(f'{method.upper()} - PCA\n{feature_name}\n{n_clusters} clusters')
            axes[0, i].set_xlabel('PC1')
            axes[0, i].set_ylabel('PC2')
            
            # t-SNE plot
            scatter = axes[1, i].scatter(features_tsne[:, 0], features_tsne[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7)
            axes[1, i].set_title(f'{method.upper()} - t-SNE\n{feature_name}\n{n_clusters} clusters')
            axes[1, i].set_xlabel('t-SNE 1')
            axes[1, i].set_ylabel('t-SNE 2')
        
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/clustering_visualization_{feature_name.replace(" ", "_")}.png', 
                   dpi=150, bbox_inches='tight')
        plt.show()

    def investigate_data_structure_bug(self, layer='layer_-1'): #eep
        """
        Investigate the critical bug in data structure that's causing impossible clustering results.
        """
        print("=== CRITICAL BUG INVESTIGATION ===")
        
        # Check original embeddings shape
        embeddings = self.embeddings[layer]
        N, num_agents, d_model = embeddings.shape
        print(f"Original embeddings shape: {embeddings.shape}")
        print(f"N={N}, num_agents={num_agents}, d_model={d_model}")
        
        # Check how per_agent features are created
        print(f"\n=== CHECKING FEATURE PREPARATION ===")
        
        # Replicate the per_agent feature preparation
        per_agent_embeddings = embeddings.reshape(-1, d_model)
        print(f"Reshaped to: {per_agent_embeddings.shape}")
        print(f"Expected shape: ({N * num_agents}, {d_model}) = ({N * num_agents}, {d_model})")
        
        # Check if reshape is correct by examining first few points
        print(f"\n=== CHECKING RESHAPE LOGIC ===")
        print("First 10 reshaped indices should map to:")
        for i in range(10):
            sample_idx = i // num_agents
            agent_idx = i % num_agents
            print(f"  Index {i}: sample {sample_idx}, agent {agent_idx}")
        
        # Check what the actual clustering labels look like
        if hasattr(self, 'clustering_results'):
            results = self.clustering_results['per_agent']['clustering']['kmeans']
            labels = results['labels']
            
            print(f"\n=== CHECKING CLUSTERING LABELS ===")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            print(f"First 50 labels: {labels[:50]}")
            
            # Check label distribution
            label_counts = np.bincount(labels)
            print(f"Label counts: {label_counts}")
            print(f"All clusters have same size: {len(set(label_counts)) == 1}")
            
            # This is the smoking gun - check if labels are just repeated patterns
            print(f"\n=== CHECKING FOR LABEL PATTERNS ===")
            print("First 100 labels:")
            for i in range(0, min(100, len(labels)), 10):
                print(f"  {i:2d}-{i+9:2d}: {labels[i:i+10]}")
            
            # Check if labels repeat every 23 elements
            print(f"\n=== CHECKING LABEL PERIODICITY ===")
            period_23 = True
            for i in range(min(1000, len(labels))):
                if labels[i] != labels[i % 23]:
                    period_23 = False
                    break
            print(f"Labels repeat every 23 elements: {period_23}")
            
            if period_23:
                print("*** BUG FOUND: Labels are repeating every 23 elements! ***")
                print("This means the clustering is not working on the full dataset.")
                print("Instead, it's clustering just 23 points and repeating the result.")
        
        # Let's also check the actual features that went into clustering
        if hasattr(self, 'clustering_results'):
            features = self.clustering_results['per_agent']['features']
            features_scaled = self.clustering_results['per_agent']['features_scaled']

            print(f"\n=== Embeddings vs reshape ===")
            print(f"Agent0, sample0: True: {embeddings[0][0][0]}, Reshape: {features[0][0]}")
            print(f"Agent1, sample0: True: {embeddings[0][1][0]}, Reshape: {features[1][0]}")
            print(f"Agent2, sample0: True: {embeddings[0][2][0]}, Reshape: {features[2][0]}")
            print(f"Agent22, sample0: True: {embeddings[0][22][0]}, Reshape: {features[22][0]}")
            print(f"Agent0, sample1: True: {embeddings[1][0][0]}, Reshape: {features[23][0]}")
            
            print(f"\n=== CHECKING FEATURE MATRIX ===")
            print(f"Features shape: {features.shape}")
            print(f"Features_scaled shape: {features_scaled.shape}")
            
            # Check if features are actually different
            # If clustering is wrong, we might have identical features
            print(f"\n=== CHECKING FEATURE UNIQUENESS ===")
            unique_features = np.unique(features_scaled, axis=0)
            print(f"Unique feature vectors: {unique_features.shape[0]} out of {features_scaled.shape[0]}")
            
            if unique_features.shape[0] < features_scaled.shape[0]:
                print("*** BUG CONFIRMED: Features are not unique! ***")
                print("Many feature vectors are identical, explaining the clustering pattern.")
                
                # Find which features are duplicated
                feature_strings = [str(row) for row in features_scaled]
                duplicates = Counter(feature_strings)
                most_common = duplicates.most_common(5)
                
                print(f"Most common feature vectors:")
                for i, (feature_str, count) in enumerate(most_common):
                    print(f"  Pattern {i+1}: appears {count} times")
        
        return None

    def debug_clustering_pattern(self, layer='layer_-1', feature_type='per_agent', method='kmeans'): #eep
        """
        Debug the clustering pattern to understand why we get consecutive agent groupings.
        """
        if not hasattr(self, 'clustering_results'):
            print("No clustering results found. Run perform_clustering() first.")
            return
        
        results = self.clustering_results[feature_type]['clustering'][method]
        labels = results['labels']
        features_scaled = self.clustering_results[feature_type]['features_scaled']
        
        N, num_agents, d_model = self.embeddings[layer].shape
        
        print(f"=== CLUSTERING DEBUG ANALYSIS ===")
        print(f"Total data points: {len(labels)}")
        print(f"Expected: {N} samples × {num_agents} agents = {N * num_agents}")
        print(f"Unique clusters: {len(np.unique(labels))}")
        
        # Analyze the pattern of consecutive agents
        print(f"\n=== CONSECUTIVE AGENT ANALYSIS ===")
        
        cluster_agent_patterns = {}
        
        for cluster_id in np.unique(labels):
            if cluster_id == -1:  # Skip noise
                continue
                
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            # Get agent indices for this cluster
            agent_indices = [idx % num_agents for idx in cluster_indices]
            unique_agents = sorted(set(agent_indices))
            
            # Check if agents are consecutive
            is_consecutive = True
            if len(unique_agents) > 1:
                # Handle wrapping (e.g., agents 22, 0, 1, 2)
                diffs = []
                for i in range(len(unique_agents)):
                    next_idx = (i + 1) % len(unique_agents)
                    diff = (unique_agents[next_idx] - unique_agents[i]) % num_agents
                    diffs.append(diff)
                
                # Check if differences are mostly 1 (consecutive) or large (wrapping)
                consecutive_count = sum(1 for d in diffs if d == 1)
                is_consecutive = consecutive_count >= len(diffs) - 1
            
            cluster_agent_patterns[cluster_id] = {
                'agents': unique_agents,
                'is_consecutive': is_consecutive,
                'size': len(cluster_indices),
                'agent_counts': {agent: agent_indices.count(agent) for agent in unique_agents}
            }
        
        # Print pattern analysis
        consecutive_clusters = sum(1 for p in cluster_agent_patterns.values() if p['is_consecutive'])
        print(f"Consecutive agent clusters: {consecutive_clusters}/{len(cluster_agent_patterns)}")
        
        # Analyze cluster sizes
        cluster_sizes = [p['size'] for p in cluster_agent_patterns.values()]
        print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, avg={np.mean(cluster_sizes):.1f}")
        
        # Check if each agent appears equally often
        print(f"\n=== AGENT DISTRIBUTION ANALYSIS ===")
        agent_total_appearances = {}
        for cluster_info in cluster_agent_patterns.values():
            for agent, count in cluster_info['agent_counts'].items():
                agent_total_appearances[agent] = agent_total_appearances.get(agent, 0) + count
        
        appearances = list(agent_total_appearances.values())
        print(f"Agent appearances: min={min(appearances)}, max={max(appearances)}, avg={np.mean(appearances):.1f}")
        print(f"Expected appearances per agent: {N} (once per sample)")
        
        # Check for the specific pattern you observed
        print(f"\n=== SPECIFIC PATTERN ANALYSIS ===")
        print("Clusters with exactly 4 agents:")
        four_agent_clusters = 0
        for cluster_id, info in cluster_agent_patterns.items():
            if len(info['agents']) == 4:
                four_agent_clusters += 1
                print(f"  Cluster {cluster_id}: agents {info['agents']} (consecutive: {info['is_consecutive']})")
        
        print(f"Total 4-agent clusters: {four_agent_clusters}")
        
        # Investigate the mathematical relationship
        print(f"\n=== MATHEMATICAL RELATIONSHIP ===")
        print(f"If we have {num_agents} agents and {len(cluster_agent_patterns)} clusters:")
        print(f"Agents per cluster: {num_agents / len(cluster_agent_patterns):.2f}")
        print(f"This explains why each cluster has ~4 agents (23/6 ≈ 3.8)")
        
        return cluster_agent_patterns
    
def generate_tactical_report(analyzer, folder='analysis/interpretation', save_path='tactical_pattern_report.txt'): #here mght eep
    """
    Generate a comprehensive tactical analysis report
    """
    report = []
    report.append("=== SOCCER TACTICAL PATTERN ANALYSIS REPORT ===\n")
        
    # Summary of clustering results
    report.append("CLUSTERING SUMMARY:")
    for feature_type, results in analyzer.clustering_results.items():
        report.append(f"\n{feature_type.upper()} Features:")
        for method, clustering in results['clustering'].items():
            labels = clustering['labels']
            silhouette = results['silhouette']
            n_clusters = len(np.unique(labels[labels >= 0]))
            report.append(f"  - {method}: {n_clusters} patterns identified - silhouette: {silhouette}")
            for cluster_id in range(n_clusters):
                n_samples = np.sum(labels == cluster_id)
                report.append(f"     - Cluster {cluster_id}: {n_samples} samples")
        
    report_text = "\n".join(report)

    # Save report
    os.makedirs(folder, exist_ok=True) 
    report_text = "\n".join(report)
    save_path = folder + "/" + save_path.split(".")[0] + "." + save_path.split(".")[1]
    with open(save_path, 'w') as f:
        f.write(report_text)
        
    print(f"Tactical report saved to {save_path}")
    #print("\n" + report_text)
        
    return report_text

def analyze_embedded_clusters_with_positional_data_new(clustering_results, metadata, layer='layer_-1', 
                                                   feature_type='combined', algorithm='kmeans', team_ids=None, all_features=None): #used
    """
    Analyze embedded clusters by examining similarities in underlying positional raw data.
    
    Args:
        clustering_results: Results from perform_clustering()
        metadata: Raw positional metadata corresponding to the embeddings
        layer: Which layer's clustering results to analyze
        feature_type: Which feature type clustering to analyze ('combined', 'per_agent', etc.)
        algorithm: Which clustering algorithm results to use ('kmeans', 'dbscan', 'hierarchical')
        team_ids: List of team IDs for each player/agent (e.g., [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,2])
                 If None, analyze all players together. If provided, analyze each team separately.
                 Note: Ball (team_id=2) is excluded from player_states analysis.
    
    Returns:
        dict: Comprehensive analysis of clusters based on positional data
              If team_ids is None: single analysis
              If team_ids is provided: {'team_0': analysis, 'team_1': analysis, 'full': analysis}
    """
    
    # If no team_ids provided, run original analysis
    if team_ids is None:
        return _analyze_single_team(clustering_results, metadata, layer, feature_type, algorithm, None, "Full Team")
    
    # Validate team_ids
    team_ids = np.array(team_ids)
    unique_teams = np.unique(team_ids)
    player_teams = unique_teams[unique_teams != 2]  # Exclude ball (team_id=2)
    
    print(f"Analyzing teams: {player_teams} (excluding ball with team_id=2)")
    print(f"Team distribution: {[(team, np.sum(team_ids == team)) for team in unique_teams]}")
    
    # Run analysis for each team and full dataset
    results = {}
    
    # Full analysis (original behavior)
    print("\n" + "="*80)
    print("ANALYZING FULL DATASET (ALL TEAMS)")
    print("="*80)
    results['full'] = _analyze_single_team(clustering_results, metadata, layer, feature_type, algorithm, None, "Full Dataset")
    
    # Team-specific analyses
    for team_id in player_teams:
        team_mask = team_ids == team_id
        team_player_indices = np.where(team_mask)[0]
        
        print(f"\n" + "="*80)
        print(f"ANALYZING TEAM {team_id} ({np.sum(team_mask)} players)")
        print("="*80)
        
        results[f'team_{team_id}'] = _analyze_single_team(
            clustering_results, metadata, layer, feature_type, algorithm, 
            team_player_indices, f"Team {team_id}"
        )
    
    return results


def _analyze_single_team(clustering_results, metadata, layer, feature_type, algorithm, 
                        team_player_indices, team_name, extract_per_team:bool=True): #used
    """
    Internal function to analyze a single team or full dataset.
    
    Args:
        team_player_indices: None for full analysis, or array of player indices for team analysis
        team_name: String name for this analysis (for printing)
    """
   
    
    # Extract clustering information
    if feature_type not in clustering_results:
        print(f"Feature type '{feature_type}' not found in clustering results")
        return None
        
    cluster_info = clustering_results[feature_type]['clustering']
    if algorithm not in cluster_info:
        print(f"Algorithm '{algorithm}' not found for feature type '{feature_type}'")
        return None
    
    labels = cluster_info[algorithm]['labels']
    n_clusters = len(np.unique(labels[labels >= 0]))  # Exclude noise points (-1) for DBSCAN
    
    print(f"Analyzing {n_clusters} clusters from {algorithm} clustering on {feature_type} features ({team_name})")
    
    # Get positional metadata
    if layer in metadata:
        pos_metadata = metadata[layer]
        if isinstance(pos_metadata, dict) and 'player_states' in pos_metadata:
            raw_positions = pos_metadata['player_states']
        else:
            raw_positions = pos_metadata
    else:
        print(f"Layer '{layer}' not found in metadata")
        return None

    if extract_per_team:
        print("Extracting per team for raw_positions")

        # Split into team 0 (players 0-10) (no ball (index 22))
        team_0_indices = list(range(11)) # [0,1,...,10]
        team_0 = raw_positions[:, :, team_0_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Split into team 1 (players 11-21) (no ball (index 22))
        team_1_indices = list(range(11, 22))  # [11,...,21,22]
        team_1 = raw_positions[:, :, team_1_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Concatenate along batch dimension
        raw_positions = np.concatenate([team_0, team_1], axis=0)  # Shape: [2 * batch, 11, 2 or 3]

        print(f"Raw positions reshaped to: {raw_positions.shape}")
        
    # Filter positions by team if specified
    if team_player_indices is not None:
        raw_positions = _filter_positions_by_team(raw_positions, team_player_indices)
        
        # Also need to filter labels if they correspond to per-agent analysis
        if feature_type == 'per_agent':
            labels = _filter_labels_by_team(labels, team_player_indices, raw_positions.shape)
    
    print(f"Raw positions shape: {raw_positions.shape if hasattr(raw_positions, 'shape') else len(raw_positions)}")
    
    # Initialize analysis results
    analysis = {
        'team_info': {
            'name': team_name,
            'player_indices': team_player_indices.tolist() if team_player_indices is not None else None,
            'n_players': len(team_player_indices) if team_player_indices is not None else 'all'
        },
        'cluster_summary': {},
        'positional_statistics': {},
        'cluster_characteristics': {},
        'discriminative_features': {},
        'visualization_data': {}
    }
    
    # Handle per_agent feature type differently (reshape labels to match samples)
    if feature_type == 'per_agent':
        # Labels are per agent, need to map back to samples
        n_samples = len(raw_positions)
        if team_player_indices is not None:
            n_agents = len(team_player_indices)
        else:
            n_agents = len(labels) // n_samples
            
        sample_labels = []
        
        for i in range(n_samples):
            agent_labels = labels[i*n_agents:(i+1)*n_agents]
            # Use most common cluster for this sample, or majority vote
            unique_labels, counts = np.unique(agent_labels[agent_labels >= 0], return_counts=True)
            if len(unique_labels) > 0:
                sample_labels.append(unique_labels[np.argmax(counts)])
            else:
                sample_labels.append(-1)  # No valid clusters
        labels = np.array(sample_labels)
        n_clusters = len(np.unique(labels[labels >= 0]))
    
    # Extract positional features from metadata - IMPROVED VERSION
    positional_features = extract_meaningful_positional_features(raw_positions)
    # print(f"Extracted features: {list(positional_features.keys())}")
    # print(f"Feature shapes: {[(k, v.shape) for k, v in positional_features.items()]}")
    
    # Filter out noise points for DBSCAN
    valid_mask = labels >= 0
    valid_labels = labels[valid_mask]
    valid_features = {key: val[valid_mask] for key, val in positional_features.items()}
    
    print(f"Valid samples: {len(valid_labels)} out of {len(labels)}")
    
    # Check if features have any variation
    for feature_name, feature_values in valid_features.items():
        std_val = np.std(feature_values)
        if std_val < 1e-6:
            print(f"Warning: Feature '{feature_name}' has very low variance (std={std_val:.2e})")
    
    # Analyze each cluster
    for cluster_id in range(n_clusters):
        cluster_mask = valid_labels == cluster_id
        cluster_size = np.sum(cluster_mask)
        
        print(f"\n=== Cluster {cluster_id} Analysis ({cluster_size} samples) ===")
        
        cluster_analysis = {
            'size': cluster_size,
            'percentage': (cluster_size / len(valid_labels)) * 100,
            'positional_stats': {},
            'characteristic_patterns': []
        }
        
        # Analyze each positional feature with improved statistics
        for feature_name, feature_values in valid_features.items():
            cluster_values = feature_values[cluster_mask]
            other_values = feature_values[~cluster_mask]
            
            # Skip if no variation
            if np.std(feature_values) < 1e-6:
                continue
            
            # Basic statistics
            stats = {
                'mean': np.mean(cluster_values),
                'std': np.std(cluster_values),
                'median': np.median(cluster_values),
                'min': np.min(cluster_values),
                'max': np.max(cluster_values),
                'q25': np.percentile(cluster_values, 25),
                'q75': np.percentile(cluster_values, 75)
            }
            
            # Compare with other clusters
            global_mean = np.mean(feature_values)
            global_std = np.std(feature_values)
            
            # Z-score for cluster mean vs global (more robust)
            if global_std > 1e-6:
                z_score = (stats['mean'] - global_mean) / global_std
            else:
                z_score = 0
            
            # Statistical significance test (t-test is more appropriate for means)
            try:
                t_stat, p_value = ttest_ind(cluster_values, other_values)
                significant = p_value < 0.05
                effect_size = abs(t_stat) / np.sqrt(len(cluster_values) + len(other_values))
            except:
                t_stat, p_value, significant, effect_size = 0, 1, False, 0
            
            # Additional measures
            # Cohen's d for effect size
            pooled_std = np.sqrt(((len(cluster_values) - 1) * np.var(cluster_values, ddof=1) + 
                                 (len(other_values) - 1) * np.var(other_values, ddof=1)) / 
                                (len(cluster_values) + len(other_values) - 2))
            if pooled_std > 1e-6:
                cohens_d = (np.mean(cluster_values) - np.mean(other_values)) / pooled_std
            else:
                cohens_d = 0
            
            feature_analysis = {
                **stats,
                'global_mean': global_mean,
                'global_std': global_std,
                'z_score': z_score,
                'p_value': p_value,
                'significant': significant,
                'cohens_d': cohens_d,
                'effect_size': effect_size,
                'distinctiveness': max(abs(z_score), abs(cohens_d))  # Use stronger measure
            }
            
            cluster_analysis['positional_stats'][feature_name] = feature_analysis
            
            # Identify characteristic patterns with stricter criteria
            if significant and abs(z_score) > 0.5:  # Lower threshold but still meaningful
                direction = "higher" if z_score > 0 else "lower"
                magnitude = "significantly" if abs(z_score) > 1.0 else "moderately"
                cluster_analysis['characteristic_patterns'].append(
                    f"{feature_name}: {magnitude} {direction} than average "
                    f"(z={z_score:.2f}, d={cohens_d:.2f}, p={p_value:.3f})"
                )
            
            # Add extreme value patterns
            if abs(cohens_d) > 0.5:  # Medium effect size
                direction = "much higher" if cohens_d > 0 else "much lower"
                cluster_analysis['characteristic_patterns'].append(
                    f"{feature_name}: {direction} values (effect size d={cohens_d:.2f})"
                )
        
        analysis['cluster_summary'][cluster_id] = cluster_analysis
        
        print(f"Size: {cluster_size} samples ({cluster_analysis['percentage']:.1f}%)")
        # print("Key characteristics:")
        # if cluster_analysis['characteristic_patterns']:
        #     for pattern in cluster_analysis['characteristic_patterns'][:5]:  # Show top 5
        #         print(f"  - {pattern}")
        # else:
        #     print("  - No significant distinguishing characteristics found")
        #    # Show top differences anyway
        #     feature_diffs = []
        #     for feature_name, feature_stats in cluster_analysis['positional_stats'].items():
        #         feature_diffs.append((feature_name, abs(feature_stats.get('z_score', 0))))
        #     feature_diffs.sort(key=lambda x: x[1], reverse=True)
        #     print("  - Top differences (even if not significant):")
        #     for fname, diff in feature_diffs[:3]:
        #         fstats = cluster_analysis['positional_stats'][fname]
        #         direction = "higher" if fstats.get('z_score', 0) > 0 else "lower"
        #         print(f"    • {fname}: {direction} (z={fstats.get('z_score', 0):.3f})")
    
    # Find most discriminative features across all clusters (improved)
    feature_discriminativeness = defaultdict(list)
    feature_significance = defaultdict(list)
    
    for cluster_id, cluster_data in analysis['cluster_summary'].items():
        for feature_name, feature_stats in cluster_data['positional_stats'].items():
            feature_discriminativeness[feature_name].append(feature_stats.get('distinctiveness', 0))
            feature_significance[feature_name].append(1 if feature_stats.get('significant', False) else 0)
    
    # Rank features by their ability to discriminate between clusters
    discriminative_ranking = {}
    for feature_name, distinctiveness_scores in feature_discriminativeness.items():
        avg_distinctiveness = np.mean(distinctiveness_scores)
        max_distinctiveness = np.max(distinctiveness_scores)
        significance_ratio = np.mean(feature_significance[feature_name])
        
        # Combined score favoring significant and distinctive features
        score = avg_distinctiveness * max_distinctiveness * (1 + significance_ratio)
        
        discriminative_ranking[feature_name] = {
            'avg_distinctiveness': avg_distinctiveness,
            'max_distinctiveness': max_distinctiveness,
            'significance_ratio': significance_ratio,
            'score': score
        }
    
    # Sort by discriminative power
    sorted_features = sorted(discriminative_ranking.items(), 
                           key=lambda x: x[1]['score'], reverse=True)
    
    analysis['discriminative_features'] = dict(sorted_features)
    
    print(f"\n=== Most Discriminative Positional Features ({team_name}) ===")
    for i, (feature_name, scores) in enumerate(sorted_features[:10]):
        print(f"{i+1}. {feature_name}: score={scores['score']:.3f} "
              f"(avg={scores['avg_distinctiveness']:.2f}, "
              f"max={scores['max_distinctiveness']:.2f}, "
              f"sig_ratio={scores['significance_ratio']:.2f})")
    
    # Create summary statistics
    print(f"\n=== Cluster Analysis Summary ({team_name}) ===")
    total_significant_features = sum(1 for _, scores in sorted_features 
                                   if scores['significance_ratio'] > 0)
    print(f"Features with significant differences: {total_significant_features}/{len(sorted_features)}")
    
    if total_significant_features == 0:
        print("WARNING: No features show significant differences between clusters!")
        print("This suggests:")
        print("  1. Clusters may be based on high-level patterns not captured by basic positional features")
        print("  2. The embedding space captures more subtle relationships")
        print("  3. Consider extracting more sophisticated features (tactical patterns, etc.)")
    
    # Create visualization data
    vis_data = prepare_cluster_visualization_data(valid_features, valid_labels, n_clusters)
    analysis['visualization_data'] = vis_data
    
    # Plot cluster comparisons for top discriminative features
    if len(sorted_features) > 0:
        plot_improved_cluster_comparison(valid_features, valid_labels, sorted_features[:6], n_clusters, team_name,
                                        feature_type=feature_type)
    
    return analysis


def _filter_positions_by_team(raw_positions, team_player_indices): #used
    """
    Filter raw positions to include only players from specified team.
    
    Args:
        raw_positions: Original position data
        team_player_indices: Indices of players to keep
    
    Returns:
        Filtered position data
    """
    
    # Convert to numpy if it's a torch tensor
    if isinstance(raw_positions, torch.Tensor):
        positions = raw_positions.numpy()
    else:
        positions = np.array(raw_positions)
    
    if len(positions.shape) == 4:  # (N, seq_len, num_agents, feature_dim)
        filtered_positions = positions[:, :, team_player_indices, :]
    elif len(positions.shape) == 3:  # (N, num_agents, feature_dim)
        filtered_positions = positions[:, team_player_indices, :]
    else:
        print(f"Warning: Unexpected position shape {positions.shape}, returning original")
        return raw_positions
    
    return filtered_positions


def _filter_labels_by_team(labels, team_player_indices, position_shape): #used
    """
    Filter labels for per_agent analysis to match team players.
    
    Args:
        labels: Original labels (per agent)
        team_player_indices: Indices of team players
        position_shape: Shape of position data to determine sample structure
        
    Returns:
        Filtered labels
    """
    
    if len(position_shape) == 4:  # (N, seq_len, num_agents, feature_dim)
        n_samples, _, total_agents, _ = position_shape
    elif len(position_shape) == 3:  # (N, num_agents, feature_dim)
        n_samples, total_agents, _ = position_shape
    else:
        print(f"Warning: Cannot filter labels for shape {position_shape}")
        return labels
    
    # Reshape labels to (n_samples, total_agents)
    labels_reshaped = labels.reshape(n_samples, total_agents)
    
    # Filter by team indices
    filtered_labels = labels_reshaped[:, team_player_indices]
    
    # Flatten back
    return filtered_labels.flatten()

def get_features(raw_positions): #used
    # Extract basic positional features
    basic_features = extract_meaningful_positional_features(raw_positions)
        
    # Extract transformer-relevant features
    transformer_features = extract_transformer_relevant_features(raw_positions)
        
    # Extract formation shape analysis
    #geometry_features = extract_formation_geometry(raw_positions)
        
    # Combine all features
    all_features = {}
    all_features.update(basic_features)
    all_features.update(transformer_features)
    #all_features.update(geometry_features)
    # print(all_features.keys())
        
    return all_features


def extract_meaningful_positional_features(raw_positions): #used
    """
    Extract more meaningful and robust positional features.
    Focus on features that capture tactical and spatial patterns.
    """
    
    features = {}
    
    # Convert to numpy if it's a torch tensor
    if isinstance(raw_positions, torch.Tensor):
        positions = raw_positions.numpy()
    else:
        positions = np.array(raw_positions)
    
    print(f"Processing positions with shape: {positions.shape}")
    
    if len(positions.shape) == 4:  # (N, seq_len, num_agents, feature_dim)
        N, seq_len, num_agents, feature_dim = positions.shape
        
        # Use last timestep for current state
        last_positions = positions[:, -1, :, :]  # (N, num_agents, feature_dim)
        
        # For sports data, often first 2 dims are x,y coordinates
        if feature_dim >= 2:
            xy_positions = last_positions[:, :, :2]  # (N, num_agents, 2)
            
            # Extract more robust features
            for i in range(N):
                sample_pos = xy_positions[i]  # (num_agents, 2)
                sample_features = extract_robust_spatial_features(sample_pos)
                
                if i == 0:
                    for fname in sample_features.keys():
                        features[fname] = []
                
                for fname, value in sample_features.items():
                    features[fname].append(value)
            
            # Add temporal features if we have multiple timesteps
            if seq_len > 1:
                temporal_features = extract_robust_temporal_features(positions)
                for fname, values in temporal_features.items():
                    features[fname] = values
        
        else:
            # If not spatial coordinates, use statistical features
            print(f"Non-spatial data detected, using statistical features")
            features.update(extract_statistical_features(positions))
    
    elif len(positions.shape) == 3:  # (N, num_agents, feature_dim)
        N, num_agents, feature_dim = positions.shape
        
        if feature_dim >= 2:
            xy_positions = positions[:, :, :2]
            
            for i in range(N):
                sample_pos = xy_positions[i]
                sample_features = extract_robust_spatial_features(sample_pos)
                
                if i == 0:
                    for fname in sample_features.keys():
                        features[fname] = []
                
                for fname, value in sample_features.items():
                    features[fname].append(value)
        else:
            features.update(extract_statistical_features(positions))
    
    else:
        print(f"Unexpected shape {positions.shape}, using basic features")
        features.update(extract_statistical_features(positions))
    
    # Convert lists to numpy arrays
    for fname in features:
        if isinstance(features[fname], list):
            features[fname] = np.array(features[fname])
    
    # Remove features with no variation
    features_to_remove = []
    for fname, values in features.items():
        if np.std(values) < 1e-10:
            features_to_remove.append(fname)
    
    for fname in features_to_remove:
        print(f"Removing feature '{fname}' (no variation)")
        del features[fname]
    
    # print(f"Final features: {list(features.keys())}")
    return features


def extract_robust_spatial_features(positions): #used
    """Extract robust spatial features that are more likely to show meaningful differences"""
    features = {}
    
    if len(positions.shape) != 2 or positions.shape[1] < 2:
        return {'invalid': 0}
    
    num_agents = positions.shape[0]
    
    # Normalize positions to handle different coordinate systems
    pos_normalized = positions - np.mean(positions, axis=0)
    
    # Basic spatial statistics (normalized)
    features['formation_center_x'] = np.mean(positions[:, 0])
    features['formation_center_y'] = np.mean(positions[:, 1])
    
    # Formation shape descriptors
    features['formation_width'] = np.max(positions[:, 0]) - np.min(positions[:, 0])
    features['formation_height'] = np.max(positions[:, 1]) - np.min(positions[:, 1])
    features['formation_area'] = features['formation_width'] * features['formation_height']
    
    # Compactness measures
    centroid = np.mean(positions, axis=0)
    distances_to_center = np.linalg.norm(positions - centroid, axis=1)
    features['avg_distance_to_center'] = np.mean(distances_to_center)
    features['formation_compactness'] = np.std(distances_to_center)
    features['formation_radius'] = np.max(distances_to_center)
    
    # Formation density in different regions
    # Divide field into quadrants
    x_mid = np.median(positions[:, 0])
    y_mid = np.median(positions[:, 1])
    
    q1 = np.sum((positions[:, 0] < x_mid) & (positions[:, 1] < y_mid))
    q2 = np.sum((positions[:, 0] >= x_mid) & (positions[:, 1] < y_mid))
    q3 = np.sum((positions[:, 0] < x_mid) & (positions[:, 1] >= y_mid))
    q4 = np.sum((positions[:, 0] >= x_mid) & (positions[:, 1] >= y_mid))
    
    total = q1 + q2 + q3 + q4
    if total > 0:
        features['density_q1'] = q1 / total
        features['density_q2'] = q2 / total
        features['density_q3'] = q3 / total
        features['density_q4'] = q4 / total
        
        # Balance measures
        features['horizontal_balance'] = abs((q1 + q3) - (q2 + q4)) / total
        features['vertical_balance'] = abs((q1 + q2) - (q3 + q4)) / total
    
    # Nearest neighbor distances (local density)
    # if num_agents > 1:
    #     nn_distances = []
    #     for i in range(num_agents):
    #         distances = [np.linalg.norm(positions[i] - positions[j]) 
    #                     for j in range(num_agents) if i != j]
    #         nn_distances.append(min(distances))
        
    #     features['avg_nearest_neighbor'] = np.mean(nn_distances)
    #     features['min_nearest_neighbor'] = np.min(nn_distances)
    #     features['nearest_neighbor_std'] = np.std(nn_distances)
    
    # Convex hull area (formation spread)
    try:
        if num_agents >= 3:
            hull = ConvexHull(positions)
            features['convex_hull_area'] = hull.volume  # 2D volume = area
            features['hull_to_bbox_ratio'] = hull.volume / features['formation_area'] if features['formation_area'] > 0 else 0
    except:
        pass  # Skip if scipy not available or insufficient points
    
    return features


def extract_robust_temporal_features(positions): #used
    """Extract temporal features that capture movement patterns"""
    features = {}
    N, seq_len, num_agents, feature_dim = positions.shape
    
    if seq_len < 2:
        return features
    
    # Use first and last few timesteps for stability
    start_window = positions[:, :min(5, seq_len//4), :, :2]  # First 5 or 25% of sequence
    end_window = positions[:, -min(5, seq_len//4):, :, :2]   # Last 5 or 25% of sequence
    
    start_pos = np.mean(start_window, axis=1)  # Average over time window
    end_pos = np.mean(end_window, axis=1)
    
    # Movement vectors
    movement = end_pos - start_pos  # (N, num_agents, 2)
    
    # Aggregate movement features
    movement_distances = np.linalg.norm(movement, axis=2)  # (N, num_agents)
    features['total_movement'] = np.sum(movement_distances, axis=1)
    features['avg_movement_per_agent'] = np.mean(movement_distances, axis=1)
    features['movement_variation'] = np.std(movement_distances, axis=1)
    
    # Formation centroid movement
    start_centroids = np.mean(start_pos, axis=1)  # (N, 2)
    end_centroids = np.mean(end_pos, axis=1)
    centroid_movement = np.linalg.norm(end_centroids - start_centroids, axis=1)
    features['formation_drift'] = centroid_movement
    
    # Formation shape changes
    start_spreads = [np.std(start_pos[i].flatten()) for i in range(N)]
    end_spreads = [np.std(end_pos[i].flatten()) for i in range(N)]
    features['formation_expansion'] = np.array(end_spreads) - np.array(start_spreads)
    
    # Coordination measures
    # How much do agents move in similar directions?
    coordination_scores = []
    for i in range(N):
        agent_movements = movement[i]  # (num_agents, 2)
        if len(agent_movements) > 1:
            # Calculate pairwise movement direction similarities
            similarities = []
            for j in range(len(agent_movements)):
                for k in range(j+1, len(agent_movements)):
                    v1, v2 = agent_movements[j], agent_movements[k]
                    if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        similarities.append(cos_sim)
            coordination_scores.append(np.mean(similarities) if similarities else 0)
        else:
            coordination_scores.append(0)
    
    features['movement_coordination'] = np.array(coordination_scores)
    
    return features


def extract_statistical_features(positions): #used
    """Extract basic statistical features when spatial interpretation isn't clear"""
    features = {}
    
    # Global statistics
    features['global_mean'] = np.mean(positions, axis=tuple(range(1, len(positions.shape))))
    features['global_std'] = np.std(positions, axis=tuple(range(1, len(positions.shape))))
    features['global_max'] = np.max(positions, axis=tuple(range(1, len(positions.shape))))
    features['global_min'] = np.min(positions, axis=tuple(range(1, len(positions.shape))))
    
    # Reshape for easier analysis
    N = positions.shape[0]
    flat_positions = positions.reshape(N, -1)
    
    features['sample_mean'] = np.mean(flat_positions, axis=1)
    features['sample_std'] = np.std(flat_positions, axis=1)
    features['sample_range'] = np.max(flat_positions, axis=1) - np.min(flat_positions, axis=1)
    features['sample_skewness'] = np.array([
        np.mean(((flat_positions[i] - np.mean(flat_positions[i])) / (np.std(flat_positions[i]) + 1e-8))**3)
        for i in range(N)
    ])
    
    return features


def plot_improved_cluster_comparison(features, labels, top_features, n_clusters, team_name="", feature_type="", folder: str = "analysis/visualisation"): #used
    """Plot improved comparison with better statistical visualization"""
    
    n_features = min(len(top_features), 6)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    # Set up color palette
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, (feature_name, feature_info) in enumerate(top_features[:n_features]):
        ax = axes[i]
        
        feature_values = features[feature_name]
        
        # Create box plots for each cluster
        cluster_data = []
        cluster_labels_for_plot = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = labels == cluster_id
            if np.any(cluster_mask):
                cluster_values = feature_values[cluster_mask]
                cluster_data.append(cluster_values)
                cluster_labels_for_plot.append(f'C{cluster_id}')
        
        # Box plot
        if cluster_data:
            bp = ax.boxplot(cluster_data, labels=cluster_labels_for_plot, patch_artist=True)
            
            # Color the boxes
            for patch, color in zip(bp['boxes'], colors[:len(cluster_data)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        
        ax.set_title(f'{feature_name}\n(Score: {feature_info["score"]:.3f})', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle(f'Top Discriminative Features - {team_name}- {feature_type}', fontsize=16)
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/topdiscriminative_features_team_{team_name}_{feature_type}.png')
    plt.show()


def prepare_cluster_visualization_data(features, labels, n_clusters): #used
    """Prepare data for cluster visualization with fixed t-SNE parameters"""
    
    vis_data = {}
    
    # Prepare feature matrix
    feature_names = list(features.keys())
    feature_matrix = np.column_stack([features[name] for name in feature_names])
    
    # Handle any NaN or infinite values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    vis_data['feature_matrix'] = feature_matrix_scaled
    vis_data['feature_names'] = feature_names
    vis_data['labels'] = labels
    
    # PCA for dimensionality reduction
    pca = PCA(n_components=min(2, feature_matrix_scaled.shape[1]))
    pca_result = pca.fit_transform(feature_matrix_scaled)
    vis_data['pca'] = pca_result
    vis_data['pca_variance_ratio'] = pca.explained_variance_ratio_
    
    # t-SNE with fixed parameters (removing problematic max_iter)
    if feature_matrix_scaled.shape[0] > 1 and feature_matrix_scaled.shape[1] > 0:
        try:
            # Use only basic parameters that are universally supported
            tsne = TSNE(
                n_components=2, 
                random_state=42, 
                perplexity=min(30, max(5, feature_matrix_scaled.shape[0] // 4))
            )
            tsne_result = tsne.fit_transform(feature_matrix_scaled)
            vis_data['tsne'] = tsne_result
        except Exception as e:
            print(f"t-SNE failed: {e}")
            # Fallback to PCA if t-SNE fails
            vis_data['tsne'] = pca_result
            print("Using PCA as fallback for t-SNE")
    else:
        vis_data['tsne'] = pca_result
    
    return vis_data

def create_comprehensive_cluster_report(analysis_results, save_path=None): #here mght eep
    """Create a comprehensive report of cluster analysis results"""
    if isinstance(analysis_results, dict) and 'full' in analysis_results:
        # Multi-team analysis
        print("="*100)
        print("COMPREHENSIVE CLUSTER ANALYSIS REPORT")
        print("="*100)
        
        for team_key, team_analysis in analysis_results.items():
            if team_analysis is None:
                continue
                
            print(f"\n{'='*80}")
            print(f"TEAM: {team_analysis['team_info']['name']}")
            print(f"{'='*80}")
            
            _print_team_analysis(team_analysis)
            
        # Cross-team comparison
        _compare_teams(analysis_results)
        
    else:
        # Single team analysis
        print("="*80)
        print("CLUSTER ANALYSIS REPORT")
        print("="*80)
        _print_team_analysis(analysis_results)


def _print_team_analysis(analysis): #here mght eep
    """Print analysis for a single team"""
    # Team info
    team_info = analysis['team_info']
    print(f"Team: {team_info['name']}")
    print(f"Players: {team_info['n_players']}")
    if team_info['player_indices']:
        print(f"Player indices: {team_info['player_indices']}")
    
    # Cluster summary
    print(f"\nCLUSTER SUMMARY:")
    print("-" * 50)
    
    for cluster_id, cluster_data in analysis['cluster_summary'].items():
        print(f"\nCluster {cluster_id}:")
        print(f"  Size: {cluster_data['size']} samples ({cluster_data['percentage']:.1f}%)")
        print(f"  Key characteristics:")
        
        if cluster_data['characteristic_patterns']:
            for pattern in cluster_data['characteristic_patterns'][:5]:
                print(f"    • {pattern}")
        else:
            print("    • No significant distinguishing patterns found")
    
    # Top discriminative features
    print(f"\nTOP DISCRIMINATIVE FEATURES:")
    print("-" * 50)
    
    discriminative_features = analysis['discriminative_features']
    for i, (feature_name, scores) in enumerate(list(discriminative_features.items())[:10]):
        print(f"{i+1:2d}. {feature_name:25s} | Score: {scores['score']:6.3f} | "
              f"Significance: {scores['significance_ratio']:4.2f}")


def _compare_teams(analysis_results): #here mght eep
    """Compare analysis results across teams"""
    print(f"\n{'='*80}")
    print("CROSS-TEAM COMPARISON")
    print("="*80)
    
    team_keys = [k for k in analysis_results.keys() if k.startswith('team_')]
    
    if len(team_keys) < 2:
        print("Not enough teams for comparison")
        return
    
    # Compare top discriminative features
    print("\nTOP DISCRIMINATIVE FEATURES BY TEAM:")
    print("-" * 60)
    
    for team_key in team_keys:
        team_analysis = analysis_results[team_key]
        if team_analysis is None:
            continue
            
        team_name = team_analysis['team_info']['name']
        top_features = list(team_analysis['discriminative_features'].items())[:5]
        
        print(f"\n{team_name}:")
        for i, (feature_name, scores) in enumerate(top_features):
            print(f"  {i+1}. {feature_name} (score: {scores['score']:.3f})")
    
    # Find common discriminative features
    common_features = None
    for team_key in team_keys:
        team_analysis = analysis_results[team_key]
        if team_analysis is None:
            continue
            
        team_features = set(team_analysis['discriminative_features'].keys())
        if common_features is None:
            common_features = team_features
        else:
            common_features = common_features.intersection(team_features)
    
    if common_features:
        print(f"\nCOMMON DISCRIMINATIVE FEATURES:")
        print("-" * 40)
        for feature in sorted(common_features):
            print(f"  • {feature}")
    else:
        print(f"\nNo common discriminative features found across all teams")
        
        
def extract_transformer_relevant_features(raw_positions): #used
    """
    Extract features that transformers typically focus on - higher-level patterns
    that aren't captured by basic positional statistics.
    """
    
    features = {}
    
    # Convert to numpy if needed
    if hasattr(raw_positions, 'numpy'):
        positions = raw_positions.numpy()
    else:
        positions = np.array(raw_positions)
    
    print(f"(Not from class) Extracting transformer-relevant features from shape: {positions.shape}")
    
    if len(positions.shape) == 4:  # (N, seq_len, num_agents, feature_dim)
        N, seq_len, num_agents, feature_dim = positions.shape
        
        # Use last timestep for current analysis
        current_positions = positions[:, -1, :, :2]  # (N, num_agents, 2)
        
        for i in range(N):
            sample_pos = current_positions[i]  # (num_agents, 2)
            sample_features = extract_tactical_patterns(sample_pos)
            
            if i == 0:
                for fname in sample_features.keys():
                    features[fname] = []
            
            for fname, value in sample_features.items():
                features[fname].append(value)
        
        # Add sequence-level features if multiple timesteps
        if seq_len > 1:
            temporal_features = extract_sequence_patterns(positions)
            for fname, values in temporal_features.items():
                features[fname] = values
    
    elif len(positions.shape) == 3:  # (N, num_agents, feature_dim)
        N, num_agents, feature_dim = positions.shape
        current_positions = positions[:, :, :2]
        
        for i in range(N):
            sample_pos = current_positions[i]
            sample_features = extract_tactical_patterns(sample_pos)
            
            if i == 0:
                for fname in sample_features.keys():
                    features[fname] = []
            
            for fname, value in sample_features.items():
                features[fname].append(value)
    
    # Convert to numpy arrays
    for fname in features:
        if isinstance(features[fname], list):
            features[fname] = np.array(features[fname])
    
    return features


def extract_tactical_patterns(positions): #used
    """
    Extract tactical patterns that transformers are likely to learn:
    - Formation shapes and structures
    - Player relationships and distances
    - Spatial coverage and positioning
    """
    features = {}
    num_agents = len(positions)
    
    if num_agents < 3:
        return {'insufficient_agents': 0}
    
    # 1. FORMATION STRUCTURE ANALYSIS
    # Distances between all players (formation compactness patterns)
    dist_matrix = squareform(pdist(positions))
    
    # Formation compactness metrics
    features['formation_compactness_mean'] = np.mean(dist_matrix[dist_matrix > 0])
    features['formation_compactness_std'] = np.std(dist_matrix[dist_matrix > 0])
    features['max_player_distance'] = np.max(dist_matrix)
    features['min_player_distance'] = np.min(dist_matrix[dist_matrix > 0])
    
    # 2. SPATIAL DISTRIBUTION PATTERNS
    # Principal component analysis of formation
    try:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(positions)
        features['formation_pc1_variance'] = pca.explained_variance_ratio_[0]
        features['formation_pc2_variance'] = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
        features['formation_elongation'] = pca.explained_variance_ratio_[0] / (pca.explained_variance_ratio_[1] + 1e-8)
    except:
        features['formation_pc1_variance'] = 0
        features['formation_pc2_variance'] = 0
        features['formation_elongation'] = 1
    
    # 3. POSITIONAL ROLE PATTERNS
    # Identify potential lines/groups based on x/y coordinates
    x_positions = positions[:, 0]
    y_positions = positions[:, 1]
    
    # X-coordinate clustering (defensive/midfield/attacking lines)
    x_sorted_indices = np.argsort(x_positions)
    x_sorted = x_positions[x_sorted_indices]
    
    # Find gaps in x-positions (potential lines)
    x_gaps = np.diff(x_sorted)
    large_x_gaps = x_gaps > (np.std(x_gaps) + np.mean(x_gaps))
    features['num_x_lines'] = np.sum(large_x_gaps) + 1
    features['max_x_gap'] = np.max(x_gaps) if len(x_gaps) > 0 else 0
    
    # Same for y-coordinates (left/center/right)
    y_sorted = np.sort(y_positions)
    y_gaps = np.diff(y_sorted)
    large_y_gaps = y_gaps > (np.std(y_gaps) + np.mean(y_gaps))
    features['num_y_lines'] = np.sum(large_y_gaps) + 1
    features['max_y_gap'] = np.max(y_gaps) if len(y_gaps) > 0 else 0
    
    # 4. GEOMETRIC PATTERNS
    # Triangle formations, clusters, etc.
    centroid = np.mean(positions, axis=0)
    
    # Identify players in different zones relative to centroid
    distances_from_centroid = np.linalg.norm(positions - centroid, axis=1)
    close_to_center = np.sum(distances_from_centroid < np.percentile(distances_from_centroid, 33))
    far_from_center = np.sum(distances_from_centroid > np.percentile(distances_from_centroid, 67))
    
    features['players_near_center'] = close_to_center / num_agents
    features['players_far_center'] = far_from_center / num_agents
    
    # 5. SYMMETRY AND BALANCE
    # Check formation symmetry around center line
    y_center = np.mean(y_positions)
    left_players = np.sum(y_positions < y_center)
    right_players = np.sum(y_positions > y_center)
    features['left_right_balance'] = abs(left_players - right_players) / num_agents
    
    # Forward-backward balance
    x_center = np.mean(x_positions)
    forward_players = np.sum(x_positions > x_center)
    backward_players = np.sum(x_positions < x_center)
    features['forward_backward_balance'] = abs(forward_players - backward_players) / num_agents
    
    # 6. NEAREST NEIGHBOR PATTERNS
    # Local density around each player
    nn_distances = []
    for i in range(num_agents):
        distances = dist_matrix[i]
        distances = distances[distances > 0]  # Remove self-distance
        if len(distances) > 0:
            nn_distances.append(np.min(distances))
    
    if nn_distances:
        features['min_nearest_neighbor'] = np.min(nn_distances)
        features['avg_nearest_neighbor'] = np.mean(nn_distances) 
        features['std_nearest_neighbor'] = np.std(nn_distances)
        features['nearest_neighbor_variance'] = np.var(nn_distances)
        features['isolated_players'] = np.sum(np.array(nn_distances) > np.mean(nn_distances) + np.std(nn_distances))
    
    return features

# SEQUENCE DATA
# UPDATED
def extract_sequence_patterns(positions): #used
    """
    Extract comprehensive temporal patterns that transformers learn from sequences
    
    Args:
        positions: (N, seq_len, num_agents, feature_dim) array
    
    Returns:
        dict: Comprehensive feature dictionary with temporal patterns
    """
    features = {}
    N, seq_len, num_agents, feature_dim = positions.shape
    
    # === MOVEMENT DYNAMICS ===
    movement_vectors = positions[:, 1:, :, :2] - positions[:, :-1, :, :2]  # (N, seq_len-1, num_agents, 2)
    
    # Enhanced movement coherence with temporal windows
    features.update(_extract_movement_coherence(movement_vectors))
    
    # Velocity and acceleration patterns
    features.update(_extract_velocity_patterns(movement_vectors))
    
    # === FORMATION DYNAMICS ===
    features.update(_extract_formation_patterns(positions))
    
    # === INTERACTION PATTERNS ===
    features.update(_extract_interaction_patterns(positions))
    
    # === TEMPORAL SEQUENCE PATTERNS ===
    features.update(_extract_temporal_patterns(positions, movement_vectors))
    
    # === ATTENTION-LIKE PATTERNS ===
    features.update(_extract_attention_patterns(positions))
    
    # === PREDICTIVE PATTERNS ===
    features.update(_extract_predictive_patterns(positions, movement_vectors))
    
    return features


def compare_embeddings_vs_raw_clustering(clustering_results, metadata, layer='layer_-1', feature_type='combined',all_features=None, extract_per_team:bool=True): #used
    """
    Compare clustering results between embeddings and raw positional data
    to understand what the transformer captures that raw features don't
    """
    print("=== EMBEDDING vs RAW CLUSTERING COMPARISON ===")
    
    # Get embedding clusters
    embedding_labels = clustering_results[feature_type]['clustering']['kmeans']['labels']
    n_clusters = len(np.unique(embedding_labels[embedding_labels >= 0]))
    
    # Get raw positions and extract features
    raw_positions = metadata[layer]['player_states'] if isinstance(metadata[layer], dict) else metadata[layer]

    if extract_per_team:
        print("Extracting per team for raw_positions")

        # Split into team 0 (players 0-10) (no ball (index 22))
        team_0_indices = list(range(11)) # [0,1,...,10]
        team_0 = raw_positions[:, :, team_0_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Split into team 1 (players 11-21) + ball (index 22)
        team_1_indices = list(range(11, 22))  # [11,...,21,22]
        team_1 = raw_positions[:, :, team_1_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Concatenate along batch dimension
        raw_positions = np.concatenate([team_0, team_1], axis=0)  # Shape: [2 * batch, 11, 2 or 3]

        print(f"Raw positions reshaped to: {raw_positions.shape}")

    if all_features == None:
        # Extract basic positional features
        all_features = get_features(raw_positions)

    all_feature_names = list(all_features.keys())
    
    # Create feature matrix
    feature_matrix = np.column_stack([all_features[name] for name in all_feature_names])
    feature_matrix = np.nan_to_num(feature_matrix)
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Cluster raw features with same number of clusters
    raw_kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    raw_labels = raw_kmeans.fit_predict(feature_matrix_scaled)
    
    # Compare clustering results
    ari_score = adjusted_rand_score(embedding_labels, raw_labels)
    
    # Calculate silhouette scores
    embedding_silhouette = silhouette_score(feature_matrix_scaled, embedding_labels)
    raw_silhouette = silhouette_score(feature_matrix_scaled, raw_labels)
    
    print(f"Clustering Comparison Results:")
    print(f"  Adjusted Rand Index: {ari_score:.3f}")
    print(f"  Embedding clustering silhouette: {embedding_silhouette:.3f}")
    print(f"  Raw feature clustering silhouette: {raw_silhouette:.3f}")
    
    if ari_score < 0.3:
        print("\nLOW AGREEMENT: Embeddings capture patterns not obvious in raw features.")
    elif ari_score > 0.7:
        print("\nHIGH AGREEMENT: Embeddings mostly reflect raw positional patterns.")
    else:
        print("\nMODERATE AGREEMENT: Embeddings capture some additional patterns.")
    
    # Analyze which features are most important for each clustering
    analyze_feature_importance_difference(feature_matrix_scaled, all_feature_names, 
                                        embedding_labels, raw_labels)
    
    return all_features, {
        'ari_score': ari_score,
        'embedding_silhouette': embedding_silhouette,
        'raw_silhouette': raw_silhouette,
        'embedding_labels': embedding_labels,
        'raw_labels': raw_labels,
        'feature_matrix': feature_matrix_scaled,
        'feature_names': all_feature_names
    }

def _extract_movement_coherence(movement_vectors): #used
    """Extract various movement coherence metrics"""
    N, seq_len_minus1, num_agents, _ = movement_vectors.shape
    features = {}
    
    # Multi-scale coherence (different time windows)
    window_sizes = [1, 3, 5, min(10, seq_len_minus1)]
    for window in window_sizes:
        coherence_scores = []
        for i in range(N):
            sample_coherence = []
            for start in range(0, seq_len_minus1 - window + 1, window):
                end = start + window
                window_movements = movement_vectors[i, start:end]  # (window, num_agents, 2)
                
                # Calculate mean movement direction in window
                mean_movements = np.mean(window_movements, axis=0)  # (num_agents, 2)
                
                # Pairwise coherence
                similarities = []
                for j in range(num_agents):
                    for k in range(j+1, num_agents):
                        v1, v2 = mean_movements[j], mean_movements[k]
                        if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                            cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            similarities.append(cos_sim)
                
                if similarities:
                    sample_coherence.append(np.mean(similarities))
            
            coherence_scores.append(np.mean(sample_coherence) if sample_coherence else 0)
        
        features[f'movement_coherence_w{window}'] = np.array(coherence_scores)
    
    # Movement synchronization entropy
    sync_entropy = []
    for i in range(N):
        # Discretize movement directions into bins
        movements = movement_vectors[i].reshape(-1, 2)  # (seq_len-1 * num_agents, 2)
        angles = np.arctan2(movements[:, 1], movements[:, 0])
        # Bin angles into 8 directions
        angle_bins = np.digitize(angles, np.linspace(-np.pi, np.pi, 9))
        # Calculate entropy
        unique, counts = np.unique(angle_bins, return_counts=True)
        probs = counts / len(angle_bins)
        sync_entropy.append(entropy(probs))
    
    features['movement_sync_entropy'] = np.array(sync_entropy)
    
    return features

def _extract_velocity_patterns(movement_vectors): #used
    """Extract velocity and acceleration patterns"""
    N, seq_len_minus1, num_agents, _ = movement_vectors.shape
    features = {}
    
    # Velocity statistics
    velocities = np.linalg.norm(movement_vectors, axis=3)  # (N, seq_len-1, num_agents)
    
    features['velocity_mean'] = np.mean(velocities, axis=(1, 2))
    features['velocity_std'] = np.std(velocities, axis=(1, 2))
    features['velocity_max'] = np.max(velocities, axis=(1, 2))
    features['velocity_min'] = np.min(velocities, axis=(1, 2))
    
    # Acceleration patterns
    if seq_len_minus1 > 1:
        accelerations = movement_vectors[:, 1:] - movement_vectors[:, :-1]  # (N, seq_len-2, num_agents, 2)
        accel_magnitudes = np.linalg.norm(accelerations, axis=3)  # (N, seq_len-2, num_agents)
        
        features['acceleration_mean'] = np.mean(accel_magnitudes, axis=(1, 2))
        features['acceleration_std'] = np.std(accel_magnitudes, axis=(1, 2))
        
        # Jerk (rate of change of acceleration)
        if accelerations.shape[1] > 1:
            jerk = accelerations[:, 1:] - accelerations[:, :-1]
            jerk_magnitudes = np.linalg.norm(jerk, axis=3)
            features['jerk_mean'] = np.mean(jerk_magnitudes, axis=(1, 2))
    
    # Velocity persistence (how much do agents maintain their velocity?)
    velocity_persistence = []
    for i in range(N):
        persistences = []
        for j in range(num_agents):
            agent_velocities = movement_vectors[i, :, j]  # (seq_len-1, 2)
            if len(agent_velocities) > 1:
                # Calculate velocity change magnitudes
                vel_changes = np.linalg.norm(agent_velocities[1:] - agent_velocities[:-1], axis=1)
                # Persistence is inverse of velocity change
                persistence = 1 / (1 + np.mean(vel_changes))
                persistences.append(persistence)
        velocity_persistence.append(np.mean(persistences) if persistences else 0)
    
    features['velocity_persistence'] = np.array(velocity_persistence)
    
    return features

def _extract_formation_patterns(positions): #used
    """Extract formation and spatial patterns"""
    N, seq_len, num_agents, _ = positions.shape
    features = {}
    
    # Formation centroid dynamics
    centroids = np.mean(positions[:, :, :, :2], axis=2)  # (N, seq_len, 2)
    centroid_velocities = centroids[:, 1:] - centroids[:, :-1]  # (N, seq_len-1, 2)
    
    features['centroid_velocity_mean'] = np.mean(np.linalg.norm(centroid_velocities, axis=2), axis=1)
    features['centroid_velocity_std'] = np.std(np.linalg.norm(centroid_velocities, axis=2), axis=1)
    
    # Formation spread dynamics
    formation_spreads = []
    spread_changes = []
    
    for i in range(N):
        spreads = []
        for t in range(seq_len):
            # Calculate distances from centroid
            centroid = centroids[i, t]
            distances = np.linalg.norm(positions[i, t, :, :2] - centroid, axis=1)
            spreads.append(np.mean(distances))
        
        formation_spreads.append(np.mean(spreads))
        if len(spreads) > 1:
            spread_changes.append(np.std(np.diff(spreads)))
        else:
            spread_changes.append(0)
    
    features['formation_spread_mean'] = np.array(formation_spreads)
    features['formation_spread_variability'] = np.array(spread_changes)
    
    # Formation shape stability (using relative positions)
    shape_stability = []
    for i in range(N):
        stabilities = []
        for t in range(seq_len - 1):
            # Normalize positions relative to centroid
            pos_t = positions[i, t, :, :2] - centroids[i, t]
            pos_t1 = positions[i, t+1, :, :2] - centroids[i, t+1]
            
            # Calculate shape change
            shape_change = np.mean(np.linalg.norm(pos_t1 - pos_t, axis=1))
            stabilities.append(shape_change)
        
        shape_stability.append(np.mean(stabilities) if stabilities else 0)
    
    features['formation_shape_stability'] = np.array(shape_stability)
    
    return features

def _extract_interaction_patterns(positions): #used
    """Extract agent interaction patterns"""
    N, seq_len, num_agents, _ = positions.shape
    features = {}
    
    # Pairwise distance dynamics
    distance_stats = []
    distance_changes = []
    
    for i in range(N):
        all_distances = []
        all_distance_changes = []
        
        for t in range(seq_len):
            # Calculate pairwise distances
            pos_t = positions[i, t, :, :2]  # (num_agents, 2)
            distances = pdist(pos_t)  # pairwise distances
            all_distances.extend(distances)
            
            if t > 0:
                pos_prev = positions[i, t-1, :, :2]
                distances_prev = pdist(pos_prev)
                distance_change = np.abs(distances - distances_prev)
                all_distance_changes.extend(distance_change)
        
        distance_stats.append([np.mean(all_distances), np.std(all_distances)])
        if all_distance_changes:
            distance_changes.append(np.mean(all_distance_changes))
        else:
            distance_changes.append(0)
    
    distance_stats = np.array(distance_stats)
    features['pairwise_distance_mean'] = distance_stats[:, 0]
    features['pairwise_distance_std'] = distance_stats[:, 1]
    features['pairwise_distance_change'] = np.array(distance_changes)
    
    # Nearest neighbor patterns
    # nearest_neighbor_distances = []
    # for i in range(N):
    #     nn_distances = []
    #     for t in range(seq_len):
    #         pos_t = positions[i, t, :, :2]
    #         distances = squareform(pdist(pos_t))
    #         np.fill_diagonal(distances, np.inf)  # Remove self-distances
    #         nearest_distances = np.min(distances, axis=1)
    #         nn_distances.extend(nearest_distances)
        
    #     nearest_neighbor_distances.append(np.mean(nn_distances))
    
    # features['nearest_neighbor_distance'] = np.array(nearest_neighbor_distances)
    
    return features

def _extract_temporal_patterns(positions, movement_vectors): #used
    """Extract temporal sequence patterns"""
    N, seq_len, num_agents, _ = positions.shape
    features = {}
    
    # Temporal autocorrelation of positions
    position_autocorr = []
    for i in range(N):
        autocorrs = []
        for agent in range(num_agents):
            for dim in range(2):  # x, y dimensions
                pos_series = positions[i, :, agent, dim]
                if len(pos_series) > 1:
                    # Calculate lag-1 autocorrelation
                    autocorr = np.corrcoef(pos_series[:-1], pos_series[1:])[0, 1]
                    if not np.isnan(autocorr):
                        autocorrs.append(autocorr)
        
        position_autocorr.append(np.mean(autocorrs) if autocorrs else 0)
    
    features['position_autocorr'] = np.array(position_autocorr)
    
    # Movement direction persistence
    direction_persistence = []
    for i in range(N):
        persistences = []
        for agent in range(num_agents):
            movements = movement_vectors[i, :, agent]  # (seq_len-1, 2)
            if len(movements) > 1:
                # Calculate direction changes
                angles = np.arctan2(movements[:, 1], movements[:, 0])
                angle_diffs = np.abs(np.diff(angles))
                # Handle angle wrapping
                angle_diffs = np.minimum(angle_diffs, 2*np.pi - angle_diffs)
                persistence = 1 / (1 + np.mean(angle_diffs))
                persistences.append(persistence)
        
        direction_persistence.append(np.mean(persistences) if persistences else 0)
    
    features['direction_persistence'] = np.array(direction_persistence)
    
    # Temporal complexity (entropy of position sequences)
    temporal_complexity = []
    for i in range(N):
        complexities = []
        for agent in range(num_agents):
            for dim in range(2):
                pos_series = positions[i, :, agent, dim]
                # Discretize positions into bins
                if np.std(pos_series) > 1e-6:
                    pos_bins = np.digitize(pos_series, np.linspace(np.min(pos_series), np.max(pos_series), 10))
                    unique, counts = np.unique(pos_bins, return_counts=True)
                    probs = counts / len(pos_bins)
                    complexities.append(entropy(probs))
        
        temporal_complexity.append(np.mean(complexities) if complexities else 0)
    
    features['temporal_complexity'] = np.array(temporal_complexity)
    
    return features

def _extract_attention_patterns(positions): #used
    """Extract attention-like patterns (what agents might attend to)"""
    N, seq_len, num_agents, _ = positions.shape
    features = {}
    
    # Relative position patterns (like relative positional encoding)
    relative_position_variance = []
    for i in range(N):
        relative_vars = []
        for t in range(seq_len):
            pos_t = positions[i, t, :, :2]  # (num_agents, 2)
            # Calculate all pairwise relative positions
            relative_positions = []
            for j in range(num_agents):
                for k in range(num_agents):
                    if j != k:
                        rel_pos = pos_t[j] - pos_t[k]
                        relative_positions.append(rel_pos)
            
            if relative_positions:
                relative_positions = np.array(relative_positions)
                relative_vars.append(np.var(relative_positions.flatten()))
        
        relative_position_variance.append(np.mean(relative_vars) if relative_vars else 0)
    
    features['relative_position_variance'] = np.array(relative_position_variance)
    
    # Attention dispersion (how spread out are the agents' focus?)
    attention_dispersion = []
    for i in range(N):
        dispersions = []
        for t in range(seq_len):
            pos_t = positions[i, t, :, :2]
            # Calculate convex hull area as proxy for attention dispersion
            if num_agents >= 3:
                try:
                    hull = ConvexHull(pos_t)
                    dispersions.append(hull.volume)  # Area in 2D
                except:
                    # Fallback: use bounding box area
                    min_coords = np.min(pos_t, axis=0)
                    max_coords = np.max(pos_t, axis=0)
                    area = np.prod(max_coords - min_coords)
                    dispersions.append(area)
        
        attention_dispersion.append(np.mean(dispersions) if dispersions else 0)
    
    features['attention_dispersion'] = np.array(attention_dispersion)
    
    return features

def _extract_predictive_patterns(positions, movement_vectors): #used
    """Extract patterns useful for prediction"""
    N, seq_len, num_agents, _ = positions.shape
    features = {}
    
    # Trajectory curvature
    trajectory_curvature = []
    for i in range(N):
        curvatures = []
        for agent in range(num_agents):
            if movement_vectors.shape[1] > 1:
                velocities = movement_vectors[i, :, agent]  # (seq_len-1, 2)
                if len(velocities) > 1:
                    # Calculate curvature as change in direction
                    v1, v2 = velocities[:-1], velocities[1:]
                    # Normalize velocities
                    v1_norm = np.linalg.norm(v1, axis=1, keepdims=True)
                    v2_norm = np.linalg.norm(v2, axis=1, keepdims=True)
                    
                    # Avoid division by zero
                    valid_mask = (v1_norm.flatten() > 1e-6) & (v2_norm.flatten() > 1e-6)
                    if np.any(valid_mask):
                        v1_normalized = v1[valid_mask] / v1_norm[valid_mask]
                        v2_normalized = v2[valid_mask] / v2_norm[valid_mask]
                        
                        # Calculate angle between consecutive velocity vectors
                        dot_products = np.sum(v1_normalized * v2_normalized, axis=1)
                        dot_products = np.clip(dot_products, -1, 1)  # Numerical stability
                        angles = np.arccos(dot_products)
                        curvatures.extend(angles)
        
        trajectory_curvature.append(np.mean(curvatures) if curvatures else 0)
    
    features['trajectory_curvature'] = np.array(trajectory_curvature)
    
    # Predictability score (how consistent are the movement patterns?)
    predictability = []
    for i in range(N):
        pred_scores = []
        for agent in range(num_agents):
            movements = movement_vectors[i, :, agent]  # (seq_len-1, 2)
            if len(movements) > 2:
                # Calculate consistency of movement direction
                angles = np.arctan2(movements[:, 1], movements[:, 0])
                angle_variance = np.var(angles)
                # Convert to predictability score (lower variance = higher predictability)
                pred_score = 1 / (1 + angle_variance)
                pred_scores.append(pred_score)
        
        predictability.append(np.mean(pred_scores) if pred_scores else 0)
    
    features['movement_predictability'] = np.array(predictability)
    
    # Future-past correlation (how well do past movements predict future ones?)
    future_past_corr = []
    for i in range(N):
        correlations = []
        mid_point = seq_len // 2
        if mid_point > 1:
            past_positions = positions[i, :mid_point, :, :2]
            future_positions = positions[i, mid_point:, :, :2]
            
            # Calculate movement patterns in past and future
            past_movements = past_positions[1:] - past_positions[:-1]
            future_movements = future_positions[1:] - future_positions[:-1]
            
            # Compare movement characteristics
            past_speeds = np.linalg.norm(past_movements, axis=2).flatten()
            future_speeds = np.linalg.norm(future_movements, axis=2).flatten()
            
            min_len = min(len(past_speeds), len(future_speeds))
            if min_len > 1:
                correlation = np.corrcoef(past_speeds[:min_len], future_speeds[:min_len])[0, 1]
                if not np.isnan(correlation):
                    correlations.append(correlation)
        
        future_past_corr.append(np.mean(correlations) if correlations else 0)
    
    features['future_past_correlation'] = np.array(future_past_corr)
    
    return features
# END SEQUENCE DATA

def analyze_cluster_differences_manual(clustering_results, metadata, layer='layer_-1', 
                                     feature_type='combined', algorithm='kmeans', 
                                     team_ids=None, sample_indices=None, transformer_features=None,
                                      save_path="cluster_difference_analysis.txt",
                                      folder="analysis/interpretation",
                                      extract_per_team: bool =True): #used
    """
    Manual deep-dive analysis to understand what makes clusters different.
    This function helps you manually inspect specific samples from each cluster.
    """
    
    # Get clustering labels
    labels = clustering_results[feature_type]['clustering'][algorithm]['labels']
    unique_clusters = np.unique(labels[labels >= 0])
    
    report = []
    report.append(f"=== MANUAL CLUSTER DIFFERENCE ANALYSIS ===")
    report.append(f"Found {len(unique_clusters)} clusters for {feature_type}")

    # Get raw positions
    raw_positions = metadata[layer]['player_states'] if isinstance(metadata[layer], dict) else metadata[layer]
    
    if extract_per_team:
        print("Extracting per team for raw_positions")

        # Split into team 0 (players 0-10) (no ball (index 22))
        team_0_indices = list(range(11)) # [0,1,...,10]
        team_0 = raw_positions[:, :, team_0_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Split into team 1 (players 11-21) + ball (index 22)
        team_1_indices = list(range(11, 22))  # [11,...,21,22]
        team_1 = raw_positions[:, :, team_1_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Concatenate along batch dimension
        raw_positions = np.concatenate([team_0, team_1], axis=0)  # Shape: [2 * batch, 11, 2 or 3]

    if transformer_features == None:
        # Extract transformer-relevant features
        transformer_features = get_features(raw_positions) #here
    
    # For each cluster, show example samples and their characteristics
    cluster_examples = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        report.append(f"\n{'='*60}")
        report.append(f"CLUSTER {cluster_id} - {len(cluster_indices)} samples")
        report.append(f"{'='*60}")
        
        # Show 3-5 example samples from this cluster
        n_examples = min(3, len(cluster_indices))
        example_indices = np.random.choice(cluster_indices, n_examples, replace=False)
        
        cluster_examples[cluster_id] = {
            'indices': example_indices,
            'positions': [],
            'features': {}
        }
        
        print(f"Example samples: {example_indices}")
        
        # Analyze features for this cluster
        cluster_feature_stats = {}
        for feature_name, feature_values in transformer_features.items():
            cluster_values = feature_values[cluster_mask]
            cluster_feature_stats[feature_name] = {
                'mean': np.mean(cluster_values),
                'std': np.std(cluster_values),
                'min': np.min(cluster_values),
                'max': np.max(cluster_values)
            }
        
        # Show top distinguishing features for this cluster
        report.append(f"Top distinguishing features (vs other clusters):\n")
        feature_distinctiveness = []
        
        for feature_name, cluster_stats in cluster_feature_stats.items():
            other_clusters_values = transformer_features[feature_name][~cluster_mask]
            if len(other_clusters_values) > 0:
                other_mean = np.mean(other_clusters_values)
                other_std = np.std(other_clusters_values)
                
                if other_std > 1e-6:
                    z_score = (cluster_stats['mean'] - other_mean) / other_std
                    feature_distinctiveness.append((feature_name, abs(z_score), z_score))
        
        # Sort by distinctiveness
        feature_distinctiveness.sort(key=lambda x: x[1], reverse=True)
        
        for i, (fname, abs_z, z_score) in enumerate(feature_distinctiveness[:5]):
            direction = "higher" if z_score > 0 else "lower"
            #print(f"  {i+1}. {fname}: {direction} (z={z_score:.2f})")
            report.append(f"  {i+1}. {fname}: {direction} (z={z_score:.2f})\n")
            
            
        cluster_examples[cluster_id]['features'] = cluster_feature_stats
        
        # Store example positions for visualization
        if hasattr(raw_positions, 'numpy'):
            pos_array = raw_positions.numpy()
        else:
            pos_array = np.array(raw_positions)
            
        if len(pos_array.shape) == 4:  # (N, seq_len, num_agents, feature_dim)
            example_positions = pos_array[example_indices, -1, :, :2]  # Use last timestep
        else:
            example_positions = pos_array[example_indices, :, :2]
            
        cluster_examples[cluster_id]['positions'] = example_positions
    
    # Visualize example formations from each cluster
    # visualize_cluster_formations(cluster_examples, unique_clusters)
    # Save report
    os.makedirs(folder, exist_ok=True) 
    report_text = "\n".join(report)
    save_path = folder + "/" + save_path.split(".")[0] + "_" + feature_type + "." + save_path.split(".")[1]
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    return transformer_features, cluster_examples


def visualize_cluster_formations(cluster_examples, unique_clusters, folder: str = "analysis/visualisation"): #eep
    """
    Visualize example formations from each cluster to manually identify patterns
    """
    
    n_clusters = len(unique_clusters)
    fig, axes = plt.subplots(n_clusters, 3, figsize=(20, 4*n_clusters))
    
    if n_clusters == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_data = cluster_examples[cluster_id]
        example_positions = cluster_data['positions']
        
        for j, positions in enumerate(example_positions):
            ax = axes[i, j] if n_clusters > 1 else axes[j]
            
            # Plot player positions
            ax.scatter(positions[:, 0], positions[:, 1], 
                      c=[colors[i]], s=100, alpha=0.8)
            
            # Add player numbers
            for k, (x, y) in enumerate(positions):
                ax.annotate(str(k), (x, y), xytext=(3, 3), 
                           textcoords='offset points', fontsize=8)
            
            ax.set_title(f'Cluster {cluster_id} - Sample {j+1}')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Set reasonable axis limits
            ax.set_xlim(np.min(positions[:, 0])-5, np.max(positions[:, 0])+5)
            ax.set_ylim(np.min(positions[:, 1])-5, np.max(positions[:, 1])+5)
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/cluster_formation_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

def analyze_feature_importance_difference(feature_matrix, feature_names, 
                                        embedding_labels, raw_labels): #used
    """
    Analyze which features are most important for distinguishing clusters
    in embedding vs raw clustering
    """
    
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    # Train classifiers to predict cluster labels
    rf_embedding = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_raw = RandomForestClassifier(n_estimators=100, random_state=42)
    
    rf_embedding.fit(feature_matrix, embedding_labels)
    rf_raw.fit(feature_matrix, raw_labels)
    
    # Get feature importances
    embedding_importance = rf_embedding.feature_importances_
    raw_importance = rf_raw.feature_importances_
    
    # Find features that are much more important for embedding clustering
    importance_diff = embedding_importance - raw_importance
    
    # Sort features by importance difference
    importance_ranking = list(zip(feature_names, importance_diff, 
                                embedding_importance, raw_importance))
    importance_ranking.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nFeatures MORE important for embedding clustering:")
    print(f"{'Feature':<30} {'Diff':<8} {'Embed':<8} {'Raw':<8}")
    print("-" * 60)
    
    for fname, diff, embed_imp, raw_imp in importance_ranking[:10]:
        if diff > 0.01:  # Only show meaningful differences
            print(f"{fname:<30} {diff:>6.3f} {embed_imp:>6.3f} {raw_imp:>6.3f}")
    
    print(f"\nFeatures MORE important for raw feature clustering:")
    print(f"{'Feature':<30} {'Diff':<8} {'Embed':<8} {'Raw':<8}")
    print("-" * 60)
    
    for fname, diff, embed_imp, raw_imp in reversed(importance_ranking[-10:]):
        if diff < -0.01:  # Only show meaningful differences
            print(f"{fname:<30} {diff:>6.3f} {embed_imp:>6.3f} {raw_imp:>6.3f}")


# Update the main analysis function to use new features
def analyze_embedded_clusters_with_positional_data_enhanced(clustering_results, metadata, 
                                                          layer='layer_-1', feature_type='combined', 
                                                          algorithm='kmeans', team_ids=None): #used
    """
    Enhanced version that uses transformer-relevant features
    """
    # First run the comparison analysis
    # Cluster positional data in the same number of clusters as embeddings
    # and check if that amount of clusters make sense
    print("Step 1: Comparing embedding vs raw clustering...")
    all_features, comparison_results = compare_embeddings_vs_raw_clustering(clustering_results=clustering_results, 
                                                                            metadata=metadata, 
                                                                            layer=layer,
                                                                            feature_type=feature_type,
                                                                            all_features=None)
    
    # Then run manual inspection
    print("\nStep 2: Manual cluster inspection...")
    _, cluster_examples = analyze_cluster_differences_manual(clustering_results, 
                                                             metadata, 
                                                             layer, 
                                                             feature_type, 
                                                             algorithm, 
                                                             team_ids, 
                                                             transformer_features=all_features)
    
    # Finally run the original analysis with enhanced features
    print("\nStep 3: Running enhanced statistical analysis...")
    
    try:
        statistical_results = analyze_embedded_clusters_with_positional_data_new(
            clustering_results, metadata, layer, feature_type, algorithm, team_ids)
    except Exception as e:
        print(f"WARNING: {e}")
        

    return {
        'comparison': comparison_results,
        'manual_examples': cluster_examples,
        'statistical_analysis': statistical_results
    }
    
    
def compute_cluster_prototypes(clustering_results, metadata, layer='layer_-1', 
                             feature_type='combined', algorithm='kmeans', spatial_features=None, extract_per_team:bool=True): #used
    """
    Compute interpretable prototypes for each embedding-based cluster.
    
    This creates a "behavioral profile" for each cluster by averaging
    spatial features within each cluster.
    """
    print("=== CLUSTER PROTOTYPE ANALYSIS ===")
    
    # Get clustering labels
    labels = clustering_results[feature_type]['clustering'][algorithm]['labels']
    unique_clusters = np.unique(labels[labels >= 0])
    
    # Get raw positions and extract features
    raw_positions = metadata[layer]['player_states'] if isinstance(metadata[layer], dict) else metadata[layer]

    # need reshape
    if extract_per_team:
        print("Extracting per team for raw_positions")

        # Split into team 0 (players 0-10) (no ball (index 22))
        team_0_indices = list(range(11)) # [0,1,...,10]
        team_0 = raw_positions[:, :, team_0_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Split into team 1 (players 11-21) (no ball (index 22))
        team_1_indices = list(range(11, 22))  # [11,...,21,22]
        team_1 = raw_positions[:, :, team_1_indices, :]  # Shape: [batch, 11, 2 or 3]

        # Concatenate along batch dimension
        raw_positions = np.concatenate([team_0, team_1], axis=0)  # Shape: [2 * batch, 11, 2 or 3]

        print(f"Raw positions reshaped to: {raw_positions.shape}")
    
    if spatial_features == None:
        spatial_features = get_features(raw_positions)
    
    # Create feature matrix
    feature_names = list(spatial_features.keys())
    feature_matrix = np.column_stack([spatial_features[name] for name in feature_names])
    feature_matrix = np.nan_to_num(feature_matrix)
    
    # Compute prototypes
    prototypes = {}
    prototype_matrix = []
    
    for cluster_id in unique_clusters:
        cluster_mask = labels == cluster_id
        cluster_features = feature_matrix[cluster_mask]
        
        # Compute prototype (centroid)
        prototype = np.mean(cluster_features, axis=0)
        prototype_std = np.std(cluster_features, axis=0)
        
        prototypes[cluster_id] = {
            'mean': prototype,
            'std': prototype_std,
            'size': np.sum(cluster_mask),
            'feature_names': feature_names
        }
        
        prototype_matrix.append(prototype)
    
    prototype_matrix = np.array(prototype_matrix)
    
    # Create interpretable names for clusters based on their characteristics
    cluster_names = generate_cluster_names(prototypes, feature_names)
    
    # Visualize prototypes
    visualize_cluster_prototypes(prototypes, cluster_names, feature_names, feature_type)
    
    # Create comparison matrix
    comparison_df = create_prototype_comparison_table(prototypes, cluster_names, feature_names)
    
    # Analyze cluster relationships
    cluster_relationships = analyze_cluster_relationships(prototype_matrix, cluster_names, feature_type)
    
    results = {
        'prototypes': prototypes,
        'cluster_names': cluster_names,
        'comparison_table': comparison_df,
        'relationships': cluster_relationships,
        'prototype_matrix': prototype_matrix,
        'feature_names': feature_names
    }
    
    return results


def generate_cluster_names(prototypes, feature_names): #used
    """
    Generate interpretable names for clusters based on their most distinctive features
    """
    
    cluster_names = {}
    
    # Compute global feature means for comparison
    all_features = []
    for cluster_id, proto in prototypes.items():
        all_features.append(proto['mean'])
    
    global_means = np.mean(all_features, axis=0)
    global_stds = np.std(all_features, axis=0)
    
    for cluster_id, proto in prototypes.items():
        # Find most distinctive features (highest z-scores)
        z_scores = (proto['mean'] - global_means) / (global_stds + 1e-8)
        
        # Get top positive and negative features
        top_indices = np.argsort(np.abs(z_scores))[-3:][::-1]
        
        name_parts = []
        for idx in top_indices:
            if abs(z_scores[idx]) > 0.5:  # Only include meaningful differences
                feature_name = feature_names[idx]
                direction = "High" if z_scores[idx] > 0 else "Low"
                
                # Simplify feature names for readability
                simple_name = simplify_feature_name(feature_name)
                name_parts.append(f"{direction} {simple_name}")
        
        if name_parts:
            cluster_names[cluster_id] = " + ".join(name_parts[:2])  # Limit to 2 main characteristics
        else:
            cluster_names[cluster_id] = f"Cluster {cluster_id}"
    
    return cluster_names


def simplify_feature_name(feature_name): #used
    """
    Convert technical feature names to more readable versions
    """
    name_mapping = {
        'formation_compactness_mean': 'Compactness',
        'formation_elongation': 'Elongation',
        'num_x_lines': 'X-Lines',
        'num_y_lines': 'Y-Lines',
        'left_right_balance': 'L-R Balance',
        'forward_backward_balance': 'F-B Balance',
        'players_near_center': 'Central Players',
        'players_far_center': 'Wide Players',
        'avg_nearest_neighbor': 'Avg Spacing',
        'movement_coherence': 'Move Coherence',
        'movement_magnitude': 'Move Intensity',
        'formation_stability': 'Stability'
    }
    
    return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())


def visualize_cluster_prototypes(prototypes, cluster_names, feature_names, feature_type="", folder: str = "analysis/visualisation"): #used
    """
    Visualize cluster prototypes as radar charts and heatmaps
    """
    n_clusters = len(prototypes)
    n_features = len(feature_names)
    
    # Create prototype matrix for heatmap
    prototype_matrix = []
    cluster_ids = []
    
    for cluster_id, proto in prototypes.items():
        prototype_matrix.append(proto['mean'])
        cluster_ids.append(cluster_names[cluster_id])
    
    prototype_matrix = np.array(prototype_matrix)
    
    # Normalize for visualization
    scaler = StandardScaler()
    prototype_matrix_norm = scaler.fit_transform(prototype_matrix.T).T
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Heatmap
    ax = axes[0]
    sns.heatmap(prototype_matrix_norm, 
                xticklabels=[simplify_feature_name(name) for name in feature_names],
                yticklabels=cluster_ids,
                cmap='RdBu_r', center=0, annot=False, ax=ax)
    ax.set_title(f'{feature_type.title()} Cluster Prototypes (Standardized)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Feature importance across clusters
    ax = axes[1]
    feature_importance = np.std(prototype_matrix_norm, axis=0)
    sorted_indices = np.argsort(feature_importance)[::-1]
    
    ax.barh(range(len(feature_names)), feature_importance[sorted_indices])
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([simplify_feature_name(feature_names[i]) for i in sorted_indices])
    ax.set_xlabel('Standard Deviation Across Clusters')
    ax.set_title('Feature Discriminative Power')
    
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{feature_type}_cluster_prototypes.png', dpi=150, bbox_inches='tight')
    plt.show()

def create_prototype_comparison_table(prototypes, cluster_names, feature_names): #used
    """
    Create a comprehensive comparison table of cluster prototypes
    """
    # Create comparison matrix
    comparison_data = []
    
    for cluster_id, proto in prototypes.items():
        row = {
            'Cluster': cluster_names[cluster_id],
            'Size': proto['size'],
            'Size_Pct': f"{proto['size'] / sum(p['size'] for p in prototypes.values()) * 100:.1f}%"
        }
        
        # Add feature values
        for i, feature_name in enumerate(feature_names):
            simplified_name = simplify_feature_name(feature_name)
            row[simplified_name] = proto['mean'][i]
            row[f"{simplified_name}_Std"] = proto['std'][i]
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Round numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(3)
    
    return df


def analyze_cluster_relationships(prototype_matrix, cluster_names, feature_type="", folder: str = "analysis/visualisation"): #used
    """
    Analyze relationships and similarities between clusters
    """
    # Compute pairwise distances between prototypes
    distances = pdist(prototype_matrix, metric='euclidean')
    distance_matrix = squareform(distances)
    
    # Hierarchical clustering of prototypes
    linkage_matrix = linkage(distances, method='ward')
    
    # Create dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, labels=list(cluster_names.values()), 
               orientation='top', leaf_rotation=45)
    plt.title(f'{feature_type.title()} Cluster Relationship Dendrogram')
    plt.ylabel('Distance')
    plt.tight_layout()
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{feature_type}_cluster_relationships.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Find most similar and dissimilar clusters
    cluster_ids = list(cluster_names.keys())
    similarities = []
    
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            dist = distance_matrix[i, j]
            similarities.append((
                cluster_names[cluster_ids[i]], 
                cluster_names[cluster_ids[j]], 
                dist
            ))
    
    similarities.sort(key=lambda x: x[2])
    
    return {
        'distance_matrix': distance_matrix,
        'linkage_matrix': linkage_matrix,
        'most_similar': similarities[:3],
        'most_dissimilar': similarities[-3:],
        'all_similarities': similarities
    }


def advanced_cluster_interpretation(clustering_results, metadata, layer='layer_-1', feature_type='combined'): #used
    """
    Master function that runs all advanced clustering analyses
    """
    print("=" * 80)
    print("ADVANCED CLUSTER INTERPRETATION PIPELINE")
    print("=" * 80)
    
    results = {}
    
    # 1. Hybrid Clustering Analysis
    #print("\n1. Running Hybrid Clustering Analysis...")
    #try:
    #    hybrid_results = hybrid_clustering_analysis(clustering_results, metadata, layer)
    #    results['hybrid_clustering'] = hybrid_results
    #    print("✓ Hybrid clustering completed")
    #except Exception as e:
    #    print(f"✗ Hybrid clustering failed: {e}")
    #    results['hybrid_clustering'] = None
    
    # 2. Cluster Prototypes Analysis
    print("\n2. Computing Cluster Prototypes...")
    try:
        prototype_results = compute_cluster_prototypes(clustering_results=clustering_results, 
                                                       metadata=metadata, 
                                                       layer=layer, 
                                                       feature_type=feature_type)
        results['prototypes'] = prototype_results
        print("Prototype analysis completed")
        
        # Display comparison table
        # print("\nCluster Comparison Table:")
        # print(prototype_results['comparison_table'].to_string(index=False))
        
    except Exception as e:
        print(f"Prototype analysis failed: {e}")
        results['prototypes'] = None
    
    # 3. Behavioral Phase Detection
    # print("\n3. Detecting Behavioral Phases...")
    # not implemented yet
    results['behavioral_phases'] = None
    
    # 4. Generate Summary Report
    print("\n4. Generating Summary Report...")
    summary_report = generate_advanced_summary_report(results=results,
                                                      feature_type=feature_type)
    results['summary_report'] = summary_report
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


def generate_advanced_summary_report(results, feature_type='combined'): #used
    """
    Generate a comprehensive summary report of all analyses
    """
    report = []
    report.append("ADVANCED CLUSTERING ANALYSIS SUMMARY")
    report.append("=" * 50)
    
    # Hybrid Clustering Summary
    if results.get(feature_type):
        hybrid = results[feature_type]
        report.append(f"\n1. {feature_type.upper()} CLUSTERING")
        report.append(f"   - Optimal clusters: {hybrid['best_n_clusters']}")
        report.append(f"   - Best silhouette score: {hybrid['cluster_metrics'][hybrid['best_n_clusters']]['silhouette']:.3f}")
        report.append(f"   - Feature weights: {hybrid['feature_weights']}")
    
    # Prototype Analysis Summary
    if results.get('prototypes'):
        prototypes = results['prototypes']
        report.append(f"\n2. CLUSTER PROTOTYPES")
        report.append(f"   - Number of interpretable clusters: {len(prototypes['cluster_names'])}")
        report.append("   - Cluster characteristics:")
        for cluster_id, name in prototypes['cluster_names'].items():
            size = prototypes['prototypes'][cluster_id]['size']
            report.append(f"     * Cluster {cluster_id}: {name}: {size} samples")
    
    return "\n".join(report)


# Example usage function
def run_complete_advanced_analysis(clustering_results, metadata, layer='layer_-1', feature_type='combined', full_results=None,
                                            folder="analysis/interpretation",
                                              save_path="cluster_deep_dive.txt"): #used
    """
    Example function showing how to run the complete advanced analysis pipeline
    """
    print("Starting complete advanced clustering analysis...")
    
    # Run all analyses
    results = advanced_cluster_interpretation(clustering_results=clustering_results, 
                                              metadata=metadata, 
                                              layer=layer,
                                              feature_type=feature_type)

    report = []

    
    # Print summary report
    if 'summary_report' in results:
        print("\n" + results['summary_report'])
        report.append(results['summary_report'])
    
    # 2. Deep dive into most interesting clusters
    if results.get('prototypes'):
        print("\n" + "="*60)
        print("CLUSTER DEEP DIVE ANALYSIS")
        print("="*60)
        report.append("="*60)
        report.append("CLUSTER DEEP DIVE ANALYSIS")
        report.append("="*60)
        
        prototypes = results['prototypes']['prototypes']
        # Find clusters with most extreme characteristics
        most_compact = min(prototypes.keys(), 
                          key=lambda k: prototypes[k]['mean'][0] if len(prototypes[k]['mean']) > 0 else float('inf'))
        most_spread = max(prototypes.keys(), 
                         key=lambda k: prototypes[k]['mean'][0] if len(prototypes[k]['mean']) > 0 else float('-inf'))
        
        print(f"Most compact formation cluster: {most_compact}")
        print(f"Most spread formation cluster: {most_spread}")
        report.append(f"Most compact formation cluster: {most_compact}")
        report.append(f"Most spread formation cluster: {most_spread}")

    os.makedirs(folder, exist_ok=True) 
    report_text = "\n".join(report)
    save_path = folder + "/" + save_path.split(".")[0] + "_" + feature_type + "." + save_path.split(".")[1]
    with open(save_path, 'w') as f:
        f.write(report_text)
        
    print(f"Tactical report saved to {save_path}")

    if full_results==None:
        full_results = {}
    full_results[feature_type] = results
    
    return full_results

class TacticalExplainabilityLayer:
    """
    Explainability Layer for tactical pattern analysis
    Provides SHAP-based attribution, attention visualization, and embedding trajectory analysis
    """
    
    def __init__(self, model, analysis_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize explainability layer
        
        Args:
            model: Trained transformer model
            analysis_data: Dict containing embeddings, metadata, attention patterns
            device: Computing device
        """
        self.device = device
        self.model = model
        self.embeddings = analysis_data['embeddings']
        self.metadata = analysis_data['metadata']
        self.attention_patterns = analysis_data['attention_patterns']
        self.feature_importance = analysis_data['feature_importance']
        
        # Store computed explanations
        self.feature_attributions = {}
        self.attention_maps = {}
        self.embedding_trajectories = {}
        
    def compute_feature_attributions(self, cluster_labels: np.ndarray, 
                                   cluster_id: int, 
                                   layer: str = 'layer_-1',
                                   method: str = 'gradient_shap') -> Dict[str, Any]: #used
        """
        Compute feature attributions for a specific tactical cluster
        
        Args:
            cluster_labels: Cluster assignments for each sample
            cluster_id: Target cluster to explain
            layer: Which layer embeddings to use
            method: Attribution method ('gradient_shap', 'integrated_gradients', 'lime')
        """
        # Get samples belonging to this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = np.where(cluster_mask)[0]
        
        if len(cluster_samples) == 0:
            print(f"No samples found for cluster {cluster_id}")
            return {}
        
        print(f"Computing feature attributions for cluster {cluster_id} ({len(cluster_samples)} samples)")
        
        # Get cluster metadata - extract relevant metadata for cluster samples
        cluster_metadata = self._extract_cluster_metadata(cluster_samples, layer)
        cluster_embeddings = self.embeddings[layer][cluster_samples]
        
        # Compute attributions based on method
        if method == 'gradient_shap':
            #attributions = self._compute_gradient_shap(cluster_samples, cluster_embeddings, layer)
            pass
        elif method == 'integrated_gradients':
            attributions = self._compute_integrated_gradients(cluster_samples, cluster_embeddings, layer)
        elif method == 'lime':
            attributions = self._compute_lime_attributions(cluster_samples, cluster_embeddings, layer)
        else:
            raise ValueError(f"Unknown attribution method: {method}")
        
        # Analyze and interpret attributions
        attribution_analysis = self._analyze_feature_attributions(
            attributions, cluster_metadata, cluster_id
        )
        
        self.feature_attributions[cluster_id] = {
            'method': method,
            'attributions': attributions,
            'analysis': attribution_analysis,
            'sample_indices': cluster_samples
        }
        
        return attribution_analysis
    
    def _extract_cluster_metadata(self, cluster_samples: np.ndarray, layer: str) -> List[Dict]: #used
        """
        Extract metadata for cluster samples from the metadata structure
        """
        cluster_metadata = []
        
        if layer in self.metadata:
            layer_metadata = self.metadata[layer]
            
            # Extract relevant information for each sample in the cluster
            for i, sample_idx in enumerate(cluster_samples):
                sample_meta = {}
                
                # Extract available metadata fields for this sample
                for key, value in layer_metadata.items():
                    if isinstance(value, (np.ndarray, torch.Tensor)):
                        # Handle array-like data
                        if len(value.shape) > 0 and sample_idx < len(value):
                            sample_meta[key] = value[sample_idx]
                        else:
                            sample_meta[key] = value
                    else:
                        # Handle scalar or non-indexable data
                        sample_meta[key] = value
                
                cluster_metadata.append(sample_meta)
        
        return cluster_metadata
    
    def _compute_integrated_gradients(self, sample_indices: np.ndarray, 
                                    embeddings: torch.Tensor, 
                                    layer: str) -> Dict[str, np.ndarray]: #used
        """
        Compute integrated gradients for feature attribution
        """
        # Simplified implementation
        
        attributions = {}
        
        # Get baseline (zeros or mean)
        baseline = torch.zeros_like(embeddings[0])
        
        # Compute integrated gradients (simplified)
        sample_attributions = []
        
        for idx in range(min(10, len(embeddings))):
            sample_embedding = embeddings[idx]
            
            # Approximate integrated gradient
            steps = 50
            path_attributions = []
            
            for step in range(steps):
                alpha = step / steps
                interpolated = baseline + alpha * (sample_embedding - baseline)
                
                # Compute gradient at this point (simplified)
                gradient = interpolated - baseline
                path_attributions.append(gradient.cpu().numpy())
            
            # Average over path
            integrated_grad = np.mean(path_attributions, axis=0)
            sample_attributions.append(integrated_grad)
        
        if sample_attributions:
            attributions['player_importance'] = np.mean(sample_attributions, axis=0)
            
            # If we have multiple dimensions, average across embedding dimensions
            if len(attributions['player_importance'].shape) > 1:
                attributions['feature_importance'] = np.mean(attributions['player_importance'], axis=0)
            else:
                attributions['feature_importance'] = attributions['player_importance']
        
        return attributions
    
    def _analyze_feature_attributions(self, attributions: Dict[str, np.ndarray], 
                                    metadata: List[Dict], 
                                    cluster_id: int) -> Dict[str, Any]: #used
        """
        Analyze and interpret feature attributions
        """
        analysis = {}
        
        if 'player_importance' in attributions:
            player_importance = attributions['player_importance']
            
            # Handle different shapes of player_importance
            if len(player_importance.shape) == 1:
                # Single dimension - treat as overall importance
                avg_importance = player_importance
            else:
                # Multiple dimensions - average across features
                avg_importance = np.mean(player_importance, axis=-1)
            
            # Find most important players
            most_important_players = np.argsort(avg_importance)[-5:][::-1]
            
            analysis['most_important_players'] = {
                'indices': most_important_players.tolist(),
                'importance_scores': avg_importance[most_important_players].tolist()
            }
            
            # Analyze player roles
            player_roles = self._infer_player_roles(player_importance, most_important_players)
            analysis['player_roles'] = player_roles
        
        if 'feature_importance' in attributions:
            feature_importance = attributions['feature_importance']
            
            # Map to interpretable feature names
            feature_names = self._get_embedding_feature_names(len(feature_importance))
            top_features = np.argsort(feature_importance)[-10:][::-1]
            
            analysis['most_important_features'] = {
                'indices': top_features.tolist(),
                'names': [feature_names[i] if i < len(feature_names) else f"Feature_{i}" 
                         for i in top_features],
                'importance_scores': feature_importance[top_features].tolist()
            }
        
        # Temporal analysis if metadata available
        if metadata and len(metadata) > 0:
            temporal_analysis = self._analyze_temporal_attributions(attributions, metadata)
            analysis['temporal_patterns'] = temporal_analysis
        
        return analysis
    
    def _infer_player_roles(self, player_importance: np.ndarray, 
                          important_players: np.ndarray) -> Dict[str, Any]: #used
        """
        Infer tactical roles based on player importance patterns
        """
        roles = {}
        
        # Handle different shapes of player_importance
        if len(player_importance.shape) == 1:
            # Single dimension case
            for i, player_idx in enumerate(important_players):
                importance_score = player_importance[player_idx]
                
                if importance_score > np.mean(player_importance) * 1.5:
                    role = "Key Decision Maker"
                elif importance_score > np.mean(player_importance):
                    role = "Tactical Coordinator"
                else:
                    role = "System Player"
                
                roles[f"Player_{player_idx}"] = {
                    'role': role,
                    'importance_score': float(importance_score),
                    'importance_pattern': [float(importance_score)]
                }
        else:
            # Multi-dimensional case
            for i, player_idx in enumerate(important_players):
                importance_vector = player_importance[player_idx]
                
                # Simple role classification based on importance distribution
                if np.max(importance_vector) > np.mean(importance_vector) * 2:
                    role = "Key Decision Maker"
                elif np.std(importance_vector) > np.mean(importance_vector):
                    role = "Tactical Coordinator"
                else:
                    role = "System Player"
                
                roles[f"Player_{player_idx}"] = {
                    'role': role,
                    'importance_score': float(np.mean(importance_vector)),
                    'importance_pattern': importance_vector.tolist()
                }
        
        return roles
    
    def _get_embedding_feature_names(self, num_features: int) -> List[str]: #used
        """
        Get interpretable names for embedding features
        """
        return [f"Embedding_Dim_{i}" for i in range(num_features)]
    
    def _analyze_temporal_attributions(self, attributions: Dict[str, np.ndarray], 
                                     metadata: List[Dict]) -> Dict[str, Any]: #used
        """
        Analyze how attributions change over time
        """
        temporal_analysis = {}
        
        # This would analyze how feature importance changes across sequence
        # For now, return placeholder
        temporal_analysis['pattern'] = "Stable importance across time"
        temporal_analysis['key_moments'] = []
        
        return temporal_analysis

    def visualize_attention_patterns_with_average(self, cluster_labels: np.ndarray, 
                                cluster_id: int, 
                                layer: str = 'layer_-1',
                                head: int = 0,
                                max_samples: int = 5,
                                cluster_average: bool = False,
                                folder: str = "analysis/visualisation") -> None: #used
        """
        Visualize attention patterns for a specific cluster
        
        Args:
            cluster_labels: Array of cluster assignments
            cluster_id: ID of cluster to visualize
            layer: Layer name to extract attention from
            head: Attention head index
            max_samples: Maximum number of individual samples to show (when cluster_average=False)
            cluster_average: If True, show average attention across cluster; if False, show individual samples
        """
        if not self.attention_patterns:
            print("No attention patterns available")
            return
        
        # Get samples for this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_samples = np.where(cluster_mask)[0]
        
        if len(cluster_samples) == 0:
            print(f"No samples found for cluster {cluster_id}")
            return
        
        print(f"Visualizing attention patterns for cluster {cluster_id}")
        print(f"Total samples in cluster: {len(cluster_samples)}")
        
        if cluster_average:
            # Calculate average attention patterns across all samples in cluster
            print("Computing cluster average attention patterns...")
            
            attention_matrices = []
            for sample_idx in cluster_samples:
                attention_weights = self._get_attention_weights(sample_idx, layer, head)
                if attention_weights is not None:
                    attention_matrices.append(attention_weights)
            
            if not attention_matrices:
                print(f"No valid attention weights found for cluster {cluster_id}")
                return
            
            # Compute average attention matrix
            avg_attention = np.mean(attention_matrices, axis=0)
            
            # Create visualization for average
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Average player-to-player attention
            im1 = axes[0].imshow(avg_attention, cmap='Blues', aspect='auto')
            axes[0].set_title(f'Cluster {cluster_id} Average\nPlayer-Player Attention\n({len(attention_matrices)} samples)')
            axes[0].set_xlabel('Player Index')
            axes[0].set_ylabel('Player Index')
            plt.colorbar(im1, ax=axes[0])
            
            # Average attention distribution
            # Stack into 3D array: (num_samples, num_players, num_players)
            attention_array = np.stack(attention_matrices, axis=0)
            # Compute per-sample attention distributions
            attention_dists = attention_array.mean(axis=1)
            # Mean and std deviation of distributions
            avg_attention_dist = attention_dists.mean(axis=0)
            std_attention_dist = attention_dists.std(axis=0)

            # Bar plot with standard deviation error bars
            axes[1].bar(
                range(len(avg_attention_dist)),
                avg_attention_dist,
                yerr=std_attention_dist,
                capsize=4,
                color='skyblue',
                edgecolor='black'
            )
            #axes[1].bar(range(len(avg_attention_dist)), avg_attention_dist)
            axes[1].set_title('Average Attention Distribution')
            axes[1].set_xlabel('Player Index')
            axes[1].set_ylabel('Attention Weight')
            
            plt.tight_layout()
            os.makedirs(folder, exist_ok=True) 
            plt.savefig(f'{folder}/attention_patterns_cluster_{cluster_id}_average.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            # Print summary statistics
            print(f"Average attention matrix shape: {avg_attention.shape}")
            print(f"Average attention range: [{avg_attention.min():.4f}, {avg_attention.max():.4f}]")
            print(f"Standard deviation across positions: {np.std(avg_attention_dist):.4f}")
            
        else:
            # Show individual samples (original behavior)
            cluster_samples_subset = cluster_samples[:max_samples]
            print(f"Showing {len(cluster_samples_subset)} individual samples")
            
            # Create attention visualization
            fig, axes = plt.subplots(2, len(cluster_samples_subset), figsize=(4*len(cluster_samples_subset), 8))
            if len(cluster_samples_subset) == 1:
                axes = axes.reshape(2, 1)
            
            for i, sample_idx in enumerate(cluster_samples_subset):
                # Get attention weights for this sample
                attention_weights = self._get_attention_weights(sample_idx, layer, head)
                
                if attention_weights is not None:
                    # Player-to-player attention
                    im1 = axes[0, i].imshow(attention_weights, cmap='Blues', aspect='auto')
                    axes[0, i].set_title(f'Sample {sample_idx}\nPlayer-Player Attention')
                    axes[0, i].set_xlabel('Player Index')
                    axes[0, i].set_ylabel('Player Index')
                    plt.colorbar(im1, ax=axes[0, i])
                    
                    # Attention distribution
                    attention_dist = np.mean(attention_weights, axis=0)
                    axes[1, i].bar(range(len(attention_dist)), attention_dist)
                    axes[1, i].set_title('Average Attention Distribution')
                    axes[1, i].set_xlabel('Player Index')
                    axes[1, i].set_ylabel('Attention Weight')
            
            plt.tight_layout()
            os.makedirs(folder, exist_ok=True)
            plt.savefig(f'{folder}/attention_patterns_cluster_{cluster_id}_samples.png', dpi=150, bbox_inches='tight')
            plt.show()
    
    
    def _get_attention_weights(self, sample_idx: int, layer: str, head: int) -> Optional[np.ndarray]: #used
        """
        Extract attention weights for a specific sample
        """
        # This would depend on your model's attention storage format
        # For now, return mock attention weights
        if layer in self.embeddings:
            num_agents = self.embeddings[layer].shape[1]
        else:
            num_agents = 22  # Default for football
        
        # Mock attention pattern - in practice, extract from model
        attention = np.random.rand(num_agents, num_agents)
        attention = attention / np.sum(attention, axis=1, keepdims=True)  # Normalize
        
        return attention
    
    def analyze_embedding_trajectories(self, cluster_labels: np.ndarray, 
                                     method: str = 'pca',
                                     folder: str = "analysis/visualisation") -> Dict[str, Any]: #eep
        """
        Analyze how embeddings evolve across different layers
        """
        print(f"Analyzing embedding trajectories using {method}")
        
        trajectory_analysis = {}
        
        # Get embeddings from all layers
        layer_embeddings = {}
        for layer_name, embeddings in self.embeddings.items():
            # Flatten embeddings for trajectory analysis
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
            flattened = embeddings.reshape(embeddings.shape[0], -1)
            layer_embeddings[layer_name] = flattened
        
        # Reduce dimensionality for visualization
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cluster_labels)-1))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Analyze trajectories for each cluster
        unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])
        
        fig, axes = plt.subplots(1, len(unique_clusters), figsize=(5*len(unique_clusters), 5))
        if len(unique_clusters) == 1:
            axes = [axes]
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_mask = cluster_labels == cluster_id
            
            # Plot trajectory for this cluster
            colors = plt.cm.viridis(np.linspace(0, 1, len(layer_embeddings)))
            
            prev_centroid = None
            for j, (layer_name, embeddings) in enumerate(layer_embeddings.items()):
                cluster_embeddings = embeddings[cluster_mask]
                
                if len(cluster_embeddings) > 1:
                    # Reduce dimensionality
                    embedded = reducer.fit_transform(cluster_embeddings)
                    
                    # Plot
                    axes[i].scatter(embedded[:, 0], embedded[:, 1], 
                                  c=[colors[j]], label=layer_name, alpha=0.7)
                    
                    # Add trajectory lines (connect centroids)
                    centroid = np.mean(embedded, axis=0)
                    if prev_centroid is not None:
                        axes[i].plot([prev_centroid[0], centroid[0]], 
                                   [prev_centroid[1], centroid[1]], 
                                   'k--', alpha=0.5)
                    prev_centroid = centroid
            
            axes[i].set_title(f'Cluster {cluster_id} Trajectory')
            axes[i].set_xlabel(f'{method.upper()} 1')
            axes[i].set_ylabel(f'{method.upper()} 2')
            axes[i].legend()
        
        plt.tight_layout()
        os.makedirs(folder, exist_ok=True)
        plt.savefig(f'{folder}/embedding_trajectories_{method}.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Compute trajectory metrics
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            
            # Compute embedding evolution metrics
            evolution_metrics = self._compute_evolution_metrics(
                layer_embeddings, cluster_mask, cluster_id
            )
            trajectory_analysis[cluster_id] = evolution_metrics
        
        self.embedding_trajectories = trajectory_analysis
        return trajectory_analysis
    
    def _compute_evolution_metrics(self, layer_embeddings: Dict[str, np.ndarray], 
                                 cluster_mask: np.ndarray, 
                                 cluster_id: int) -> Dict[str, Any]: #eep
        """
        Compute metrics for how embeddings evolve across layers
        """
        metrics = {}
        
        layer_names = list(layer_embeddings.keys())
        centroids = []
        
        for layer_name in layer_names:
            cluster_embeddings = layer_embeddings[layer_name][cluster_mask]
            if len(cluster_embeddings) > 0:
                centroid = np.mean(cluster_embeddings, axis=0)
                centroids.append(centroid)
        
        if len(centroids) > 1:
            # Compute trajectory length
            trajectory_length = 0
            for i in range(1, len(centroids)):
                distance = np.linalg.norm(centroids[i] - centroids[i-1])
                trajectory_length += distance
            
            metrics['trajectory_length'] = float(trajectory_length)
            metrics['avg_step_size'] = float(trajectory_length / (len(centroids) - 1))
            
            # Compute trajectory smoothness
            if len(centroids) > 2:
                direction_changes = []
                for i in range(1, len(centroids) - 1):
                    v1 = centroids[i] - centroids[i-1]
                    v2 = centroids[i+1] - centroids[i]
                    
                    # Compute angle change
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle_change = np.arccos(np.clip(cos_angle, -1, 1))
                    direction_changes.append(angle_change)
                
                metrics['trajectory_smoothness'] = float(np.mean(direction_changes))
        
        return metrics
    
    def generate_explanation_report(self, cluster_labels: np.ndarray, 
                                  save_path: str = 'explanation_report.txt') -> str: #used
        """
        Generate comprehensive explanation report
        """
        report = []
        report.append("=== TACTICAL PATTERN EXPLAINABILITY REPORT ===\n")
        print(cluster_labels)
        # Feature attribution summary
        if self.feature_attributions:
            report.append("FEATURE ATTRIBUTION ANALYSIS:")
            for cluster_id, attribution_data in self.feature_attributions.items():
                analysis = attribution_data['analysis']
                report.append(f"\nCluster {cluster_id}:")
                
                if 'most_important_players' in analysis:
                    players = analysis['most_important_players']
                    report.append(f"  Key Players: {players['indices']}")
                    report.append(f"  Importance Scores: {[f'{s:.3f}' for s in players['importance_scores']]}")
                
                if 'player_roles' in analysis:
                    roles = analysis['player_roles']
                    report.append(f"  Player Roles:")
                    for player, role_data in roles.items():
                        report.append(f"    {player}: {role_data['role']} (score: {role_data['importance_score']:.3f})")
        
        # Embedding trajectory summary
        if self.embedding_trajectories:
            report.append("\n\nEMBEDDING TRAJECTORY ANALYSIS:")
            for cluster_id, trajectory_data in self.embedding_trajectories.items():
                report.append(f"\nCluster {cluster_id}:")
                if 'trajectory_length' in trajectory_data:
                    report.append(f"  Trajectory Length: {trajectory_data['trajectory_length']:.3f}")
                    report.append(f"  Average Step Size: {trajectory_data['avg_step_size']:.3f}")
                if 'trajectory_smoothness' in trajectory_data:
                    report.append(f"  Trajectory Smoothness: {trajectory_data['trajectory_smoothness']:.3f}")
        
        report_text = "\n".join(report)
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(f"Explanation report saved to {save_path}")
        return report_text