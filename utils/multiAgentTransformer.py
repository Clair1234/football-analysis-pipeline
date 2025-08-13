import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import math
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from itertools import islice
import gc
import random
import os

class PositionalEncoding(nn.Module):
    """Add positional encoding to transformer inputs"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float16)
        position = torch.arange(0, max_len, dtype=torch.float16).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        #return x + self.pe[:x.size(0), :].to(x.dtype)
        return x + self.pe[:x.size(0), :x.size(1)].to(x.dtype)

class MultiHeadAttentionWithVisualization(nn.Module):
    """Multi-head attention layer that stores attention weights for visualization"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.attention_weights = None  # Store for visualization
        
    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        output, attention_weights = self.attention(
            query, key, value, 
            attn_mask=attn_mask, 
            key_padding_mask=key_padding_mask,
            average_attn_weights=False  # Keep individual head weights
        )
        # MEMORY: Only store attention weights if needed (detach to avoid gradients)
        if self.training or hasattr(self, '_store_attention'):
            self.attention_weights = attention_weights.detach()
        else:
            self.attention_weights = None
        return output

class TransformerEncoderLayerWithVisualization(nn.Module):
    """Transformer encoder layer that captures attention weights"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None, dropout: float = 0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 2
        self.self_attn = MultiHeadAttentionWithVisualization(d_model, nhead, dropout)
        
        # MEMORY: Use inplace operations where possible
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        #self.dropout = nn.Dropout(dropout, inplace=False)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        #self.dropout1 = nn.Dropout(dropout, inplace=False)  # Keep False for residual
        #self.dropout2 = nn.Dropout(dropout, inplace=False)  # Keep False for residual
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention with memory optimization
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        #src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed forward with inplace ReLU
        src2 = self.linear1(src)
        src2 = torch.relu(src2)  # Out-of-place ReLU
        #src2 = self.dropout(src2)
        src2 = self.linear2(src2)
        #src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src
    
    def get_attention_weights(self):
        return self.self_attn.attention_weights

class MultiAgentTransformer(nn.Module):
    """Multi-agent temporal transformer for soccer trajectory prediction with attention visualization"""
    
    def __init__(self, 
                 player_input_dim: int = 22,
                 ball_input_dim: int = 4,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 forecast_horizon: int = 10,
                 dropout: float = 0.1,
                 use_mixed_precision: bool = True,
                 gradient_checkpointing: bool = False):  # NEW: Add gradient checkpointing option
        
        super().__init__()
        self.use_mixed_precision = use_mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        self.d_model = d_model
        self.forecast_horizon = forecast_horizon
        self.num_players = 22
        self.num_agents = self.num_players + 1
        
        # Input projections
        self.player_projection = nn.Linear(player_input_dim, d_model)
        self.ball_projection = nn.Linear(ball_input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Agent embeddings
        self.agent_embedding = nn.Embedding(self.num_agents, d_model)
        
        # Team embeddings
        self.team_embedding = nn.Embedding(3, d_model)
        
        # Transformer encoder with visualization capability
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithVisualization(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # MEMORY: Smaller intermediate dimensions for output heads
        hidden_dim = max(d_model // 4, 32)  # Ensure minimum size
        
        # Output heads with reduced memory footprint
        self.player_output_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),  # Inplace ReLU
            #nn.Dropout(dropout),  # Inplace dropout
            nn.Linear(hidden_dim, 2)  # Direct to x, y coordinates
        )
        
        self.ball_output_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),  # Inplace ReLU
            #nn.Dropout(dropout),  # Inplace dropout
            nn.Linear(hidden_dim, 2)  # Direct to x, y coordinates
        )
        
        # Dropout
        #self.dropout = nn.Dropout(dropout)
        
        # MEMORY: Pre-allocate tensors for common operations to avoid repeated allocation
        self._register_buffers()
        
    def _register_buffers(self):
        """Register commonly used tensors as buffers to avoid repeated allocation"""
        # These will be resized as needed during forward pass
        self.register_buffer('_temp_player_ids', torch.empty(0, dtype=torch.long))
        self.register_buffer('_temp_ball_ids', torch.empty(0, dtype=torch.long))
        
    def forward_training_optimized(self, player_states, ball_states, team_ids=None):
        """Training-optimized forward pass with minimal memory usage and deterministic behavior"""
        if not self.training:
            return self.forward(player_states, ball_states, team_ids)
        
        # Disable attention storage during forward pass
        was_storing_attention = any(
            hasattr(layer.self_attn, '_store_attention') 
            for layer in self.encoder_layers
        )
        
        if was_storing_attention:
            self.disable_attention_storage()
        
        try:
            # Always use the same implementation for consistency
            return self._forward_impl(player_states, ball_states, team_ids)
        finally:
            # Don't restore attention storage during training
            pass

    def forward(self, player_states, ball_states, team_ids=None):
        if self.use_mixed_precision:
            with torch.cuda.amp.autocast():
                return self._forward_impl(player_states, ball_states, team_ids)
        else:
            return self._forward_impl(player_states, ball_states, team_ids) 

    def _get_agent_ids(self, batch_size, num_players, device):
        """Memory-efficient agent ID generation with consistent behavior"""
        total_players = batch_size * num_players
        
        # Always create fresh tensors to ensure consistency
        # This avoids the train/eval mode differences from buffer reuse
        player_ids = torch.arange(num_players, device=device).repeat(batch_size)
        ball_ids = torch.full((batch_size,), num_players, dtype=torch.long, device=device)
        
        return torch.cat([player_ids, ball_ids])

    def _forward_impl(self, player_states, ball_states, team_ids=None):
        """
        Memory-optimized forward implementation with deterministic behavior
        """
        batch_size, seq_len, num_players, _ = player_states.shape
        device = player_states.device
        
        # Ensure deterministic behavior by disabling gradient checkpointing during inference
        use_checkpointing = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        
        # MEMORY: Process projections with gradient checkpointing if enabled
        if use_checkpointing:
            player_features = torch.utils.checkpoint.checkpoint(
                self.player_projection, player_states
            )
            ball_features = torch.utils.checkpoint.checkpoint(
                self.ball_projection, ball_states
            )
        else:
            player_features = self.player_projection(player_states)
            ball_features = self.ball_projection(ball_states)
        
        # MEMORY: More efficient reshaping - use explicit operations for consistency
        # Ensure operations are identical regardless of training mode
        player_features = player_features.permute(1, 0, 2, 3)
        player_features = player_features.contiguous()
        player_features = player_features.view(seq_len, batch_size * num_players, self.d_model)
        
        ball_features = ball_features.transpose(0, 1)
        
        # Combine features
        all_features = torch.cat([player_features, ball_features], dim=1)
        
        # MEMORY: Use consistent agent ID generation - always create fresh tensors
        agent_ids = self._get_agent_ids(batch_size, num_players, device)
        
        # Get embeddings
        agent_embeddings = self.agent_embedding(agent_ids)
        agent_embeddings = agent_embeddings.unsqueeze(0)
        agent_embeddings = agent_embeddings.expand(seq_len, -1, -1)
        
        # Team embeddings (if provided) - ensure consistent tensor creation 
        if team_ids is not None:
            team_assignments = torch.cat([
                team_ids.view(-1),  # Use view instead of flatten for consistency
                torch.full((batch_size,), 2, dtype=torch.long, device=device)
            ])
            team_embeddings = self.team_embedding(team_assignments)
            team_embeddings = team_embeddings.unsqueeze(0)
            team_embeddings = team_embeddings.expand(seq_len, -1, -1)
            features = all_features + agent_embeddings + team_embeddings
        else:
            features = all_features + agent_embeddings
        
        # Add positional encoding - ensure consistent scaling
        scale_factor = math.sqrt(self.d_model)
        features = features * scale_factor
        features = self.pos_encoder(features)
        
        # MEMORY: Apply transformer layers with optional gradient checkpointing
        if use_checkpointing:
            for layer in self.encoder_layers:
                features = torch.utils.checkpoint.checkpoint(layer, features)
        else:
            for layer in self.encoder_layers:
                features = layer(features)
        
        # Use last timestep for prediction
        last_features = features[-1]
        
        # Split back into player and ball features - ensure consistent indexing
        player_final = last_features[:batch_size * num_players]
        player_final = player_final.contiguous()
        player_final = player_final.view(batch_size, num_players, self.d_model)
        
        ball_final = last_features[batch_size * num_players:]
        
        # Generate predictions
        player_preds = self.player_output_head(player_final)  # (batch, num_players, 2)
        ball_preds = self.ball_output_head(ball_final)        # (batch, 2)
        
        # Reshape to match expected output format
        player_preds = player_preds.unsqueeze(2)  # (batch, num_players, 1, 2)
        ball_preds = ball_preds.unsqueeze(1)      # (batch, 1, 2)
        
        return player_preds, ball_preds

    def get_attention_weights(self, layer_idx: int = -1) -> torch.Tensor:
        """Get attention weights from specified layer (-1 for last layer)"""
        if layer_idx == -1:
            layer_idx = len(self.encoder_layers) - 1
        return self.encoder_layers[layer_idx].get_attention_weights()

    def get_all_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all layers"""
        return [layer.get_attention_weights() for layer in self.encoder_layers 
                if layer.get_attention_weights() is not None]

    def enable_attention_storage(self):
        """Enable storing attention weights for visualization"""
        for layer in self.encoder_layers:
            layer.self_attn._store_attention = True

    def disable_attention_storage(self):
        """Disable storing attention weights to save memory"""
        for layer in self.encoder_layers:
            if hasattr(layer.self_attn, '_store_attention'):
                delattr(layer.self_attn, '_store_attention')
            layer.self_attn.attention_weights = None

    def extract_embeddings(self, player_states, ball_states, team_ids=None, layer_idx=-1):
        """Extract embeddings with memory optimization and consistent behavior"""
        was_training = self.training
        self.eval()
        
        # Disable attention storage during embedding extraction to save memory
        attention_was_enabled = any(
            hasattr(layer.self_attn, '_store_attention') 
            for layer in self.encoder_layers
        )
        if attention_was_enabled:
            self.disable_attention_storage()
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings, metadata = self._extract_embeddings_impl(
                            player_states, ball_states, team_ids, layer_idx
                        )
                else:
                    embeddings, metadata = self._extract_embeddings_impl(
                        player_states, ball_states, team_ids, layer_idx
                    )
                
                # Compute tactical context with memory efficiency
                tactical_context = self._compute_tactical_context(embeddings, metadata)
                metadata['tactical_context'] = tactical_context
                
                return embeddings, metadata
        finally:
            # Restore attention storage state and training mode
            if attention_was_enabled:
                self.enable_attention_storage()
            self.train(was_training)

    def _extract_embeddings_impl(self, player_states, ball_states, team_ids=None, layer_idx=-1):
        """Memory-efficient embedding extraction with consistent behavior"""
        batch_size, seq_len, num_players, _ = player_states.shape
        device = player_states.device
        
        # Use the same forward logic but stop at specified layer
        player_features = self.player_projection(player_states)
        ball_features = self.ball_projection(ball_states)
        
        # Efficient reshaping
        player_features = player_features.permute(1, 0, 2, 3).contiguous().reshape(
            seq_len, batch_size * num_players, self.d_model
        )
        ball_features = ball_features.transpose(0, 1)
        all_features = torch.cat([player_features, ball_features], dim=1)
        
        # Agent embeddings (consistent with _forward_impl)
        agent_ids = self._get_agent_ids(batch_size, num_players, device)
        agent_embeddings = self.agent_embedding(agent_ids).unsqueeze(0).expand(seq_len, -1, -1)
        
        if team_ids is not None:
            team_assignments = torch.cat([
                team_ids.flatten(),
                torch.full((batch_size,), 2, dtype=torch.long, device=device)
            ])
            team_embeddings = self.team_embedding(team_assignments).unsqueeze(0).expand(seq_len, -1, -1)
            features = all_features + agent_embeddings + team_embeddings
        else:
            features = all_features + agent_embeddings
        
        features = self.pos_encoder(features * math.sqrt(self.d_model))
        
        # Forward through layers up to target
        target_layer = layer_idx if layer_idx >= 0 else len(self.encoder_layers) + layer_idx
        
        for i, layer in enumerate(self.encoder_layers):
            features = layer(features)
            if i == target_layer:
                break
        
        # Extract embeddings
        final_embeddings = features[-1]
        embeddings = final_embeddings.reshape(batch_size, self.num_agents, self.d_model)
        
        # Minimal metadata to save memory
        metadata = {
            'positions': player_states[:, -1, :, :2].clone(),
            'velocities': player_states[:, -1, :, 2:4].clone() if player_states.shape[-1] > 2 else None,
            'ball_position': ball_states[:, -1, :2].clone(),
            'team_ids': team_ids.clone() if team_ids is not None else None,
            'sequence_length': seq_len,
            'batch_size': batch_size
        }
        
        return embeddings, metadata

    def _compute_tactical_context(self, embeddings, metadata):
        """Memory-efficient tactical context computation"""
        batch_size, num_agents, d_model = embeddings.shape
        device = embeddings.device
        
        # Only compute essential metrics to save memory
        player_embeddings = embeddings[:, :-1, :]  # Exclude ball
        
        # Simplified formation analysis - compute pairwise distances in embedding space
        formation_coherence = torch.zeros(batch_size, device=device)
        for batch_idx in range(batch_size):
            player_emb = player_embeddings[batch_idx]  # (num_players, d_model)
            # Compute mean pairwise cosine similarity
            norm_emb = torch.nn.functional.normalize(player_emb, dim=1)
            similarity_matrix = torch.mm(norm_emb, norm_emb.t())
            # Exclude diagonal (self-similarity)
            mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=device)
            formation_coherence[batch_idx] = similarity_matrix[mask].mean()
        
        # Team coordination with reduced memory footprint
        coordination_scores = torch.zeros(batch_size, device=device)
        
        if metadata['team_ids'] is not None:
            team_ids = metadata['team_ids']
            for batch_idx in range(batch_size):
                team_a_mask = team_ids[batch_idx] == 0
                team_b_mask = team_ids[batch_idx] == 1
                
                batch_score = 0.0
                count = 0
                
                # Team A coordination
                if team_a_mask.sum() > 1:
                    team_a_emb = player_embeddings[batch_idx, team_a_mask]
                    team_a_norm = torch.nn.functional.normalize(team_a_emb, dim=1)
                    team_a_sim = torch.mm(team_a_norm, team_a_norm.t())
                    mask_a = ~torch.eye(team_a_sim.size(0), dtype=torch.bool, device=device)
                    batch_score += team_a_sim[mask_a].mean()
                    count += 1
                
                # Team B coordination
                if team_b_mask.sum() > 1:
                    team_b_emb = player_embeddings[batch_idx, team_b_mask]
                    team_b_norm = torch.nn.functional.normalize(team_b_emb, dim=1)
                    team_b_sim = torch.mm(team_b_norm, team_b_norm.t())
                    mask_b = ~torch.eye(team_b_sim.size(0), dtype=torch.bool, device=device)
                    batch_score += team_b_sim[mask_b].mean()
                    count += 1
                
                coordination_scores[batch_idx] = batch_score / max(count, 1)
        
        # Spatial control - simplified as embedding magnitudes
        spatial_control = torch.norm(player_embeddings, dim=-1)  # (batch_size, num_players)
        
        return {
            'formation_coherence': formation_coherence,
            'team_coordination': coordination_scores,
            'spatial_control': spatial_control
        }

    def memory_efficient_inference(self, player_states, ball_states, team_ids=None):
        """
        Memory-efficient inference mode - disables attention storage and uses minimal memory
        """
        was_training = self.training
        self.eval()
        self.disable_attention_storage()
        
        try:
            with torch.no_grad():
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        return self._forward_impl(player_states, ball_states, team_ids)
                else:
                    return self._forward_impl(player_states, ball_states, team_ids)
        finally:
            self.train(was_training)

    def extract_all_layer_embeddings(self, player_states, ball_states, team_ids=None):
        """Extract embeddings from all transformer layers for hierarchical analysis"""
        all_embeddings = []
        
        for layer_idx in range(len(self.encoder_layers)):
            embeddings, metadata = self.extract_embeddings(
                player_states, ball_states, team_ids, layer_idx
            )
            all_embeddings.append(embeddings)
        
        return all_embeddings, metadata

    def get_feature_importance_maps(self, player_states, ball_states, team_ids=None):
        """
        Get attention-based feature importance for interpretability analysis
        Returns attention weights and gradient-based importance scores
        """
        self.eval()
        
        # Enable gradients for input
        player_states.requires_grad_(True)
        ball_states.requires_grad_(True)
        
        # Forward pass
        player_preds, ball_preds = self.forward(player_states, ball_states, team_ids)
        
        # Get attention weights from all layers
        all_attention_weights = self.get_all_attention_weights()
        
        # Compute gradients w.r.t. inputs for saliency
        loss = player_preds.sum() + ball_preds.sum()  # Simple aggregation for gradients
        loss.backward()
        
        importance_maps = {
            'player_gradients': player_states.grad.abs() if player_states.grad is not None else None,
            'ball_gradients': ball_states.grad.abs() if ball_states.grad is not None else None,
            'attention_weights': all_attention_weights,
            'predictions': {'players': player_preds.detach(), 'ball': ball_preds.detach()}
        }
        
        return importance_maps

class AttentionVisualizer:
    """Utility class for visualizing attention patterns"""
    
    def __init__(self, num_players: int = 22):
        self.num_players = num_players
        self.num_agents = num_players + 1
        
    def plot_attention_heatmap(self, attention_weights: torch.Tensor, 
                            layer_idx: int = 0, head_idx: int = 0,
                            save_path: str = None, figsize: tuple = (12, 10)):
        """
        Plot attention heatmap for a specific layer and head with proper agent extraction
        
        Args:
            attention_weights: Shape (batch*seq_len, num_heads, total_tokens, total_tokens)
            layer_idx: Which layer to visualize
            head_idx: Which attention head to visualize
        """
        
        print(f"Processing attention weights with shape: {attention_weights.shape}")
        
        # The attention weights have shape (batch*seq_len, num_heads, total_tokens, total_tokens)
        # where total_tokens = batch_size * num_agents per sequence step
        
        # Take first sequence step and specified head
        if attention_weights.dim() == 4:
            attn = attention_weights[0, head_idx].cpu().numpy()  # (total_tokens, total_tokens)
        else:
            print(f"Unexpected attention shape: {attention_weights.shape}")
            return
        
        print(f"Extracted attention matrix shape: {attn.shape}")
        
        # Assuming the first batch's agents are in positions 0:num_agents
        num_agents = self.num_agents  # 23
        
        if attn.shape[0] >= num_agents and attn.shape[1] >= num_agents:
            # Extract first batch's agent-to-agent attention
            attn_map = attn[:num_agents, :num_agents]
            print(f"Extracted {num_agents}x{num_agents} attention map for first batch")
        else:
            print(f"Cannot extract {num_agents}x{num_agents} from {attn.shape}")
            return
        
        plt.figure(figsize=figsize)
        
        # Create labels
        labels = [f'P{i}' for i in range(self.num_players)] + ['Ball']
        
        try:
            # Plot heatmap
            sns.heatmap(attn_map, 
                    xticklabels=labels, 
                    yticklabels=labels,
                    cmap='Blues', 
                    annot=False,
                    cbar_kws={'label': 'Attention Weight'})
            
            plt.title(f'Attention Pattern - Layer {layer_idx}, Head {head_idx}')
            plt.xlabel('Key (Attended To)')
            plt.ylabel('Query (Attending From)')
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            #plt.show()
            
            
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            print(f"Attention map shape: {attn_map.shape}")
        plt.close()
    
    def plot_agent_attention_summary(self, attention_weights: torch.Tensor,
                                agent_idx: int = 22,  # Ball by default
                                save_path: str = None):
        """
        Plot how much attention a specific agent receives from all others
        """
        
        print(f"Processing agent attention with shape: {attention_weights.shape}")
        
        # Extract first batch, average over heads
        if attention_weights.dim() == 4:
            # Average over heads for first sequence step
            attn = attention_weights[0].mean(dim=0).cpu().numpy()  # (total_tokens, total_tokens)
        else:
            return
        
        # Extract first batch's agents
        num_agents = self.num_agents
        if attn.shape[0] >= num_agents and attn.shape[1] >= num_agents:
            attn_map = attn[:num_agents, :num_agents]
        else:
            print(f"Cannot extract agent attention from shape {attn.shape}")
            return
        
        # Get attention TO the specified agent
        if agent_idx < num_agents:
            attention_to_agent = attn_map[:, agent_idx]
        else:
            print(f"Agent index {agent_idx} out of bounds")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create labels
        labels = [f'P{i}' for i in range(self.num_players)] + ['Ball']
        agent_name = labels[agent_idx] if agent_idx < len(labels) else f'Agent{agent_idx}'
        
        # Bar plot
        bars = plt.bar(range(len(attention_to_agent)), attention_to_agent)
        plt.xlabel('Agent')
        plt.ylabel('Average Attention Weight')
        plt.title(f'Attention Directed Toward {agent_name}')
        plt.xticks(range(len(labels[:len(attention_to_agent)])), labels[:len(attention_to_agent)], rotation=45)
        
        # Highlight the target agent
        if agent_idx < len(bars):
            bars[agent_idx].set_color('red')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_attention(self, attention_weights: torch.Tensor,
                            from_agent: int = 22, to_agent: int = 0,
                            save_path: str = None):
        """
        Plot how attention between two agents changes over time
        Note: This requires sequence-level attention which may not be available
        """
        
        print(f"Temporal attention analysis with shape: {attention_weights.shape}")
        
        # For temporal analysis, we need attention across multiple sequence steps
        # The current shape suggests batch*seq_len in first dimension
        
        # This is challenging with the current attention format
        # For now, show attention strength at the current timestep
        
        if attention_weights.dim() == 4:
            # Average over heads for first few sequence steps
            seq_steps = min(10, attention_weights.shape[0])
            temporal_attn = []
            
            num_agents = self.num_agents
            
            for step in range(seq_steps):
                attn = attention_weights[step].mean(dim=0).cpu().numpy()  # Average over heads
                if attn.shape[0] >= num_agents and attn.shape[1] >= num_agents:
                    attn_map = attn[:num_agents, :num_agents]
                    if from_agent < num_agents and to_agent < num_agents:
                        temporal_attn.append(attn_map[from_agent, to_agent])
            
            if temporal_attn:
                plt.figure(figsize=(10, 6))
                plt.plot(temporal_attn, marker='o')
                plt.xlabel('Sequence Step')
                plt.ylabel('Attention Weight')
                
                labels = [f'P{i}' for i in range(self.num_players)] + ['Ball']
                from_name = labels[from_agent] if from_agent < len(labels) else f'Agent{from_agent}'
                to_name = labels[to_agent] if to_agent < len(labels) else f'Agent{to_agent}'
                
                plt.title(f'Attention Over Sequence: {from_name} → {to_name}')
                plt.grid(True, alpha=0.3)
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                print("Could not extract temporal attention data")
        else:
            print("Cannot analyze temporal attention with current attention format")

class SoccerTrainer:
    """Enhanced training class with attention visualization"""


    def __init__(self, 
                 model: MultiAgentTransformer,
                 dataset_processor,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 1e-5):
        
        self.model = model.to(device)
        self.dataset_processor = dataset_processor
        self.device = device
        self.visualizer = AttentionVisualizer(model.num_players)
        self.train_losses = None
        self.val_losses = None
        self.metric_history = None
        self.best_val_loss = float('inf')
        
        # Set train mode
        self.model.train()
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)
        
        # Loss functions
        self.position_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
  
    def update_dataset(self, new_dataset):
        self.dataset = new_dataset

    def reset_best_val_loss(self):
        self.best_val_loss = float('inf')
        
    def visualize_attention(self, dataloader: DataLoader, num_samples: int = 1):
        """Visualize attention patterns on validation data"""
        was_training = self.model.training
        self.model.eval()
        self.model.enable_attention_storage()
        try:
            with torch.no_grad():
                for i, batch in enumerate(dataloader):
                    if i >= num_samples:
                        break
                        
                    player_states = batch['player_states'].to(self.device)
                    ball_states = batch['ball_states'].to(self.device)
                    
                    # Get team IDs if available
                    team_ids = batch.get('team_ids', None)
                    if team_ids is not None:
                        team_ids = team_ids.to(self.device)
                    
                    # Forward pass
                    _ = self.model(player_states, ball_states, team_ids)
                    
                    # Get attention weights from last layer
                    attention_weights = self.model.get_attention_weights()
                    
                    if attention_weights is not None:
                        print(f"\nVisualization for sample {i+1}")
                        print(f"Attention weights shape: {attention_weights.shape}")
                        
                        try:
                            os.makedirs("analysis/visualisation", exist_ok=True)
                            # Plot attention heatmap
                            self.visualizer.plot_attention_heatmap(
                                attention_weights, 
                                layer_idx=-1, 
                                head_idx=0,
                                save_path=f'analysis/visualisation/attention_heatmap_sample_{i+1}.png'
                            )
                            
                            # Plot ball attention
                            self.visualizer.plot_agent_attention_summary(
                                attention_weights,
                                agent_idx=22,  # Ball
                                save_path=f'analysis/visualisation/ball_attention_sample_{i+1}.png'
                            )
                            
                            # Plot temporal attention (ball to first player)
                            self.visualizer.plot_temporal_attention(
                                attention_weights,
                                from_agent=22,  # Ball
                                to_agent=0,     # First player
                                save_path=f'analysis/visualisation/temporal_attention_sample_{i+1}.png'
                            )
                        except Exception as e:
                            print(f"Error in visualization: {e}")
                            print("Skipping visualization for this sample")
                    else:
                        print("No attention weights available")
                        
                    # ADDED: Clear attention weights after each sample to save memory
                    self.model.disable_attention_storage()
                    self.model.enable_attention_storage()
                    
        finally:
            # ADDED: Always clean up attention storage and restore training state
            self.model.disable_attention_storage()
            if was_training:
                self.model.train()
            
            # ADDED: Explicit memory cleanup after visualization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def compute_loss(self, player_preds, ball_preds, target_players, target_ball, 
                 player_weight: float = 1.0, ball_weight: float = 1.0):
        """Compute combined loss for players and ball on unnormalized data if applicable"""

        if hasattr(self.dataset_processor, 'inverse_transform_predictions_tensor'):
            #print("inverse_transform_predictions_tensor")

            player_preds = self.dataset_processor.inverse_transform_predictions_tensor(player_preds, kind="position")
            target_players = self.dataset_processor.inverse_transform_predictions_tensor(target_players, kind="position")
            ball_preds = self.dataset_processor.inverse_transform_predictions_tensor(ball_preds, kind="position")
            target_ball = self.dataset_processor.inverse_transform_predictions_tensor(target_ball, kind="position")

        player_loss = self.position_loss(player_preds, target_players)
        ball_loss = self.position_loss(ball_preds, target_ball)

        total_loss = player_weight * player_loss + ball_weight * ball_loss

        return total_loss, player_loss, ball_loss
    
    def train_epoch(self, dataloader: DataLoader, player_weight: float = 1.0, ball_weight: float = 1.0):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_player_loss = 0
        total_ball_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            player_states = batch['player_states'].to(self.device)
            ball_states = batch['ball_states'].to(self.device)
            target_players = batch['target_players'].to(self.device)
            target_ball = batch['target_ball'].to(self.device)
            
            # Get team assignments if available
            team_ids = batch.get('team_ids', None)
            if team_ids is not None:
                team_ids = team_ids.to(self.device)
            
            # Reshape targets to match model output
            target_players = target_players.transpose(1, 2)
            
            # Forward pass
            player_preds, ball_preds = self.model.forward(player_states, ball_states, team_ids)

            # Compute loss
            loss, player_loss, ball_loss = self.compute_loss(
                player_preds, ball_preds, target_players, target_ball,
                player_weight, ball_weight
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += loss.item()
            total_player_loss += player_loss.item()
            total_ball_loss += ball_loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Batch {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Player: {player_loss.item():.4f}, Ball: {ball_loss.item():.4f}')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        avg_loss = total_loss / len(dataloader)
        avg_player_loss = total_player_loss / len(dataloader)
        avg_ball_loss = total_ball_loss / len(dataloader)
        
        return avg_loss, avg_player_loss, avg_ball_loss
    
    def validate(self, dataloader: DataLoader, player_weight: float = 1.0, ball_weight: float = 1.0):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_player_loss = 0
        total_ball_loss = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                player_states = batch['player_states'].to(self.device)
                ball_states = batch['ball_states'].to(self.device)
                target_players = batch['target_players'].to(self.device)
                target_ball = batch['target_ball'].to(self.device)
                
                team_ids = batch.get('team_ids', None)
                if team_ids is not None:
                    team_ids = team_ids.to(self.device)
                
                target_players = target_players.transpose(1, 2)
                    
                player_preds, ball_preds = self.model.forward(player_states, ball_states, team_ids)
                
                loss, player_loss, ball_loss = self.compute_loss(
                    player_preds, ball_preds, target_players, target_ball,
                    player_weight, ball_weight
                )
                
                total_loss += loss.item()
                total_player_loss += player_loss.item()
                total_ball_loss += ball_loss.item()
                
                if batch_idx % 50 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        avg_loss = total_loss / len(dataloader)
        avg_player_loss = total_player_loss / len(dataloader)
        avg_ball_loss = total_ball_loss / len(dataloader)
        
        return avg_loss, avg_player_loss, avg_ball_loss

    
    def compare_train_val_data(self, train_batch, val_batch):
        """
        Compare training and validation data to find differences
        """
        print(f"\n=== COMPARING TRAIN VS VAL DATA ===")
        
        for key in train_batch.keys():
            if key in val_batch:
                train_data = train_batch[key]
                val_data = val_batch[key]
                
                print(f"\n{key}:")
                print(f"  Train shape: {train_data.shape}, Val shape: {val_data.shape}")
                print(f"  Train stats: mean={train_data.mean():.4f}, std={train_data.std():.4f}")
                print(f"  Val stats: mean={val_data.mean():.4f}, std={val_data.std():.4f}")
                
                # Check for overlap
                if len(train_data.shape) > 1:
                    train_flat = train_data.flatten()
                    val_flat = val_data.flatten()
                    
                    # Sample for efficiency
                    train_sample = train_flat[::max(1, len(train_flat)//1000)]
                    val_sample = val_flat[::max(1, len(val_flat)//1000)]
                    
                    # Check if validation data is subset of training data (data leakage)
                    overlap_count = 0
                    for val_val in val_sample[:100]:  # Check first 100 values
                        if torch.any(torch.isclose(train_sample, val_val, atol=1e-6)):
                            overlap_count += 1
                    
                    overlap_percentage = overlap_count / min(100, len(val_sample)) * 100
                    print(f"  Potential data overlap: {overlap_percentage:.1f}%")

    def check_model_behavior(self, batch):
        """
        Check if model behaves consistently across different modes with detailed debugging
        """
        print(f"\n=== CHECKING MODEL BEHAVIOR ===")
        
        player_states = batch['player_states'].to(self.device)
        ball_states = batch['ball_states'].to(self.device)
        team_ids = batch.get('team_ids', None)
        
        if team_ids is not None:
            team_ids = team_ids.to(self.device)

        # Store original training state
        original_training = self.model.training
        
        # Debug: Check model state
        print(f"Model has {sum(1 for _ in self.model.parameters())} parameters")
        
        # Check for any modules that behave differently in train/eval
        modules_with_training_behavior = []
        for name, module in self.model.named_modules():
            if hasattr(module, 'training') and type(module).__name__ in [
                'Dropout', 'BatchNorm1d', 'BatchNorm2d', 'BatchNorm3d', 
                'LayerNorm', 'GroupNorm', 'InstanceNorm1d', 'InstanceNorm2d', 'InstanceNorm3d'
            ]:
                modules_with_training_behavior.append((name, type(module).__name__))
        
        if modules_with_training_behavior:
            print("Found modules that behave differently in train/eval:")
            for name, module_type in modules_with_training_behavior:
                print(f"  - {name}: {module_type}")
        else:
            print("No training-dependent modules found")
        
        # Test 1: Training mode with no_grad
        self.model.train()
        torch.manual_seed(42)  # Set seed for reproducibility
        with torch.no_grad():
            player_preds_train, ball_preds_train = self.model.forward(
                player_states, ball_states, team_ids
            )
        
        # Test 2: Eval mode
        self.model.eval()
        torch.manual_seed(42)  # Same seed
        with torch.no_grad():
            player_preds_eval, ball_preds_eval = self.model.forward(
                player_states, ball_states, team_ids
            )
        
        # Test 3: Direct _forward_impl in train mode
        self.model.train()
        torch.manual_seed(42)  # Same seed
        with torch.no_grad():
            player_preds_direct_train, ball_preds_direct_train = self.model.forward(
                player_states, ball_states, team_ids
            )
        
        # Test 4: Direct _forward_impl in eval mode
        self.model.eval()
        torch.manual_seed(42)  # Same seed
        with torch.no_grad():
            player_preds_direct_eval, ball_preds_direct_eval = self.model.forward(
                player_states, ball_states, team_ids
            )
        
        # Restore original training state
        self.model.train(original_training)
        
        # Compare outputs
        player_diff_1 = torch.abs(player_preds_train - player_preds_eval).mean()
        ball_diff_1 = torch.abs(ball_preds_train - ball_preds_eval).mean()
        
        player_diff_2 = torch.abs(player_preds_direct_train - player_preds_direct_eval).mean()
        ball_diff_2 = torch.abs(ball_preds_direct_train - ball_preds_direct_eval).mean()
        
        player_diff_3 = torch.abs(player_preds_train - player_preds_direct_train).mean()
        ball_diff_3 = torch.abs(ball_preds_train - ball_preds_direct_train).mean()
        
        print(f"\n--- Results ---")
        print(f"forward_training_optimized vs memory_efficient_inference:")
        print(f"  Player diff: {player_diff_1:.8f}")
        print(f"  Ball diff: {ball_diff_1:.8f}")
        
        print(f"_forward_impl train vs eval:")
        print(f"  Player diff: {player_diff_2:.8f}")
        print(f"  Ball diff: {ball_diff_2:.8f}")
        
        print(f"forward_training_optimized vs _forward_impl (both train):")
        print(f"  Player diff: {player_diff_3:.8f}")
        print(f"  Ball diff: {ball_diff_3:.8f}")
        
        # Additional debugging - check intermediate values
        print(f"\n--- Debug Info ---")
        print(f"Player predictions ranges:")
        print(f"  Train: [{player_preds_train.min():.6f}, {player_preds_train.max():.6f}]")
        print(f"  Eval:  [{player_preds_eval.min():.6f}, {player_preds_eval.max():.6f}]")
        
        print(f"Ball predictions ranges:")
        print(f"  Train: [{ball_preds_train.min():.6f}, {ball_preds_train.max():.6f}]")
        print(f"  Eval:  [{ball_preds_eval.min():.6f}, {ball_preds_eval.max():.6f}]")
        
        tolerance = 1e-6
        
        if player_diff_2 > tolerance or ball_diff_2 > tolerance:
            print(f"\nISSUE: _forward_impl behaves differently in train vs eval mode")
            print("This suggests the issue is in the core forward implementation")
            return False
        
        if player_diff_1 > tolerance or ball_diff_1 > tolerance:
            print(f"\nISSUE: Wrapper methods behave differently")
            print("This suggests the issue is in forward_training_optimized or memory_efficient_inference")
            return False
        
        print(f"\nModel behavior is consistent across all modes")
        return True
                
    def train(self, train_dataloader: DataLoader, val_dataloader: DataLoader, 
            epochs: int = 100, player_weight: float = 1.0, ball_weight: float = 1.0,
            save_path: str = './models/best_soccer_model.pth', visualize_every: int = 10, 
            min_epochs: int = 10):
        """Training loop with full metric tracking and final visualizations."""

        patience = 10
        patience_counter = 0

        # Initialize tracking
        self.train_losses = []
        self.val_losses = []
        self.lrs = []
        self.metric_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'player_mse': [],
            'player_mae': [],
            'player_rmse': [],
        }

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Get a batch from the training and validation dataloaders
            train_batch = next(iter(train_dataloader))
            val_batch = next(iter(val_dataloader))
        
            # Compare data
            self.compare_train_val_data(train_batch, val_batch)
        
            # Check behaviors
            self.check_model_behavior(train_batch)

            # Training
            train_loss, train_player_loss, train_ball_loss = self.train_epoch(
                train_dataloader, player_weight, ball_weight
            )

            # Validation
            val_loss, val_player_loss, val_ball_loss = self.validate(
                val_dataloader, player_weight, ball_weight
            )

            # Scheduler step and learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Save losses and learning rate
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.lrs.append(current_lr)

            print(f'Train Loss: {train_loss:.4f} (Player: {train_player_loss:.4f}, Ball: {train_ball_loss:.4f})')
            print(f'Val Loss: {val_loss:.4f} (Player: {val_player_loss:.4f}, Ball: {val_ball_loss:.4f})')
            print(f'Learning Rate: {current_lr:.6f}')

            # Evaluate full metrics
            val_metrics = self.evaluate_metrics(val_dataloader)
            self.metric_history['epoch'].append(epoch)
            self.metric_history['train_loss'].append(train_loss)
            self.metric_history['val_loss'].append(val_loss)
            self.metric_history['learning_rate'].append(current_lr)
            self.metric_history['player_mse'].append(val_metrics['player_mse'])
            self.metric_history['player_mae'].append(val_metrics['player_mae'])
            self.metric_history['player_rmse'].append(val_metrics['player_rmse'])

            # Visualize attention periodically
            if (epoch + 1) % visualize_every == 0:
                print(f"\nVisualizing attention patterns at epoch {epoch+1}")
                # Force garbage collection before visualization
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.visualize_attention(val_dataloader, num_samples=1)
                
                # ADDED: Explicit memory cleanup after visualization
                if hasattr(self.model, 'memory_cleanup'):
                    self.model.memory_cleanup()
                    
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                }, save_path)
                print(f'New best model saved with val_loss: {val_loss:.4f}')
            else:
                if epoch >= min_epochs:
                    patience_counter += 1

            if epoch >= min_epochs and patience_counter >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def get_best_val_loss(self):
        return self.best_val_loss

    def update_best_val_loss(self, new_loss):
        self.best_val_loss = new_loss

    def get_training_history(self):
        return self.metric_history
    
    def evaluate_metrics(self, dataloader: DataLoader):
        """Compute detailed evaluation metrics"""
        self.model.eval()
        
        all_player_preds = []
        all_player_targets = []
        all_ball_preds = []
        all_ball_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                player_states = batch['player_states'].to(self.device)
                ball_states = batch['ball_states'].to(self.device)
                target_players = batch['target_players'].to(self.device)
                target_ball = batch['target_ball'].to(self.device)
                
                team_ids = batch.get('team_ids', None)
                if team_ids is not None:
                    team_ids = team_ids.to(self.device)
                
                target_players = target_players.transpose(1, 2)
                
                player_preds, ball_preds = self.model.forward(player_states, ball_states, team_ids)
                
                # Convert back to original scale if needed
                if hasattr(self.dataset_processor, 'inverse_transform_predictions'):
                    player_preds_unscaled = self.dataset_processor.inverse_transform_predictions(
                        player_preds.cpu().numpy()
                    )
                    target_players_unscaled = self.dataset_processor.inverse_transform_predictions(
                        target_players.cpu().numpy()
                    )
                    # Fix: Ball predictions need proper reshaping
                    ball_preds_np = ball_preds.cpu().numpy()  # Shape: (batch, horizon, 2)
                    ball_preds_unscaled = self.dataset_processor.inverse_transform_predictions(
                        ball_preds_np
                    )
                    target_ball_np = target_ball.cpu().numpy()  # Shape: (batch, horizon, 2)
                    target_ball_unscaled = self.dataset_processor.inverse_transform_predictions(
                        target_ball_np
                    )
                else:
                    player_preds_unscaled = player_preds.cpu().numpy()
                    target_players_unscaled = target_players.cpu().numpy()
                    ball_preds_unscaled = ball_preds.cpu().numpy()
                    target_ball_unscaled = target_ball.cpu().numpy()
                
                all_player_preds.append(player_preds_unscaled)
                all_player_targets.append(target_players_unscaled)
                all_ball_preds.append(ball_preds_unscaled)
                all_ball_targets.append(target_ball_unscaled)
        
        # Concatenate all predictions
        player_preds = np.concatenate(all_player_preds, axis=0)
        player_targets = np.concatenate(all_player_targets, axis=0)
        ball_preds = np.concatenate(all_ball_preds, axis=0)
        ball_targets = np.concatenate(all_ball_targets, axis=0)
        
        # Compute metrics
        metrics = {}
        
        # Player metrics
        player_mse = mean_squared_error(player_targets.reshape(-1, 2), player_preds.reshape(-1, 2))
        player_mae = mean_absolute_error(player_targets.reshape(-1, 2), player_preds.reshape(-1, 2))
        metrics['player_mse'] = player_mse
        metrics['player_mae'] = player_mae
        metrics['player_rmse'] = np.sqrt(player_mse)
        
        # Ball metrics
        ball_mse = mean_squared_error(ball_targets.reshape(-1, 2), ball_preds.reshape(-1, 2))
        ball_mae = mean_absolute_error(ball_targets.reshape(-1, 2), ball_preds.reshape(-1, 2))
        metrics['ball_mse'] = ball_mse
        metrics['ball_mae'] = ball_mae
        metrics['ball_rmse'] = np.sqrt(ball_mse)
        
        # Average displacement error per time step
        for t in range(player_preds.shape[2]):  # should be 1
            player_errors = np.sqrt(np.sum((player_preds[:, :, t, :] - player_targets[:, :, t, :]) ** 2, axis=2))
            metrics[f'player_ade_t{t+1}'] = np.mean(player_errors)
            
            ball_errors = np.sqrt(np.sum((ball_preds[:, t, :] - ball_targets[:, t, :]) ** 2, axis=1))
            metrics[f'ball_ade_t{t+1}'] = np.mean(ball_errors)
        
        return metrics

    def collect_embeddings_for_clustering(self, dataloader: DataLoader, 
                                        layer_idx: int = -1, 
                                        max_samples: int = None,
                                        extract_per_team: bool = True):
        """
        Collect embeddings from trained model for clustering analysis
        
        Args:
            dataloader: DataLoader to extract embeddings from
            layer_idx: Which transformer layer to extract from
            max_samples: Limit number of samples (None for all)
            extract_per_team: Reshape embeddings either:
                        [2000, 12, 128] if True or 
                        [1000, 23, 128] else
        
        Returns:
            embeddings: (N, num_agents, d_model) - All embeddings
            metadata: Dict with positions, team info, etc.
            sequence_ids: Track which sequence each embedding came from
        """
        
        self.model.eval()
        
        all_embeddings = []
        all_metadata = []
        sequence_ids = []
        
        # Determine which batches to process
        total_batches = len(dataloader)
        if max_samples:
            max_batches = min(total_batches, (max_samples + dataloader.batch_size - 1) // dataloader.batch_size)
            selected_batch_indices = random.sample(range(total_batches), max_batches)
            # test consstency with first 1000
            selected_batch_indices = range(max_batches)
            #selected_batch_indices.sort()  # Sort for consistent processing order
        else:
            selected_batch_indices = list(range(total_batches))
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx not in selected_batch_indices:
                    continue
                    
                player_states = batch['player_states'].to(self.device)
                ball_states = batch['ball_states'].to(self.device)
                team_ids = batch.get('team_ids', None)
                if team_ids is not None:
                    team_ids = team_ids.to(self.device)
                
                # Extract embeddings
                embeddings, metadata = self.model.extract_embeddings(
                    player_states, ball_states, team_ids, layer_idx
                )
                
                all_embeddings.append(embeddings.cpu())
                all_metadata.append({
                    **{k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in metadata.items()},
                    'player_states': player_states.cpu(),
                    'ball_states': ball_states.cpu()
                })
                
                # Track sequence IDs
                batch_size = embeddings.shape[0]
                sequence_ids.extend([f"seq_{batch_idx}_{i}" for i in range(batch_size)])
                
                if len(all_embeddings) % 100 == 0:
                    print(f"Processed {len(all_embeddings)} batches for embedding collection")
                
                # Check if we've collected enough samples
                total_samples = sum(emb.shape[0] for emb in all_embeddings)
                if max_samples and total_samples >= max_samples:
                    break
        
        # Concatenate all embeddings
        embeddings = torch.cat(all_embeddings, dim=0)  # (N, num_agents, d_model)
        
        # If we collected more samples than needed, randomly sample max_samples
        if max_samples and embeddings.shape[0] > max_samples:
            random_indices = random.sample(range(embeddings.shape[0]), max_samples)
            embeddings = embeddings[random_indices]
            # Also subsample metadata and sequence_ids accordingly
            combined_metadata = {}
            for key in all_metadata[0].keys():
                if key in ['sequence_length']:
                    combined_metadata[key] = all_metadata[0][key]
                else:
                    try:
                        full_data = torch.cat([m[key] for m in all_metadata if m[key] is not None], dim=0)
                        combined_metadata[key] = full_data[random_indices]
                    except:
                        full_data = [item for m in all_metadata for item in (m[key] if isinstance(m[key], list) else [m[key]])]
                        combined_metadata[key] = [full_data[i] for i in random_indices]
            sequence_ids = [sequence_ids[i] for i in random_indices]
        else:
            # Combine metadata as before
            combined_metadata = {}
            for key in all_metadata[0].keys():
                if key in ['sequence_length']:
                    combined_metadata[key] = all_metadata[0][key]
                else:
                    try:
                        combined_metadata[key] = torch.cat([m[key] for m in all_metadata if m[key] is not None], dim=0)
                    except:
                        combined_metadata[key] = [m[key] for m in all_metadata]

        if extract_per_team:
            print("Extractng per team")
            # Split into team 0 (players 0-10) + ball (index 22)
            team_0_indices = list(range(11)) + [22]  # [0,1,2,...,10,22]
            team_0 = embeddings[:, team_0_indices, :]  # Shape: [1000, 12, 128]

            # Split into team 1 (players 11-21) + ball (index 22)
            team_1_indices = list(range(11, 22)) + [22]  # [11,12,13,...,21,22]
            team_1 = embeddings[:, team_1_indices, :]  # Shape: [1000, 12, 128]

            # Concatenate along the batch dimension
            embeddings = torch.cat([team_0, team_1], dim=0)  # Shape: [2000, 12, 128]
        
        print(f"Collected embeddings shape: {embeddings.shape}") #here emb
        return embeddings, combined_metadata, sequence_ids


    def analyze_tactical_patterns_post_training(self, dataloader: DataLoader,
                                               save_analysis: bool = True,
                                               extract_per_team:bool=True):
        """
        Comprehensive post-training analysis for tactical pattern discovery
        
        This prepares data for clustering and interpretability modules
        """
        print("=== POST-TRAINING TACTICAL ANALYSIS ===")
        
        # 1. Collect embeddings from different layers for hierarchical analysis
        print("1. Collecting multi-layer embeddings...")
        layer_embeddings = {}
        layer_metadata = {}
        
        for layer_idx in [-3, -2, -1]:  # Last 3 layers
            embeddings, metadata, seq_ids = self.collect_embeddings_for_clustering(
                dataloader, layer_idx=layer_idx, max_samples=1000, #samples
                extract_per_team=extract_per_team
            )
            layer_embeddings[f'layer_{layer_idx}'] = embeddings
            layer_metadata[f'layer_{layer_idx}'] = metadata
        
        # 2. Collect attention patterns for interpretability
        print("2. Analyzing attention patterns...")
        attention_analysis = self.analyze_attention_patterns(dataloader, num_samples=100)
        
        # 3. Collect feature importance maps
        print("3. Computing feature importance...")
        importance_maps = []
        total_samples = 30
        chunk_size = 10

        for start in range(0, total_samples, chunk_size):
            print(f"Collecting {start}/{total_samples}")
            chunk = self.collect_feature_importance_chunk(dataloader, start, chunk_size)
            importance_maps.extend(chunk)
            del chunk
            gc.collect()

        importance_maps = self.collect_feature_importance(dataloader, num_samples=100)
        
        # 4. Prepare data for external clustering modules
        analysis_data = {
            'embeddings': layer_embeddings,
            'metadata': layer_metadata,
            'sequence_ids': seq_ids,
            'attention_patterns': attention_analysis,
            'feature_importance': importance_maps,
            'model_config': {
                'd_model': self.model.d_model,
                'num_agents': self.model.num_agents,
                'num_players': self.model.num_players,
                'forecast_horizon': self.model.forecast_horizon # have to deal with this
            }
        }
        
        if save_analysis:
            os.makedirs("analysis", exist_ok=True)
            torch.save(analysis_data, 'analysis/tactical_analysis_data.pt')
            print("Analysis data saved to 'analysis/tactical_analysis_data.pt'")
        
        return analysis_data

    def analyze_attention_patterns(self, dataloader: DataLoader, num_samples: int = 100):
        """Collect attention patterns for interpretability analysis"""
        self.model.eval()
        
        attention_patterns = {
            'ball_to_players': [],
            'players_to_ball': [],
            'inter_player': [],
            'team_formations': []
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= num_samples:
                    break
                    
                player_states = batch['player_states'].to(self.device)
                ball_states = batch['ball_states'].to(self.device)
                team_ids = batch.get('team_ids', None)
                if team_ids is not None:
                    team_ids = team_ids.to(self.device)
                
                # Forward pass to generate attention
                _ = self.model.forward(player_states, ball_states, team_ids)
                
                # Get attention from all layers
                all_attention = self.model.get_all_attention_weights()
                
                # Analyze patterns (simplified - you'll expand this)
                for layer_idx, attention in enumerate(all_attention):
                    if attention is not None:
                        # Extract ball-player attention patterns
                        # This is a simplified version - expand based on your attention structure
                        avg_attn = attention.mean(dim=(0, 1)).cpu()  # Average over batch and heads
                        attention_patterns['ball_to_players'].append(avg_attn[-1, :22].numpy())  # Ball to players
                        attention_patterns['players_to_ball'].append(avg_attn[:22, -1].numpy())  # Players to ball
        
        return attention_patterns

    def collect_feature_importance(self, dataloader: DataLoader, num_samples: int = 50):
        """Collect feature importance maps for SHAP/interpretability analysis"""
        importance_data = []
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_samples:
                break
                
            player_states = batch['player_states'].to(self.device)
            ball_states = batch['ball_states'].to(self.device)
            team_ids = batch.get('team_ids', None)
            if team_ids is not None:
                team_ids = team_ids.to(self.device)
            
            # Get importance maps
            importance_maps = self.model.get_feature_importance_maps(
                player_states, ball_states, team_ids
            )
            
            importance_data.append({
                'player_importance': importance_maps['player_gradients'].cpu() if importance_maps['player_gradients'] is not None else None,
                'ball_importance': importance_maps['ball_gradients'].cpu() if importance_maps['ball_gradients'] is not None else None,
                'attention_weights': [attn.cpu() for attn in importance_maps['attention_weights']],
                'predictions': {k: v.cpu() for k, v in importance_maps['predictions'].items()}
            })
        
        return importance_data
    

    def collect_feature_importance_chunk(self, dataloader, start_idx, chunk_size):
        """Collect a chunk of feature importance starting from `start_idx`."""
        importance_data = []

        for batch_idx, batch in enumerate(islice(dataloader, start_idx, start_idx + chunk_size)):
            player_states = batch['player_states'].to(self.device)
            ball_states = batch['ball_states'].to(self.device)
            team_ids = batch.get('team_ids', None)
            if team_ids is not None:
                team_ids = team_ids.to(self.device)

            importance_maps = self.model.get_feature_importance_maps(
                player_states, ball_states, team_ids
            )

            importance_data.append({
                'player_importance': importance_maps['player_gradients'].cpu() if importance_maps['player_gradients'] is not None else None,
                'ball_importance': importance_maps['ball_gradients'].cpu() if importance_maps['ball_gradients'] is not None else None,
                'attention_weights': [attn.cpu() for attn in importance_maps['attention_weights']],
                'predictions': {k: v.cpu() for k, v in importance_maps['predictions'].items()}
            })

        return importance_data
