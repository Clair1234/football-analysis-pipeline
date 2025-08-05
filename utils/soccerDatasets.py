import gc
import glob
import numpy as np
import polars as pl
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset

class SoccerPredictionDataset:
    def __init__(self, seq_len=100, forecast_horizon=10, max_position_jump=10.0, 
                 max_velocity_jump=5.0, filter_restarts=True):
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.max_position_jump = max_position_jump  # Maximum allowed position jump (meters)
        self.max_velocity_jump = max_velocity_jump  # Maximum allowed velocity jump
        self.filter_restarts = filter_restarts
        
        # Scalers
        self.position_scaler = MinMaxScaler()
        self.velocity_scaler = MinMaxScaler()
        self.distance_scaler = StandardScaler()
        self.density_scaler = StandardScaler()
        self.compactness_scaler = MinMaxScaler()
        self.spread_scaler = StandardScaler()
        self.centroid_scaler = MinMaxScaler()
        self.score_scaler = MinMaxScaler() #Should show which team is dominating but not by how much cause 0-1 would have the same scaling as 0-6, change depending on data
        
        self.data = None
        self.processed_data = None
        self.valid_indices = None  # Indices of time steps without discontinuities
        
        np.set_printoptions(precision=4)

    def detect_discontinuities(self, positions, velocities=None, ball_positions=None): #used
        """
        Detect discontinuities in player positions and ball positions
        
        Args:
            positions: (time_steps, num_players, 2) array of positions
            velocities: Optional (time_steps, num_players, 2) array of velocities
            ball_positions: Optional (time_steps, 2) array of ball positions
            
        Returns:
            valid_mask: Boolean array indicating valid time steps (no discontinuities)
        """
        time_steps, num_players, _ = positions.shape
        valid_mask = np.ones(time_steps, dtype=bool)
        
        if time_steps < 2:
            return valid_mask
            
        print("Detecting position discontinuities...")
        
        # Check position jumps
        position_diffs = np.diff(positions, axis=0)  # (time_steps-1, num_players, 2)
        position_distances = np.sqrt(np.sum(position_diffs**2, axis=2))  # (time_steps-1, num_players)
        
        # Find time steps where any player has a large position jump
        max_jumps_per_timestep = np.max(position_distances, axis=1)  # (time_steps-1,)
        position_discontinuities = max_jumps_per_timestep > self.max_position_jump
        
        # Mark the timestep after the jump as invalid (and optionally the timestep before)
        invalid_timesteps = np.where(position_discontinuities)[0] + 1  # +1 because diff reduces by 1
        valid_mask[invalid_timesteps] = False
        
        # Also mark the timestep before the jump as potentially invalid
        # invalid_timesteps_before = np.where(position_discontinuities)[0]
        # valid_mask[invalid_timesteps_before] = False
        
        print(f"Found {np.sum(position_discontinuities)} position discontinuities")
        
        # Check velocity jumps if available
        if velocities is not None:
            velocity_diffs = np.diff(velocities, axis=0)
            velocity_magnitudes = np.sqrt(np.sum(velocity_diffs**2, axis=2))
            max_vel_jumps = np.max(velocity_magnitudes, axis=1)
            velocity_discontinuities = max_vel_jumps > self.max_velocity_jump
            
            invalid_vel_timesteps = np.where(velocity_discontinuities)[0] + 1
            valid_mask[invalid_vel_timesteps] = False
            print(f"Found {np.sum(velocity_discontinuities)} velocity discontinuities")
        
        # Check ball position jumps if available
        if ball_positions is not None:
            ball_diffs = np.diff(ball_positions, axis=0)
            ball_distances = np.sqrt(np.sum(ball_diffs**2, axis=1))
            ball_discontinuities = ball_distances > self.max_position_jump
            
            invalid_ball_timesteps = np.where(ball_discontinuities)[0] + 1
            valid_mask[invalid_ball_timesteps] = False
            print(f"Found {np.sum(ball_discontinuities)} ball discontinuities")
        
        # Additional heuristics for restart detection
        if self.filter_restarts:
            valid_mask = self._detect_formation_resets(positions, valid_mask)
        
        print(f"Total valid timesteps: {np.sum(valid_mask)}/{len(valid_mask)} ({100*np.sum(valid_mask)/len(valid_mask):.1f}%)")
        
        return valid_mask

    def _detect_formation_resets(self, positions, valid_mask): #used
        """
        Detect formation resets by looking for sudden changes in team shape/spread
        """
        time_steps, num_players, _ = positions.shape
        
        # Calculate team centroids
        left_team_pos = positions[:, :11, :]  # First 11 players (left team)
        right_team_pos = positions[:, 11:, :]  # Last 11 players (right team)
        
        left_centroids = np.mean(left_team_pos, axis=1)  # (time_steps, 2)
        right_centroids = np.mean(right_team_pos, axis=1)
        
        # Calculate team spread (average distance from centroid)
        left_spreads = np.mean(np.sqrt(np.sum((left_team_pos - left_centroids[:, np.newaxis, :])**2, axis=2)), axis=1)
        right_spreads = np.mean(np.sqrt(np.sum((right_team_pos - right_centroids[:, np.newaxis, :])**2, axis=2)), axis=1)
        
        # Detect sudden changes in team spread
        if time_steps > 1:
            left_spread_changes = np.abs(np.diff(left_spreads))
            right_spread_changes = np.abs(np.diff(right_spreads))
            
            # Threshold for formation reset detection (adjust based on your field size)
            spread_threshold = 5.0  # meters
            
            formation_resets = np.logical_or(
                left_spread_changes > spread_threshold,
                right_spread_changes > spread_threshold
            )
            
            # Mark affected timesteps as invalid
            reset_timesteps = np.where(formation_resets)[0] + 1
            valid_mask[reset_timesteps] = False
            
            # print(f"Found {np.sum(formation_resets)} formation resets")
        
        return valid_mask

    def smooth_positions(self, positions, valid_mask, window_size=3): #used
        """
        Apply smoothing to positions, but only within valid segments
        """
        smoothed_positions = positions.copy()
        time_steps, num_players, _ = positions.shape
        
        # Find continuous valid segments
        valid_segments = self._find_continuous_segments(valid_mask)
        
        for start, end in valid_segments:
            if end - start > window_size:  # Only smooth segments longer than window
                segment = positions[start:end]
                
                # Apply simple moving average smoothing
                for i in range(window_size//2, len(segment) - window_size//2):
                    window_start = i - window_size//2
                    window_end = i + window_size//2 + 1
                    smoothed_positions[start + i] = np.mean(segment[window_start:window_end], axis=0)
        
        return smoothed_positions

    def _find_continuous_segments(self, valid_mask): #used
        """Find continuous segments of valid data"""
        segments = []
        start = None
        
        for i, is_valid in enumerate(valid_mask):
            if is_valid and start is None:
                start = i
            elif not is_valid and start is not None:
                segments.append((start, i))
                start = None
        
        # Handle case where valid segment goes to the end
        if start is not None:
            segments.append((start, len(valid_mask)))
        
        return segments
    
    def process_features(self): #used
        """Process features with memory optimization and discontinuity detection"""
        # Define player columns
        players = []
        for team in ["left", "right"]:
            for i in range(11):
                players.append((f"{team}_{i}_x", f"{team}_{i}_y"))
        
        # Select relevant columns
        relevant_cols = []
        for px, py in players:
            relevant_cols.extend([px, py])
        
        # Add ball columns
        relevant_cols.extend(["ball_x", "ball_y"])
        
        # Add team centroid columns
        relevant_cols.extend([
            "centroid_x_left_team", "centroid_y_left_team",
            "centroid_x_right_team", "centroid_y_right_team"
        ])
        
        # Add scores
        relevant_cols.extend(["score_left", "score_right"])
        
        # Add velocity columns if available
        velocity_cols = []
        for team in ["left", "right"]:
            for i in range(11):
                vel_col = f"{team}_{i}_velocity"
                if vel_col in self.data.columns:
                    velocity_cols.append(vel_col)
        
        if velocity_cols:
            relevant_cols.extend(velocity_cols)
            relevant_cols.append("ball_velocity")
        
        # Add distance to ball columns
        distance_cols = []
        for team in ["left", "right"]:
            for i in range(11):
                dist_col = f"distance_ball_to_player_{i}_{team}_team"
                if dist_col in self.data.columns:
                    distance_cols.append(dist_col)
        relevant_cols.extend(distance_cols)
        
        # Add zone columns
        zone_cols = []
        for team in ["left", "right"]:
            for i in range(11):
                zone_col = f"{team}_{i}_zone_id"
                if zone_col in self.data.columns:
                    zone_cols.append(zone_col)
        relevant_cols.extend(zone_cols)
        
        # Add tactical features
        tactical_cols = [
            "global_density", "right_team_density", "left_team_density",
            "global_compactness", "right_team_compactness", "left_team_compactness",
            "global_spread", "right_team_spread", "left_team_spread"
        ]
        available_tactical_cols = [col for col in tactical_cols if col in self.data.columns]
        relevant_cols.extend(available_tactical_cols)
        
        # Select and optimize data types with Polars
        # print("Selecting and optimizing data types...")
        
        # Create dtype mapping for memory optimization
        dtype_mapping = {}
        for col in relevant_cols:
            if col in self.data.columns:
                if any(x in col for x in ['_x', '_y', 'ball_x', 'ball_y', 'centroid']):
                    dtype_mapping[col] = pl.Float32  # Positions - float32 sufficient
                elif 'velocity' in col:
                    dtype_mapping[col] = pl.Float32  # Velocities - float32 sufficient  
                elif 'distance' in col:
                    dtype_mapping[col] = pl.Float32  # Distances - float32 sufficient
                elif 'score' in col:
                    dtype_mapping[col] = pl.Int16     # Scores - int16 sufficient
                elif 'zone' in col:
                    dtype_mapping[col] = pl.Int8      # Zones - int8 sufficient
                elif col in available_tactical_cols:
                    dtype_mapping[col] = pl.Float32   # Tactical features - float32 sufficient
        
        # Select and cast columns efficiently
        selected_data = self.data.select([
            pl.col(col).cast(dtype_mapping.get(col, pl.Float32)) 
            for col in relevant_cols if col in self.data.columns
        ])

        # Convert to numpy with optimized dtypes - process in chunks to save memory
        time_steps = selected_data.height
        num_players = 22
        
        # print(f"Processing {time_steps} time steps for {num_players} players")
        # print(f"Available tactical features: {available_tactical_cols}")
        
        # Process data in chunks to reduce memory usage
        chunk_size = min(10000, time_steps)  # Process 10k rows at a time
        
        # Initialize lists to collect results
        all_player_positions = []
        all_player_velocities = []
        all_player_distances = []
        all_player_zones = []
        all_player_tactical_features = []
        all_ball_states = []
        
        for chunk_start in range(0, time_steps, chunk_size):
            chunk_end = min(chunk_start + chunk_size, time_steps)
            print(f"Processing chunk {chunk_start}:{chunk_end}")
            
            # Get chunk as pandas DataFrame for easier processing
            chunk_df = selected_data.slice(chunk_start, chunk_end - chunk_start).to_pandas()
            
            # Process this chunk
            chunk_positions, chunk_velocities, chunk_distances, chunk_zones, \
            chunk_tactical, chunk_ball = self._process_chunk(
                chunk_df, available_tactical_cols, chunk_start, velocity_cols, distance_cols, zone_cols
            )
            
            # Collect results
            all_player_positions.append(chunk_positions)
            all_player_velocities.append(chunk_velocities)
            all_player_distances.append(chunk_distances)
            all_player_zones.append(chunk_zones)
            all_player_tactical_features.append(chunk_tactical)
            all_ball_states.append(chunk_ball)
            
            # Clean up chunk data
            del chunk_df
            gc.collect()
        
        # Concatenate all chunks with optimized dtypes
        self.player_positions = np.concatenate(all_player_positions, axis=0).astype(np.float32)
        self.player_velocities = np.concatenate(all_player_velocities, axis=0).astype(np.float32)
        self.player_distances = np.concatenate(all_player_distances, axis=0).astype(np.float32)
        self.player_zones = np.concatenate(all_player_zones, axis=0).astype(np.int8)
        self.player_tactical_features = np.concatenate(all_player_tactical_features, axis=0).astype(np.float32)
        self.ball_states = np.concatenate(all_ball_states, axis=0).astype(np.float32)
        
        # Clean up temporary data
        del all_player_positions, all_player_velocities, all_player_distances
        del all_player_zones, all_player_tactical_features, all_ball_states
        del selected_data
        gc.collect()
        
        print(f"Player positions shape: {self.player_positions.shape} (dtype: {self.player_positions.dtype})")
        print(f"Player velocities shape: {self.player_velocities.shape} (dtype: {self.player_velocities.dtype})")
        print(f"Player distances shape: {self.player_distances.shape} (dtype: {self.player_distances.dtype})")
        print(f"Player zones shape: {self.player_zones.shape} (dtype: {self.player_zones.dtype})")
        print(f"Player tactical features shape: {self.player_tactical_features.shape} (dtype: {self.player_tactical_features.dtype})")
        print(f"Ball states shape: {self.ball_states.shape} (dtype: {self.ball_states.dtype})")
        
        # ===== DISCONTINUITY DETECTION =====
        print("\n=== Starting Discontinuity Detection ===")
        
        # Extract velocities for discontinuity detection if available
        velocities_for_detection = None
        if len(velocity_cols) > 0:
            velocities_for_detection = self.player_velocities
        
        # Detect discontinuities
        self.valid_indices = self.detect_discontinuities(
            self.player_positions, 
            velocities=velocities_for_detection,
            ball_positions=self.ball_states[:, :2]  # First 2 columns are ball positions
        )
        
        # Optionally smooth positions within valid segments
        if hasattr(self, 'smooth_data') and getattr(self, 'smooth_data', False):
            print("Applying position smoothing...")
            self.player_positions = self.smooth_positions(self.player_positions, self.valid_indices)
            
            # Also smooth ball positions
            ball_positions_only = self.ball_states[:, :2]
            smoothed_ball_pos = self.smooth_positions(
                ball_positions_only.reshape(ball_positions_only.shape[0], 1, 2), 
                self.valid_indices
            ).reshape(ball_positions_only.shape)
            self.ball_states[:, :2] = smoothed_ball_pos
        
        # Scale features
        self._scale_features()
        
        print(f"\nProcessing complete!")
        print(f"Valid timesteps: {np.sum(self.valid_indices)}/{len(self.valid_indices)} ({100*np.sum(self.valid_indices)/len(self.valid_indices):.1f}%)")
        print("=== Discontinuity Detection Complete ===\n")

    def create_sequences_filtered(self): #used
        """Create sequences while filtering out discontinuities"""
        print("create_sequences_filtered used")
        print("Creating sequences with discontinuity filtering...")
        
        if self.valid_indices is None:
            print("Warning: No discontinuity detection performed. Using all data.")
            return self.create_sequences()
        
        # Find continuous segments of valid data
        valid_segments = self._find_continuous_segments(self.valid_indices)
        
        print(f"Found {len(valid_segments)} continuous segments")
        for i, (start, end) in enumerate(valid_segments):
            print(f"  Segment {i+1}: timesteps {start}-{end} (length: {end-start})")
        
        # Collect sequences from each valid segment
        all_X_player = []
        all_X_ball = []
        all_y_player = []
        all_y_ball = []
        
        # Prepare combined player states (use your original logic)
        player_zones_expanded = self.player_zones.astype(np.float32)[..., np.newaxis]
        player_distances_expanded = self.player_distances[..., np.newaxis]
        
        player_states = np.concatenate([
            self.player_positions,
            self.player_velocities,
            player_distances_expanded,
            player_zones_expanded,
            self.player_tactical_features
        ], axis=2, dtype=np.float32)
        
        total_sequences = 0
        
        for start, end in valid_segments:
            segment_length = end - start
            
            # Check if segment is long enough for sequences
            if segment_length < self.seq_len + 1: # 1 for prediction
                print(f"Skipping segment {start}-{end}: too short ({segment_length} < {self.seq_len + 1})")
                continue
            
            # Extract segment data
            segment_player_states = player_states[start:end]
            segment_ball_states = self.ball_states[start:end]
            segment_player_positions = self.player_positions[start:end]
            
            # Create sequences within this segment
            num_sequences = segment_length - self.seq_len #- self.forecast_horizon + 1
            
            for i in range(num_sequences):
                seq_start = i
                input_end = seq_start + self.seq_len
                target_end = input_end + 1 #self.forecast_horizon
                
                X_player_seq = segment_player_states[seq_start:input_end]
                X_ball_seq = segment_ball_states[seq_start:input_end]
                y_player_seq = segment_player_positions[input_end:target_end]
                y_ball_seq = segment_ball_states[input_end:target_end, :2]
                
                all_X_player.append(X_player_seq)
                all_X_ball.append(X_ball_seq)
                all_y_player.append(y_player_seq)
                all_y_ball.append(y_ball_seq)
                
                total_sequences += 1
        
        if total_sequences == 0:
            raise ValueError("No valid sequences found after filtering discontinuities")
        
        # Convert to numpy arrays
        X_player = np.array(all_X_player, dtype=np.float32)
        X_ball = np.array(all_X_ball, dtype=np.float32)
        y_player = np.array(all_y_player, dtype=np.float32)
        y_ball = np.array(all_y_ball, dtype=np.float32)
        
        print(f"Created {total_sequences} filtered sequences:")
        print(f"  X_player: {X_player.shape}")
        print(f"  X_ball: {X_ball.shape}")
        print(f"  y_player: {y_player.shape}")
        print(f"  y_ball: {y_ball.shape}")
        
        return X_player, X_ball, y_player, y_ball

    def analyze_discontinuities(self):
        """Analyze and report discontinuities in the data"""
        if self.valid_indices is None:
            print("No discontinuity analysis available. Run process_features() first.")
            return
        
        total_timesteps = len(self.valid_indices)
        valid_timesteps = np.sum(self.valid_indices)
        invalid_timesteps = total_timesteps - valid_timesteps
        
        print(f"\n=== Discontinuity Analysis ===")
        print(f"Total timesteps: {total_timesteps}")
        print(f"Valid timesteps: {valid_timesteps} ({100*valid_timesteps/total_timesteps:.1f}%)")
        print(f"Invalid timesteps: {invalid_timesteps} ({100*invalid_timesteps/total_timesteps:.1f}%)")
        
        # Find continuous segments
        segments = self._find_continuous_segments(self.valid_indices)
        print(f"\nContinuous segments: {len(segments)}")
        
        segment_lengths = [end - start for start, end in segments]
        if segment_lengths:
            print(f"Average segment length: {np.mean(segment_lengths):.1f} timesteps")
            print(f"Longest segment: {max(segment_lengths)} timesteps")
            print(f"Shortest segment: {min(segment_lengths)} timesteps")
            
            # Count usable segments
            usable_segments = [length for length in segment_lengths 
                             if length >= self.seq_len + 1] #1
            print(f"Segments usable for sequences: {len(usable_segments)}/{len(segments)}")
            
    def load_data(self, parquet_folder_path):
        """Load parquet data"""
        self.data = self.spark.read.parquet(parquet_folder_path)
        print("Loaded data schema:")
        self.data.printSchema()

    def load_specific(self, parquet_files):
        """Load limited parquet files more efficiently using Polars""" 
        # Use Polars for more memory efficient loading
        if len(parquet_files) == 1:
            self.data = pl.read_parquet(parquet_files[0])
        else:
            # Read and concatenate efficiently
            dataframes = []
            for file in parquet_files:
                df = pl.read_parquet(file)
                dataframes.append(df)
            self.data = pl.concat(dataframes)
            del dataframes  # Free memory immediately
            gc.collect()
        
        # Add match_id
        self.data = self.data.with_row_index("match_id")
        
        #print(f"Data shape: {self.data.shape}")
        #print("Data schema:")
        #print(self.data.dtypes)
        
    def load_data_limited(self, folder_path, max_files=100): #used
        """Load limited parquet files more efficiently using Polars"""
        
        # Get parquet files directly
        parquet_files = glob.glob(os.path.join(folder_path, "*.parquet"))[:max_files]
        print(f"Loading {len(parquet_files)} files with Polars")
        
        # Use Polars for more memory efficient loading
        if len(parquet_files) == 1:
            self.data = pl.read_parquet(parquet_files[0])
        else:
            # Read and concatenate efficiently
            dataframes = []
            for file in parquet_files:
                df = pl.read_parquet(file)
                dataframes.append(df)
            self.data = pl.concat(dataframes)
            del dataframes  # Free memory immediately
            gc.collect()
        
        # Add match_id
        self.data = self.data.with_row_index("match_id")
        
        print(f"Data shape: {self.data.shape}")
        print("Data schema:")
        print(self.data.dtypes)

    def _process_chunk(self, pdf, available_tactical_cols, chunk_offset, velocity_cols, distance_cols, zone_cols): #used
        """Process a chunk of data - helper function for process_features"""
        time_steps = len(pdf)
        
        # Initialize arrays for this chunk
        player_positions = []
        player_velocities = []  
        player_distances = []
        player_zones = []
        player_tactical_features = []
        ball_states = []
        
        for t in range(time_steps):
            # Extract tactical features for this timestep
            tactical_data = {}
            for col in available_tactical_cols:
                tactical_data[col] = pdf.iloc[t][col]
            
            # Get centroids and scores
            left_centroid = [pdf.iloc[t]["centroid_x_left_team"], pdf.iloc[t]["centroid_y_left_team"]]
            right_centroid = [pdf.iloc[t]["centroid_x_right_team"], pdf.iloc[t]["centroid_y_right_team"]]
            score_left = pdf.iloc[t]["score_left"]
            score_right = pdf.iloc[t]["score_right"]
            
            positions = []
            velocities = []
            distances = []
            zones = []
            tactical_features = []
            
            for team in ["left", "right"]:
                # Determine team-specific features
                if team == "left":
                    own_centroid = left_centroid
                    opp_centroid = right_centroid
                    own_score = score_left
                    opp_score = score_right
                    team_density = tactical_data.get("left_team_density", 0)
                    team_compactness = tactical_data.get("left_team_compactness", 0)
                    team_spread = tactical_data.get("left_team_spread", 0)
                    opp_density = tactical_data.get("right_team_density", 0)
                    opp_compactness = tactical_data.get("right_team_compactness", 0)
                    opp_spread = tactical_data.get("right_team_spread", 0)
                else:
                    own_centroid = right_centroid
                    opp_centroid = left_centroid
                    own_score = score_right
                    opp_score = score_left
                    team_density = tactical_data.get("right_team_density", 0)
                    team_compactness = tactical_data.get("right_team_compactness", 0)
                    team_spread = tactical_data.get("right_team_spread", 0)
                    opp_density = tactical_data.get("left_team_density", 0)
                    opp_compactness = tactical_data.get("left_team_compactness", 0)
                    opp_spread = tactical_data.get("left_team_spread", 0)
                
                for i in range(11):
                    # Positions
                    px, py = f"{team}_{i}_x", f"{team}_{i}_y"
                    positions.append([pdf.iloc[t][px], pdf.iloc[t][py]])
                    
                    # Velocity
                    vel_col = f"{team}_{i}_velocity"
                    if vel_col in pdf.columns:
                        velocities.append([pdf.iloc[t][vel_col], 0])
                    elif t > 0 and chunk_offset + t > 0:  # Can compute velocity
                        prev_pos = [pdf.iloc[t-1][px], pdf.iloc[t-1][py]]
                        curr_pos = [pdf.iloc[t][px], pdf.iloc[t][py]]
                        vel = [curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1]]
                        velocities.append(vel)
                    else:
                        velocities.append([0.0, 0.0])
                    
                    # Distance to ball
                    dist_col = f"distance_ball_to_player_{i}_{team}_team"
                    if dist_col in pdf.columns:
                        distances.append(pdf.iloc[t][dist_col])
                    else:
                        ball_x, ball_y = pdf.iloc[t]["ball_x"], pdf.iloc[t]["ball_y"]
                        player_x, player_y = pdf.iloc[t][px], pdf.iloc[t][py]
                        dist = np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)
                        distances.append(dist)
                    
                    # Zone
                    zone_col = f"{team}_{i}_zone_id"
                    if zone_col in pdf.columns:
                        zones.append(int(pdf.iloc[t][zone_col]))  # Ensure integer
                    else:
                        zones.append(0)
                    
                    # Tactical features
                    player_tactical = [
                        own_centroid[0], own_centroid[1],
                        team_density, team_compactness, team_spread,
                        opp_centroid[0], opp_centroid[1], 
                        opp_density, opp_compactness, opp_spread,
                        tactical_data.get("global_density", 0),
                        tactical_data.get("global_compactness", 0),
                        tactical_data.get("global_spread", 0),
                        own_score, opp_score, own_score - opp_score
                    ]
                    tactical_features.append(player_tactical)
            
            # Convert to numpy with appropriate dtypes
            positions = np.array(positions, dtype=np.float32)
            velocities = np.array(velocities, dtype=np.float32)
            distances = np.array(distances, dtype=np.float32)  
            zones = np.array(zones, dtype=np.int8)
            tactical_features = np.array(tactical_features, dtype=np.float32)
            
            player_positions.append(positions)
            player_velocities.append(velocities)
            player_distances.append(distances)
            player_zones.append(zones)
            player_tactical_features.append(tactical_features)
            
            # Ball state
            ball_pos = [pdf.iloc[t]["ball_x"], pdf.iloc[t]["ball_y"]]
            if "ball_velocity" in pdf.columns:
                ball_vel = [pdf.iloc[t]["ball_velocity"], 0]
            elif t > 0:
                prev_ball = [pdf.iloc[t-1]["ball_x"], pdf.iloc[t-1]["ball_y"]]
                ball_vel = [ball_pos[0] - prev_ball[0], ball_pos[1] - prev_ball[1]]
            else:
                ball_vel = [0.0, 0.0]
            
            ball_states.append(ball_pos + ball_vel)
        
        # Convert lists to numpy arrays with optimized dtypes
        chunk_positions = np.array(player_positions, dtype=np.float32)
        chunk_velocities = np.array(player_velocities, dtype=np.float32)
        chunk_distances = np.array(player_distances, dtype=np.float32)
        chunk_zones = np.array(player_zones, dtype=np.int8)
        chunk_tactical = np.array(player_tactical_features, dtype=np.float32)
        chunk_ball = np.array(ball_states, dtype=np.float32)
        
        return chunk_positions, chunk_velocities, chunk_distances, chunk_zones, chunk_tactical, chunk_ball

    def _scale_features(self): #used
        """Scale features with memory optimization"""
        print("Scaling features...")
        
        # Scale positions in-place to save memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        original_shape = self.player_positions.shape
        self.player_positions = self.player_positions.reshape(-1, 2)
        self.player_positions = self.position_scaler.fit_transform(self.player_positions).astype(np.float32)
        self.player_positions = self.player_positions.reshape(original_shape)
        
        # Scale velocities in-place
        original_shape = self.player_velocities.shape
        self.player_velocities = self.player_velocities.reshape(-1, 2)
        self.player_velocities = self.velocity_scaler.fit_transform(self.player_velocities).astype(np.float32)
        self.player_velocities = self.player_velocities.reshape(original_shape)
        
        # Scale distances in-place
        original_shape = self.player_distances.shape
        flat_distances = self.player_distances.reshape(-1, 1)
        scaled_distances = self.distance_scaler.fit_transform(flat_distances).astype(np.float32)
        self.player_distances = scaled_distances.reshape(original_shape)
        del flat_distances, scaled_distances
        
        # Scale tactical features with chunking for memory efficiency
        original_shape = self.player_tactical_features.shape
        chunk_size = 100000  # Process 100k rows at a time
        
        # Fit scalers on a sample first
        sample_size = min(10000, original_shape[0] * original_shape[1])
        sample_tactical = self.player_tactical_features.reshape(-1, 16)[:sample_size]
        
        # Fit scalers
        centroid_sample = np.column_stack([sample_tactical[:, 0:2], sample_tactical[:, 5:7]])
        self.centroid_scaler.fit(centroid_sample)
        self.density_scaler.fit(sample_tactical[:, [2, 7, 10]])
        self.compactness_scaler.fit(sample_tactical[:, [3, 8, 11]])
        self.spread_scaler.fit(sample_tactical[:, [4, 9, 12]])
        self.score_scaler.fit(sample_tactical[:, [13, 14, 15]])
        
        del sample_tactical, centroid_sample
        gc.collect()

        # Save scaler parameters
        self.position_scaler_torch = {
            "min": torch.tensor(self.position_scaler.data_min_, dtype=torch.float32, device=device),
            "max": torch.tensor(self.position_scaler.data_max_, dtype=torch.float32, device=device)
        }
        
        # Transform in chunks
        flat_tactical = self.player_tactical_features.reshape(-1, 16)
        
        for start_idx in range(0, len(flat_tactical), chunk_size):
            end_idx = min(start_idx + chunk_size, len(flat_tactical))
            chunk = flat_tactical[start_idx:end_idx]
            
            # Scale different feature types
            centroid_features = np.column_stack([chunk[:, 0:2], chunk[:, 5:7]])
            scaled_centroids = self.centroid_scaler.transform(centroid_features).astype(np.float32)
            #print(chunk[:, [13, 14, 15]])
            scaled_density = self.density_scaler.transform(chunk[:, [2, 7, 10]]).astype(np.float32)
            scaled_compactness = self.compactness_scaler.transform(chunk[:, [3, 8, 11]]).astype(np.float32)
            scaled_spread = self.spread_scaler.transform(chunk[:, [4, 9, 12]]).astype(np.float32)
            scaled_scores = self.score_scaler.transform(chunk[:, [13, 14, 15]]).astype(np.float32)
            #print(scaled_scores.min())
            
            # Reconstruct scaled chunk
            chunk[:, 0:2] = scaled_centroids[:, 0:2]
            chunk[:, 2] = scaled_density[:, 0] 
            chunk[:, 3] = scaled_compactness[:, 0]
            chunk[:, 4] = scaled_spread[:, 0]
            chunk[:, 5:7] = scaled_centroids[:, 2:4]
            chunk[:, 7] = scaled_density[:, 1]
            chunk[:, 8] = scaled_compactness[:, 1]
            chunk[:, 9] = scaled_spread[:, 1]
            chunk[:, 10] = scaled_density[:, 2]
            chunk[:, 11] = scaled_compactness[:, 2]
            chunk[:, 12] = scaled_spread[:, 2]
            chunk[:, 13:16] = scaled_scores
            
            flat_tactical[start_idx:end_idx] = chunk.astype(np.float32)
            
            del centroid_features, scaled_centroids, scaled_density, scaled_compactness, scaled_spread, scaled_scores
        
        self.player_tactical_features = flat_tactical.reshape(original_shape)
        del flat_tactical
        
        # Scale ball states
        self.ball_states[:, :2] = self.position_scaler.transform(self.ball_states[:, :2]).astype(np.float32)
        if self.ball_states.shape[1] > 2:
            self.ball_states[:, 2:] = self.velocity_scaler.transform(self.ball_states[:, 2:]).astype(np.float32)
        
        gc.collect()
        print("Feature scaling completed")

    def create_sequences(self): #used
        """Create sequences with memory optimization"""
        print("Creating sequences with memory optimization...")
        
        # Convert zones to expanded format efficiently
        player_zones_expanded = self.player_zones.astype(np.float32)[..., np.newaxis]
        player_distances_expanded = self.player_distances[..., np.newaxis]
        
        # Combine player states with memory-efficient concatenation
        player_states = np.concatenate([
            self.player_positions,          # (time_steps, 22, 2)
            self.player_velocities,         # (time_steps, 22, 2)  
            player_distances_expanded,      # (time_steps, 22, 1)
            player_zones_expanded,          # (time_steps, 22, 1)
            self.player_tactical_features   # (time_steps, 22, 16)
        ], axis=2, dtype=np.float32)  # (time_steps, 22, 22)
        
        # Clean up intermediate arrays
        del player_zones_expanded, player_distances_expanded
        gc.collect()
        
        total_steps = player_states.shape[0]
        num_sequences = total_steps - self.seq_len #- self.forecast_horizon + 1
        
        if num_sequences <= 0:
            raise ValueError(f"Not enough data for sequences. Need at least {self.seq_len + 1} steps, got {total_steps}")
        
        print(f"Creating {num_sequences} sequences...")
        
        # Pre-allocate arrays with correct dtypes
        X_player = np.empty((num_sequences, self.seq_len, 22, 22), dtype=np.float32)
        X_ball = np.empty((num_sequences, self.seq_len, 4), dtype=np.float32)
        y_player = np.empty((num_sequences, 1, 22, 2), dtype=np.float32)
        y_ball = np.empty((num_sequences, 1, 2), dtype=np.float32)
        
        # Fill arrays efficiently
        for i in range(num_sequences):
            start = i
            input_end = start + self.seq_len
            target_end = input_end + 1#self.forecast_horizon
            
            X_player[i] = player_states[start:input_end]
            X_ball[i] = self.ball_states[start:input_end]
            y_player[i] = self.player_positions[input_end:target_end]
            y_ball[i] = self.ball_states[input_end:target_end, :2]
            
            # Periodic garbage collection for large datasets
            if i % 1000 == 0 and i > 0:
                gc.collect()
        
        # Clean up player_states
        del player_states
        gc.collect()
        
        print(f"Created sequences:")
        print(f"  X_player: {X_player.shape} (dtype: {X_player.dtype})")
        print(f"  X_ball: {X_ball.shape} (dtype: {X_ball.dtype})")
        print(f"  y_player: {y_player.shape} (dtype: {y_player.dtype})")
        print(f"  y_ball: {y_ball.shape} (dtype: {y_ball.dtype})")
        
        return X_player, X_ball, y_player, y_ball

    def inverse_transform_predictions(self, predictions):
        """
        Inverse transform predictions back to original scale
        
        Args:
            predictions: Either player predictions (batch, num_players, horizon, 2) 
                        or ball predictions (batch, horizon, 2)
        Returns:
            Unscaled predictions in same shape
        """
        print("inverse_transform_predictions used")
        original_shape = predictions.shape
        
        if len(predictions.shape) == 4:  # Player predictions
            # Reshape to (batch * num_players * horizon, 2)
            reshaped = predictions.reshape(-1, 2)
            unscaled = self.position_scaler.inverse_transform(reshaped)
            return unscaled.reshape(original_shape)
        else:  # Ball predictions (batch, horizon, 2)
            # Reshape to (batch * horizon, 2) for inverse transform
            reshaped = predictions.reshape(-1, 2)
            unscaled = self.position_scaler.inverse_transform(reshaped)
            return unscaled.reshape(original_shape)

    def inverse_transform_predictions_tensor(self, x: torch.Tensor, kind: str = "position") -> torch.Tensor:
        """
        Inverse transform normalized predictions without leaving the GPU or autograd.
        Args:
            x: normalized predictions (torch.Tensor)
            kind: either "position" or "velocity"
        Returns:
            unnormalized predictions (torch.Tensor)
        """
        print("inverse_transform_predictions_tensor used")
        if kind == "position":
            scaler = self.position_scaler_torch
            x_unscaled = x * (scaler["max"] - scaler["min"]) + scaler["min"]
        elif kind == "velocity":
            scaler = self.velocity_scaler_torch
            x_unscaled = x * scaler["scale"] + scaler["mean"]
        else:
            raise ValueError(f"Unknown kind '{kind}'. Use 'position' or 'velocity'.")
        
        return x_unscaled

    def get_feature_info(self): #used
        """
        Get information about the features in the player states
        Not ideal that this is hard coded but it does the job
        """
        feature_info = {
            'total_features_per_player': 22,
            'feature_breakdown': {
                'position_x': 0,
                'position_y': 1,
                'velocity_x': 2,
                'velocity_y': 3,
                'distance_to_ball': 4,
                'zone_id': 5,
                'own_centroid_x': 6,
                'own_centroid_y': 7,
                'own_team_density': 8,
                'own_team_compactness': 9,
                'own_team_spread': 10,
                'opp_centroid_x': 11,
                'opp_centroid_y': 12,
                'opp_team_density': 13,
                'opp_team_compactness': 14,
                'opp_team_spread': 15,
                'global_density': 16,
                'global_compactness': 17,
                'global_spread': 18,
                'own_score': 19,
                'opp_score': 20,
                'score_difference': 21
            },
            'feature_groups': {
                'basic': ['position_x', 'position_y', 'velocity_x', 'velocity_y'],
                'spatial': ['distance_to_ball', 'zone_id'],
                'team_context': ['own_centroid_x', 'own_centroid_y', 'own_team_density', 'own_team_compactness', 'own_team_spread'],
                'opponent_context': ['opp_centroid_x', 'opp_centroid_y', 'opp_team_density', 'opp_team_compactness', 'opp_team_spread'],
                'global_context': ['global_density', 'global_compactness', 'global_spread'],
                'match_context': ['own_score', 'opp_score', 'score_difference']
            }
        }
        return feature_info

class SoccerSequenceDataset(Dataset):
    """Enhanced version of SoccerSequenceDataset """
    def __init__(self, X_player, X_ball, y_player, y_ball, team_ids=None):
        self.X_player = torch.FloatTensor(X_player)
        self.X_ball = torch.FloatTensor(X_ball)
        self.y_player = torch.FloatTensor(y_player)
        self.y_ball = torch.FloatTensor(y_ball)
    
    def __len__(self):
        return len(self.X_player)
    
    def __getitem__(self, idx):
        return {
            'player_states': self.X_player[idx],     # (seq_len, 22, 22)
            'ball_states': self.X_ball[idx],         # (seq_len, 4)
            'target_players': self.y_player[idx],    # (forecast_horizon, 22, 2)
            'target_ball': self.y_ball[idx],         # (forecast_horizon, 2)
        }