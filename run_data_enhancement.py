import glob
import pandas as pd
import numpy as np
import os
from scipy.spatial import ConvexHull


def calculate_centroid(row):
    left_team_x = []
    right_team_x = []
    left_team_y = []
    right_team_y = []
    
    for player in range(11):
        left_team_x.append(row[f"left_{player}_x"])
        left_team_y.append(row[f"left_{player}_y"])
        right_team_x.append(row[f"left_{player}_x"])
        right_team_y.append(row[f"left_{player}_y"])
        
    centroid_left_x = np.mean(left_team_x)
    centroid_left_y = np.mean(left_team_y)
    centroid_right_x = np.mean(right_team_x)
    centroid_right_y = np.mean(right_team_y)
    
    return {centroid_left_x, 
            centroid_left_y, 
            centroid_right_x, 
            centroid_right_y}
    
def calculate_distance_player_to_ball(data, player_id, side):
    distance_x = data['ball_x'] - data[f"{side}_{player_id}_x"]
    distance_y = data['ball_y'] - data[f"{side}_{player_id}_y"]
    return np.sqrt(distance_x**2 + distance_y**2)

def assign_zone(x_series, y_series, field_width=1.0, field_height=1.0, num_cols=5, num_rows=3):
    col_width = field_width / num_cols
    row_height = field_height / num_rows
    
    # Convert from [-1, 1] to [0, field_width]
    x_norm = (x_series + 1) * (field_width / 2)
    y_norm = (y_series + 1) * (field_height / 2)

    col = (x_norm // col_width).clip(upper=num_cols - 1).astype(int)
    row = (y_norm // row_height).clip(upper=num_rows - 1).astype(int)

    return row * num_cols + col  # zone ID in [0, num_cols * num_rows - 1]

def compute_density(all_players):
    num_players = all_players.shape[0]
    bounding_box_area = (max(all_players[:, 0]) - min(all_players[:, 0])) * (max(all_players[:, 1]) - min(all_players[:, 1]))
    
    # Avoid division by zero
    if bounding_box_area > 0:
        density = num_players / bounding_box_area
    else:
        density = 0
    
    return density, num_players, bounding_box_area

def compute_compactness(area_convex_hull, bounding_box_area):
    # Avoid division by zero
    if bounding_box_area > 0:
        return area_convex_hull / bounding_box_area
    else:
        return 0

def compute_spread(all_players):
    num_players = all_players.shape[0]
    
    # Avoid division by zero
    if num_players > 1:
        distances = np.linalg.norm(all_players[:, np.newaxis] - all_players[np.newaxis, :], axis=2)
        spread = np.sum(distances) / (num_players * (num_players - 1))
    else:
        spread = 0
    
    return spread

def compute_metrics_per_frame(data):
    # Initialize empty lists to hold the metrics
    global_densities = []
    global_compactnesses = []
    global_spreads = []
    
    left_team_densities = []
    left_team_compactnesses = []
    left_team_spreads = []
    
    right_team_densities = []
    right_team_compactnesses = []
    right_team_spreads = []

    # Extract player coordinates
    left_team_coords = []
    right_team_coords = []

    for player in range(11):
        left_team_coords.append([data[f"left_{player}_x"].values, data[f"left_{player}_y"].values])
        right_team_coords.append([data[f"right_{player}_x"].values, data[f"right_{player}_y"].values])

    # Convert lists to numpy arrays and transpose to get the desired shape
    left_team_coords = np.array(left_team_coords).transpose(2, 0, 1)  # Shape: (frames, players, 2)
    right_team_coords = np.array(right_team_coords).transpose(2, 0, 1)  # Shape: (frames, players, 2)

    # Loop through each frame to calculate metrics
    num_frames = left_team_coords.shape[0]

    for frame in range(num_frames):
        # Get coordinates for the current frame
        left_coords = left_team_coords[frame]
        right_coords = right_team_coords[frame]
        
        # Combine both teams' coordinates for global metrics
        all_players = np.vstack((left_coords, right_coords))
        
        # Compute global density
        global_density, num_players, bounding_box_area = compute_density(all_players)
        global_densities.append(global_density)
        
        # Compute convex hull for global compactness
        if num_players >= 3:  # Convex hull requires at least 3 points
            hull = ConvexHull(all_players)
            area_convex_hull = hull.volume
            global_compactness = compute_compactness(area_convex_hull, bounding_box_area)
        else:
            global_compactness = 0
        
        global_compactnesses.append(global_compactness)
        
        # Compute global spread
        global_spread = compute_spread(all_players)
        global_spreads.append(global_spread)

        # Compute metrics for the left team
        left_density, left_num_players, left_bounding_box_area = compute_density(left_coords)
        left_team_densities.append(left_density)
        
        if left_num_players >= 3:
            left_hull = ConvexHull(left_coords)
            left_area_convex_hull = left_hull.volume
            left_compactness = compute_compactness(left_area_convex_hull, left_bounding_box_area)
        else:
            left_compactness = 0
        
        left_team_compactnesses.append(left_compactness)
        left_team_spread = compute_spread(left_coords)
        left_team_spreads.append(left_team_spread)

        # Compute metrics for the right team
        right_density, right_num_players, right_bounding_box_area = compute_density(right_coords)
        right_team_densities.append(right_density)
        
        if right_num_players >= 3:
            right_hull = ConvexHull(right_coords)
            right_area_convex_hull = right_hull.volume
            right_compactness = compute_compactness(right_area_convex_hull, right_bounding_box_area)
        else:
            right_compactness = 0
        
        right_team_compactnesses.append(right_compactness)
        right_team_spread = compute_spread(right_coords)
        right_team_spreads.append(right_team_spread)

    # Convert lists to numpy arrays for easier handling
    global_densities = np.array(global_densities)
    global_compactnesses = np.array(global_compactnesses)
    global_spreads = np.array(global_spreads)

    left_team_densities = np.array(left_team_densities)
    left_team_compactnesses = np.array(left_team_compactnesses)
    left_team_spreads = np.array(left_team_spreads)

    right_team_densities = np.array(right_team_densities)
    right_team_compactnesses = np.array(right_team_compactnesses)
    right_team_spreads = np.array(right_team_spreads)

    return (global_densities, global_compactnesses, global_spreads,
            left_team_densities, left_team_compactnesses, left_team_spreads,
            right_team_densities, right_team_compactnesses, right_team_spreads)

def main():
    print("Dealing with raw data ingestion...")
    
    # Gather data from csv
    path = './data/raw/'  
    dir_list = os.listdir(path)

    for file in dir_list:
        data = pd.read_csv(f"{path}{file}")
        
        # Format time-serie data into consistent tensor form
        # Tensor shape
        batch_size = 100
        sequence_length = 20
        feature_dimension = 47 # 11 * 2 * 2 for players in 2D + 3 for ball
        
        # Switch to field dimensions
        def rescale_series(series, field_dim, input_min=-1, input_max=1):
            return ((series - input_min) / (input_max - input_min)) * field_dim
        columns_to_fields_x = ['ball_x']
        columns_to_fields_y = ['ball_y']
        for player in range(11):
            for side in ['left', 'right']:    
                columns_to_fields_x.append(f'{side}_{player}_x')
                columns_to_fields_y.append(f'{side}_{player}_y')
        # Apply to X columns (e.g., from [-1, 1] → [0, 105])
        data[columns_to_fields_x] = data[columns_to_fields_x].apply(
            lambda col: rescale_series(col, field_dim=105, input_min=-1, input_max=1)
        )

        # Apply to Y columns (e.g., from [-0.42, 0.42] → [0, 68])
        data[columns_to_fields_y] = data[columns_to_fields_y].apply(
            lambda col: rescale_series(col, field_dim=68, input_min=-0.42, input_max=0.42)
        )

        
        # Feature engineering
        # Team centroids
        data['centroid_x_left_team'] = data[[f"left_{player}_x" for player in range(11)]].mean(axis=1)
        data['centroid_y_left_team'] = data[[f"left_{player}_y" for player in range(11)]].mean(axis=1)
        data['centroid_x_right_team'] = data[[f"right_{player}_x" for player in range(11)]].mean(axis=1)
        data['centroid_y_right_team'] = data[[f"right_{player}_y" for player in range(11)]].mean(axis=1)
        
        # Relative distance to ball
        for player in range(11):
            data[f"distance_ball_to_player_{player}_left_team"] = calculate_distance_player_to_ball(data, player, "left")
            data[f"distance_ball_to_player_{player}_right_team"] = calculate_distance_player_to_ball(data, player, "right")
        
        # Velocity
        seconds_between_frame = 0.1 # 5mins = 3000 frames -> 1/10s between frames or 10fps
        number_of_frames = len(data)
        ## Players
        data['Time'] = data['frame'] * seconds_between_frame
        velocity_columns = {}
        for player in range(11):
            velocity_columns[f"left_{player}_velocity"] = np.sqrt(data[f"left_{player}_x"].diff()**2 + data[f"left_{player}_y"].diff()**2)/seconds_between_frame
            velocity_columns[f"right_{player}_velocity"] = np.sqrt(data[f"right_{player}_x"].diff()**2 + data[f"right_{player}_y"].diff()**2)/seconds_between_frame
        velocity_df = pd.DataFrame(velocity_columns).fillna(0)
        ## Ball
        velocity_df['ball_velocity'] = np.sqrt(data['ball_x'].diff()**2 + data['ball_y'].diff()**2 + data['ball_z'].diff()**2)/seconds_between_frame
        velocity_df['ball_velocity'] = velocity_df['ball_velocity'].fillna(0)
        data = pd.concat([data, velocity_df], axis=1)
        
        # Tactical zone
        num_cols = 5  # horizontal zones
        num_rows = 3  # vertical zones
        zone_data = {}
        for player in range(11):
            zone_data[f"left_{player}_zone_id"] = assign_zone(data[f"left_{player}_x"], data[f"left_{player}_y"],
                                                             num_cols=num_cols, num_rows=num_rows)
            zone_data[f"right_{player}_zone_id"] = assign_zone(data[f"right_{player}_x"], data[f"right_{player}_y"],
                                                              num_cols=num_cols, num_rows=num_rows)
        zone_df = pd.DataFrame(zone_data)
        data = pd.concat([data, zone_df], axis=1)
        
        # Player density - compactness - spread
        (global_densities, global_compactnesses, global_spreads,
        left_team_densities, left_team_compactnesses, left_team_spreads,
        right_team_densities, right_team_compactnesses, right_team_spreads) = compute_metrics_per_frame(data)
        ## Density
        data['global_density'] = global_densities
        data['right_team_density'] = right_team_densities
        data['left_team_density'] = left_team_densities
        ## Compactness
        data['global_compactness'] = global_compactnesses
        data['right_team_compactness'] = right_team_compactnesses
        data['left_team_compactness'] = left_team_compactnesses
        ## Spread
        data['global_spread'] = global_spreads
        data['right_team_spread'] = right_team_spreads
        data['left_team_spread'] = left_team_spreads
        
        # Save
        filename = file.split('.')[0]    
        os.makedirs("./data/processed", exist_ok=True)
        data.to_parquet(f"./data/processed/enhanced_{filename}.parquet", index=False)
        #data.to_csv(f"./enhanced_{filename}.csv", index=False)

    
if __name__ == "__main__":
    main()