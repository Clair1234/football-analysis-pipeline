import os
import glob
import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

FIELD_LENGTH = 105  # meters
FIELD_WIDTH = 68    # meters

def collect_data(folder):
    # Get list of files
    list_files = glob.glob(f"{folder}/*.parquet")
    list_files = random.sample(list_files, 100)

    data = []

    # For each file collect all data
    for i, file in enumerate(list_files):
        if i == 0:
            data = pd.read_parquet(file)
            data['match_id'] = i
        else:
            datatemp = pd.read_parquet(file)
            datatemp['match_id'] = i
            data = pd.concat([data, datatemp], axis=0)
    return data

# Calculate distance
def calculate_distances(df, player_prefix):
    distances = []
    for i in range(len(df) - 1):
        x1, y1 = df[f'{player_prefix}_x'][i], df[f'{player_prefix}_y'][i]
        x2, y2 = df[f'{player_prefix}_x'][i + 1], df[f'{player_prefix}_y'][i + 1]
        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distances.append(distance)
    return np.sum(distances)

# Distribution of distance covered per player for both teams
def compute_distance_distribution(df, plot_title, filename):
    # Total distance
    total_distances_left = []
    total_distances_right = []
    
    # Per match
    for match_id in df['match_id'].unique():
        # Filter a match 
        match_data = df[df['match_id'] == match_id]
        
        # Store data for players
        for player in range(11):  
            # Left team
            player_prefix_left = f'left_{player}'
            total_distances_left.append(calculate_distances(match_data, player_prefix_left))
            
            # Right team
            player_prefix_right = f'right_{player}'
            total_distances_right.append(calculate_distances(match_data, player_prefix_right))
    
    # Determine common bins
    all_distances = total_distances_left + total_distances_right
    min_distance = min(all_distances)
    max_distance = max(all_distances)
    bins = np.linspace(min_distance, max_distance, num=10)
    
    # Calculate counts and convert to percentages
    counts_left, _ = np.histogram(total_distances_left, bins=bins)
    counts_right, _ = np.histogram(total_distances_right, bins=bins)
    
    total_players = len(total_distances_left) + len(total_distances_right)
    percentages_left = (counts_left / total_players) * 100
    percentages_right = (counts_right / total_players) * 100
    
    # Plot overlapping histograms
    plt.figure(figsize=(10, 6))
    
    # Plot as step histograms for better overlapping visualization
    plt.hist(total_distances_left, bins=bins, alpha=0.7, color='blue', 
             edgecolor='black', label='Left Team', 
             weights=np.ones(len(total_distances_left))/total_players*100)
    plt.hist(total_distances_right, bins=bins, alpha=0.7, color='orange', 
             edgecolor='black', label='Right Team',
             weights=np.ones(len(total_distances_right))/total_players*100)
    
    plt.title(plot_title)
    plt.xlabel('Distance covered (meter)')
    plt.ylabel('Percentage of players (%)')
    plt.grid(axis='y', alpha=0.75)
    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.close()

def pitch_control(df):
    # Define pitch dimensions
    pitch_length = 105  # Length of the pitch
    pitch_width = 68    # Width of the pitch

    # Create a figure for the pitch
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw the pitch
    plt.plot([0, 0, pitch_length, pitch_length, 0], [0, pitch_width, pitch_width, 0, 0], color="black")  # Pitch outline
    plt.xlim(-10, pitch_length + 10)
    plt.ylim(-10, pitch_width + 10)

    # Define control radius
    control_radius = 5  # Control radius for each player

    # Plot player control areas
    for player in range(11):  # Left players
        x = df[f'left_{player}_x'][0] + pitch_length / 2  # Adjust x position
        y = df[f'left_{player}_y'][0] + pitch_width / 2  # Adjust y position
        circle = Circle((x, y), control_radius, color='blue', alpha=0.3)
        ax.add_patch(circle)

    for player in range(11):  # Right players
        x = df[f'right_{player}_x'][0] + pitch_length / 2  # Adjust x position
        y = df[f'right_{player}_y'][0] + pitch_width / 2  # Adjust y position
        circle = Circle((x, y), control_radius, color='red', alpha=0.3)
        ax.add_patch(circle)

    # Plot the ball
    ball_x = df['ball_x'][0] + pitch_length / 2
    ball_y = df['ball_y'][0] + pitch_width / 2
    plt.scatter(ball_x, ball_y, color='yellow', s=100, label='Ball')

    # Add labels and title
    plt.title('Pitch Control Visualization')
    plt.xlabel('Pitch Length')
    plt.ylabel('Pitch Width')
    plt.axhline(y=pitch_width / 2, color='gray', linestyle='--')  # Center line
    plt.axvline(x=pitch_length / 2, color='gray', linestyle='--')  # Center line
    plt.legend()
    plt.grid()
    #plt.show()
    
    
def normalize_to_meters(x, y):
    real_x = x * (FIELD_LENGTH)
    real_y = y * (FIELD_WIDTH)
    return real_x, real_y


def plot_player_positions_heatmap_v0(df, strategy=None):
    # Extract player positions for left and right teams
    left_players = df.filter(like='left_')
    right_players = df.filter(like='right_')
    
    # Reshape the data for heatmap
    left_positions = left_players.values.reshape(-1, 2)  # Reshape to (n, 2) for x, y
    right_positions = right_players.values.reshape(-1, 2)  # Reshape to (n, 2) for x, y
    
    # Create a heatmap for left players
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x=left_positions[:, 0], y=left_positions[:, 1], fill=True, thresh=0, levels=100, cmap='Blues', alpha=0.5)
    sns.kdeplot(x=right_positions[:, 0], y=right_positions[:, 1], fill=True, thresh=0, levels=100, cmap='Reds', alpha=0.5)

    if strategy:
        plt.title('Player Positions Heatmap for strategy: {strategy}')
    else:
        plt.title('Player Positions Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(['Left Team', 'Right Team'])
    #plt.show()
    
def plot_player_positions_heatmap(df, strategy=None):
    # Extract player positions for left and right teams
    left_players = df.filter(like='left_')
    right_players = df.filter(like='right_')
    
    # Reshape the data for heatmap
    left_positions = left_players.values.reshape(-1, 2)  # Reshape to (n, 2) for x, y
    right_positions = right_players.values.reshape(-1, 2)  # Reshape to (n, 2) for x, y
    
    # Create a heatmap for left players
    plt.figure(figsize=(12, 6))
    sns.kdeplot(x=left_positions[:, 0], y=left_positions[:, 1], fill=True, thresh=0, cmap='Blues', alpha=0.5)
    sns.kdeplot(x=right_positions[:, 0], y=right_positions[:, 1], fill=True, thresh=0, cmap='Reds', alpha=0.5)

    if strategy:
        plt.title(f'Player Positions Heatmap for strategy: {strategy}')
    else:
        plt.title('Player Positions Heatmap')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(['Left Team', 'Right Team'])
    #plt.show()


def plot_ball_position_heatmap(df, strategy=None):
    # Extract ball positions
    ball_positions = df[['ball_x', 'ball_y']]
    
    # Create a heatmap for ball positions
    # Set up the field
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)

    sns.kdeplot(data=ball_positions, x='ball_x', y='ball_y', fill=True, thresh=0, levels=100, cmap='Greens')
    plt.xlabel('Ball X Position (meters)')
    plt.ylabel('Ball Y Position (meters)')
    
    if strategy:
        plt.title(f'Ball Position Heatmap for strategy {strategy}')
        plt.savefig(f"./EDA_plot/Heatmap_ball_position_{strategy}.png")
    else:
        plt.title('Ball Position Heatmap')
        plt.savefig("./EDA_plot/Heatmap_ball_position.png")
    plt.close()
        
def plot_centroid_time_series(data, strategy=None):
    sns.lineplot(data=data, x="frame", y="centroid_x_right_team", label='X centroid right', color='red')
    sns.lineplot(data=data, x="frame", y="centroid_x_left_team", label='X centroid left', color='blue')
    plt.ylabel('Centroid Position')
    plt.legend()
    if strategy:
        plt.title(f"Centroid X time serie for strategy: {strategy}")
        plt.savefig(f"./EDA_plot/X_centroid_{strategy}.png")
    else:
        plt.title(f"Centroid X time serie")
        plt.savefig("./EDA_plot/X_centroid.png")
    plt.close()
    sns.lineplot(data=data, x="frame", y="centroid_y_right_team", label='Y centroid right', color='red')
    sns.lineplot(data=data, x="frame", y="centroid_y_left_team", label='Y centroid left', color='blue')
    plt.ylabel('Centroid Position')
    plt.legend()
    if strategy:
        plt.title(f"Centroid Y time serie for strategy: {strategy}")
        plt.savefig(f"./EDA_plot/Y_centroid_{strategy}.png")
    else:
        plt.title(f"Centroid X time serie")
        plt.savefig("./EDA_plot/Y_centroid.png")
    plt.close()
        
def compute_velocity_distribution(df, plot_title, filename):
    # Total velocity
    total_velocity_left = []
    total_velocity_right = []
    
    # Per match
    for match_id in df['match_id'].unique():
        # Filter a match 
        match_data = df[df['match_id'] == match_id]
        
        # Store data for players
        for player in range(11): 
            total_velocity_left += match_data[f'left_{player}_velocity'].to_list()
            total_velocity_right += match_data[f'right_{player}_velocity'].to_list()

    # Determine common bins
    all_velocities = total_velocity_left + total_velocity_right
    min_velocity = min(all_velocities)
    max_velocity = max(all_velocities)
    bins = np.linspace(min_velocity, max_velocity, num=100)

    # Plot
    plt.figure(figsize=(10, 6))
    
    # Option 1: Use weights instead of density for percentage display
    plt.hist(total_velocity_left, bins=bins, alpha=0.7, color='blue', edgecolor='black', 
             weights=np.ones(len(total_velocity_left)) / len(total_velocity_left) * 100,
             label='Left Team')
    plt.hist(total_velocity_right, bins=bins, alpha=0.7, color='red', edgecolor='black', 
             weights=np.ones(len(total_velocity_right)) / len(total_velocity_right) * 100,
             label='Right Team')
    
    plt.title(plot_title)
    plt.xlim(min_velocity, max_velocity)
    plt.xlabel('Velocity (meter/second)')
    plt.ylabel('Percentage (%)')
    plt.grid(axis='y', alpha=0.75)
    
    plt.legend()
    plt.savefig(f"{filename}.png")
    plt.close()
        
if __name__ == "__main__":
    folder = "tracking_data"
    folder = "./data/processed"
    data = collect_data(folder)

    # 0. Set folders
    os.makedirs("analysis/EDA", exist_ok=True)
    
    # 1. Global statistics
    print("############################## STATS ################################")
    eda_folder = "analysis/EDA"
    # 1.1 Descriptive
    print("Data description: ")
    print(data.describe())
    data.describe().to_csv(f"./{eda_folder}/data_stats.csv")
    
    # 2. Distribution of distance covered per player
    # print("############################## DISTANCE ################################")
    # 2.1 Global
    plot_title = "Distribution of distance covered per player"
    filename = f"./{eda_folder}/Distribution_of_distance_covered_per_player"
    compute_distance_distribution(data, plot_title, filename)

    # 2.2 Per main strategy
    for strategy in data['base_strategy'].unique():
        strat_data = data[data['base_strategy'] == strategy]
        plot_title_strat = f"Distribution of distance covered per player for strategy: {strategy}"
        filename_strat = f"./{eda_folder}/Distribution_of_distance_covered_per_player_for_strategy_{strategy}"
        compute_distance_distribution(strat_data, plot_title_strat, filename_strat)

    
    # 3. Heatmap of ball positions
    # 3.1 Global
    plot_ball_position_heatmap(data)
    # 3.2 Per main strategy
    for strategy in data['base_strategy'].unique():
        strat_data = data[data['base_strategy'] == strategy]
        plot_ball_position_heatmap(strat_data, strategy)
    
    # 4. Team centroid time serie
    # print("############################## CENTROIDS ################################")
    # 4.1 Global
    plot_centroid_time_series(data)
    for strategy in data['base_strategy'].unique():
        strat_data = data[data['base_strategy'] == strategy]
        plot_centroid_time_series(strat_data, strategy)
        
    # 5. Velocity
    # 5.1 Global
    plot_title = "Distribution of velocity per player"
    filename = f"./{eda_folder}/Distribution_of_velocity_per_player"
    compute_velocity_distribution(data, plot_title, filename)
    # 5.2 Per main strategy
    for strategy in data['base_strategy'].unique():
        strat_data = data[data['base_strategy'] == strategy]
        plot_title_strat = f"Distribution of velocity per player for strategy: {strategy}"
        filename_strat = f"./{eda_folder}/Distribution_of_velocity_per_playerfor_strategy_{strategy}"
        compute_velocity_distribution(strat_data, plot_title_strat, filename_strat)