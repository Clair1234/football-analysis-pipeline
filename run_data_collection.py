import gfootball.env as football_env
from gfootball.env import football_action_set
import numpy as np
import pandas as pd
import random
import argparse
import os
from pathlib import Path

# Import policy generation function
from utils.data_collection import create_policy_variations

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description='Run football simulation with specific policy')
parser.add_argument('--policy', type=str, default='random',
                    help='Policy to run (or "all" to run all policies)')
parser.add_argument('--frames', type=int, default=5000,
                    help='Number of frames to simulate')
parser.add_argument('--round', type=int, default=100,
                    help='Number of round to simulate')
parser.add_argument('--render', action='store_true',
                    help='Render the simulation visually')
parser.add_argument('--output_dir', type=str, default='./data/raw',
                    help='Directory to save output files')

args = parser.parse_args()

# Create output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# --- ENVIRONMENT SETUP ---
env = football_env.create_environment(
    env_name='11_vs_11_kaggle',
    representation='raw',
    number_of_left_players_agent_controls=1,
    render=args.render,
    other_config_options={'action_set': 'full'}
)

# Convert enum actions to lowercase string names
action_names = football_action_set.get_action_set({'action_set': 'full'})
action_names_list = [str(a).lower() for a in action_names]
# Map action name to ID (index)
action_name_to_id = {name: idx for idx, name in enumerate(action_names_list)}

for collect_round in range(args.round):
    # --- CREATE POLICY VARIATIONS ---
    # Generate variations of each policy type
    policy_variations = create_policy_variations(num_variations=20)
    #print(policy_variations)

    # --- STRATEGY MAP ---
    strategies = {
        "random": lambda obs, env, action_name_to_id: env.action_space.sample(),  # Random policy
    }

    # Add all policy variations to strategies dictionary
    for name, policy_func in policy_variations:
        strategies[name] = policy_func

    # Print available strategies
    # print("\nAvailable strategies:")
    # for strategy_name in strategies.keys():
    #     print(f"- {strategy_name}")

    # --- DETERMINE POLICIES TO RUN ---
    policies_to_run = []
    if args.policy.lower() == 'all':
        policies_to_run = list(strategies.keys())
        # print(f"\nRunning all {len(policies_to_run)} policies sequentially")
    elif args.policy in strategies:
        policies_to_run = [args.policy]
        # print(f"\nRunning policy: {args.policy}")
    else:
        print(f"Error: Policy '{args.policy}' not found. Using random policy instead.")
        policies_to_run = ['random']

    # --- RUN SIMULATIONS ---
    for strategy_name in policies_to_run:
        print(f"\nSimulating with strategy: {strategy_name}")
        policy_fn = strategies[strategy_name]
        
        # --- INIT ---
        obs = env.reset()[0]
        done = False
        log = []
        count = 0
        max_frames = args.frames
        action_interval = 1  # realism: act every 4 frames
        
        # --- MAIN LOOP ---
        while not done and count < max_frames:
            if count % action_interval == 0:
                action = [policy_fn(obs, env, action_name_to_id)]
            else:
                action = [action_name_to_id['idle']]  # NOOP
        
            obs_list, reward_list, done_list, info_list = env.step(action)
            obs = obs_list[0]
            reward = reward_list
            done = done_list
            info = info_list
        
            action_id = action[0]
            action_name = action_names[action_id]
        
            # Get base strategy type from the strategy name
            base_strategy = strategy_name.split()[-1] if " " in strategy_name else strategy_name
            if base_strategy not in ["Aggressive", "Defensive", "Passer", "Possession", "Counter"]:
                base_strategy = strategy_name  # For "random" or other special cases
        
            frame = {
                'frame': count,
                'ball_x': obs['ball'][0],
                'ball_y': obs['ball'][1],
                'ball_z': obs['ball'][2],
                'action': action_id,
                'action_name': action_name,
                'reward': reward,
                'strategy': strategy_name,
                'base_strategy': base_strategy,
                'score_left': obs['score'][0],
                'score_right': obs['score'][1]
            }
        
            for i, pos in enumerate(obs['left_team']):
                frame[f'left_{i}_x'] = pos[0]
                frame[f'left_{i}_y'] = pos[1]
        
            for i, pos in enumerate(obs['right_team']):
                frame[f'right_{i}_x'] = pos[0]
                frame[f'right_{i}_y'] = pos[1]
        
            log.append(frame)
            count += 1
        
        # --- SAVE DATA ---
        df = pd.DataFrame(log)
        output_file = Path(args.output_dir) / f'grf_tracking_data_{strategy_name.replace(" ", "_")}_{collect_round}.csv'
        df.to_csv(output_file, index=False)
        print(f"Saved tracking data with {len(df)} frames to {output_file}")

print("\nAll simulations complete!")