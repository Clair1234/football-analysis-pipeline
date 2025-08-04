import numpy as np
import random
import functools

# --- ENHANCED UTILITY FUNCTIONS ---
def get_controlled_player_pos(obs):
    """Get the position of the player being controlled."""
    player_id = obs['active']
    return obs['left_team'][player_id]

def get_controlled_player_direction(obs):
    """Get the direction of the player being controlled."""
    player_id = obs['active']
    return obs['left_team_direction'][player_id]

def get_distance(pos1, pos2):
    """Calculate Euclidean distance between two positions."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def is_ball_owned_by_team(obs, team='left'):
    """Check if the ball is owned by the specified team."""
    if team == 'left':
        return obs['ball_owned_team'] == 0
    else:
        return obs['ball_owned_team'] == 1

def is_controlled_player_with_ball(obs):
    """Check if the controlled player has the ball."""
    return obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']

def is_player_in_shooting_range(obs, shooting_distance=0.3):
    """
    ENHANCED: More aggressive shooting range check
    """
    player_pos = get_controlled_player_pos(obs)
    
    # In football coordinates, x goes from -1 (our goal) to 1 (opponent goal)
    
    # Close to goal, high probability shot regardless of angle
    if player_pos[0] > 0.7:
        return True
    
    # Good central position, increased range
    if player_pos[0] > shooting_distance and abs(player_pos[1]) < 0.45:
        return True
    
    # Wider angle but still close
    if player_pos[0] > 0.55 and abs(player_pos[1]) < 0.65:
        # Higher probability to shoot from wider angles
        return random.random() < 0.85
    
    # More aggressive long shots if in opponent half with space
    if player_pos[0] > 0.2 and abs(player_pos[1]) < 0.35:
        return random.random() < 0.35
    
    return False

def should_shoot(obs, aggression=0.7):
    """
    ENHANCED: More aggressive determination if player should shoot
    """
    player_pos = get_controlled_player_pos(obs)
    
    # Prime position - always shoot
    if player_pos[0] > 0.7 and abs(player_pos[1]) < 0.35:
        return True
    
    # Check for open shooting lane
    goal_pos = (1, 0)
    defenders_in_path = 0
    shooting_lane_width = 0.25  # Wider lane for improved shot detection
    
    # Vector from player to goal
    to_goal_vector = np.array(goal_pos) - np.array(player_pos)
    to_goal_distance = np.linalg.norm(to_goal_vector)
    to_goal_unit = to_goal_vector / to_goal_distance if to_goal_distance > 0 else np.array([1, 0])
    
    # Check opponents in shooting lane
    for opp_pos in obs['right_team']:
        # Get vector from player to opponent
        to_opp_vector = np.array(opp_pos) - np.array(player_pos)
        to_opp_distance = np.linalg.norm(to_opp_vector)
        
        # Only check opponents between player and goal
        if to_opp_distance < to_goal_distance:
            # Project opponent position onto shooting lane
            projection = np.dot(to_opp_vector, to_goal_unit)
            
            # Calculate perpendicular distance to shooting lane
            perp_vector = to_opp_vector - projection * to_goal_unit
            perp_distance = np.linalg.norm(perp_vector)
            
            # If opponent is in shooting lane
            if perp_distance < shooting_lane_width and projection > 0:
                defenders_in_path += 1
    
    # Different shooting probabilities based on position and defenders
    base_probability = 0
    
    # Excellent position - more likely to shoot even with defenders
    if player_pos[0] > 0.65 and abs(player_pos[1]) < 0.35:
        base_probability = 0.95 - (defenders_in_path * 0.1)
    # Good position
    elif player_pos[0] > 0.45 and abs(player_pos[1]) < 0.45:
        base_probability = 0.8 - (defenders_in_path * 0.15)
    # Moderate position
    elif player_pos[0] > 0.25 and abs(player_pos[1]) < 0.55:
        base_probability = 0.6 - (defenders_in_path * 0.2)
    # Long shot position
    elif player_pos[0] > 0.0:
        base_probability = 0.3 - (defenders_in_path * 0.05)
    
    # Apply aggression modifier
    shooting_probability = base_probability * (1 + aggression)
    
    # Cap probability at 1.0
    shooting_probability = min(shooting_probability, 1.0)
    
    return random.random() < shooting_probability

def count_defenders_between(obs, target_pos, width_factor=0.25):
    """Count how many defenders are between the player and the target position."""
    player_pos = get_controlled_player_pos(obs)
    
    # Vector from player to target
    to_target_vector = np.array(target_pos) - np.array(player_pos)
    to_target_distance = np.linalg.norm(to_target_vector)
    to_target_unit = to_target_vector / to_target_distance if to_target_distance > 0 else np.array([1, 0])
    
    # Width of the corridor to check
    corridor_width = width_factor
    
    # Count defenders in corridor
    defenders_in_corridor = 0
    for opp_pos in obs['right_team']:
        # Vector from player to opponent
        to_opp_vector = np.array(opp_pos) - np.array(player_pos)
        to_opp_distance = np.linalg.norm(to_opp_vector)
        
        # Only check opponents between player and target
        if to_opp_distance < to_target_distance:
            # Project opponent position onto corridor
            projection = np.dot(to_opp_vector, to_target_unit)
            
            # Calculate perpendicular distance to corridor
            perp_vector = to_opp_vector - projection * to_target_unit
            perp_distance = np.linalg.norm(perp_vector)
            
            # If opponent is in corridor
            if perp_distance < corridor_width and projection > 0:
                defenders_in_corridor += 1
    
    return defenders_in_corridor

def is_offside(obs):
    """Improved offside detection with lookahead."""
    player_pos = get_controlled_player_pos(obs)
    
    # Find the second-last defender position (x-coordinate)
    right_team_x = [player[0] for player in obs['right_team']]
    right_team_x.sort()
    
    if len(right_team_x) >= 2:
        second_last_defender_x = right_team_x[-2]  # Second-last defender
        
        # Ball position
        ball_x = obs['ball'][0]
        
        # If player is ahead of the second-last defender and the ball
        # Add a small buffer to avoid being too close to offside
        return player_pos[0] > second_last_defender_x + 0.02 and player_pos[0] > ball_x
    
    return False

def find_closest_teammate(obs, exclude_active=True):
    """Find the closest teammate to pass to."""
    player_pos = get_controlled_player_pos(obs)
    player_id = obs['active']
    
    closest_dist = float('inf')
    closest_id = -1
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if exclude_active and i == player_id:
            continue
            
        dist = get_distance(player_pos, teammate_pos)
        
        # Prefer teammates that are ahead (closer to opponent goal)
        if teammate_pos[0] > player_pos[0]:
            dist *= 0.75  # Even stronger preference for forward passes
        
        if dist < closest_dist:
            closest_dist = dist
            closest_id = i
            
    return closest_id

def find_most_advanced_teammate(obs):
    """Find the teammate in the most advanced position."""
    player_id = obs['active']
    
    most_advanced_id = -1
    most_advanced_x = -2.0  # Start with a value outside the field
    
    for i, teammate_pos in enumerate(obs['left_team']):
        if i == player_id:
            continue
            
        if teammate_pos[0] > most_advanced_x:
            most_advanced_x = teammate_pos[0]
            most_advanced_id = i
            
    return most_advanced_id

def find_open_space(obs):
    """Find open space to move into with improved forward bias."""
    player_pos = get_controlled_player_pos(obs)
    
    # Define potential movement directions with stronger forward bias
    directions = [
        (0.15, 0),     # Forward (prioritized)
        (0.15, 0.1),   # Forward right
        (0.15, -0.1),  # Forward left
        (0.05, 0.15),  # Right with small forward component
        (0.05, -0.15), # Left with small forward component
    ]
    
    best_dir = directions[0]
    best_score = -float('inf')
    
    for direction in directions:
        target_pos = (player_pos[0] + direction[0], player_pos[1] + direction[1])
        
        # Don't go out of bounds
        if abs(target_pos[1]) > 0.95:
            continue
            
        # Calculate how open this space is (distance to closest opponent)
        min_dist_to_opponent = float('inf')
        for opp_pos in obs['right_team']:
            dist = get_distance(target_pos, opp_pos)
            min_dist_to_opponent = min(min_dist_to_opponent, dist)
        
        # Prefer moving forward (toward opponent goal) - increased forward bonus
        forward_bonus = direction[0] * 0.2
        
        # Calculate score for this direction
        score = min_dist_to_opponent + forward_bonus
        
        if score > best_score:
            best_score = score
            best_dir = direction
    
    return best_dir

def should_make_run(obs):
    """Determine if player should make a forward run."""
    player_pos = get_controlled_player_pos(obs)
    
    # Don't make runs from deep positions
    if player_pos[0] < -0.5:
        return False
        
    # Check if we're in a good position to make a run
    if player_pos[0] < 0.7 and abs(player_pos[1]) < 0.6:
        # Check if there's space ahead
        space_ahead = True
        for opp_pos in obs['right_team']:
            if opp_pos[0] > player_pos[0] and opp_pos[0] < player_pos[0] + 0.3:
                if abs(opp_pos[1] - player_pos[1]) < 0.2:
                    space_ahead = False
                    break
        return space_ahead
    
    return False

def find_best_scoring_teammate(obs, player_pos):
    best_id = -1
    best_score = -1
    for i, mate_pos in enumerate(obs['left_team']):
        if mate_pos[0] > player_pos[0]:  # Forward only
            dist_to_goal = get_distance(mate_pos, (1, 0))
            angle_score = 1 - abs(mate_pos[1])  # Closer to center = better
            score = (1.0 - dist_to_goal) + angle_score
            if score > best_score:
                best_score = score
                best_id = i
    return best_id



def aggressive_policy(obs, env, action_name_to_id, 
                     shooting_distance=0.35,
                     tackle_threshold=0.05,
                     pass_probability=0.25,
                     dribble_preference=0.6,
                     shooting_aggression=0.95):
    player_pos = get_controlled_player_pos(obs)
    ball_pos = obs['ball'][:2]
    goal_pos = (1, 0)

    if is_controlled_player_with_ball(obs):
        dist_to_goal = get_distance(player_pos, goal_pos)

        # High xG zone = SHOOT!
        if player_pos[0] > 0.35 and abs(player_pos[1]) < 0.25 and dist_to_goal < shooting_distance:
            return action_name_to_id['shot']

        # Smart shooting with fewer defenders nearby
        close_defenders = sum(1 for opp in obs['right_team']
                              if get_distance(player_pos, opp) < 0.12)
        if close_defenders <= 1 and random.random() < shooting_aggression:
            return action_name_to_id['shot']

        # Smart vertical pass
        if random.random() < pass_probability:
            best_teammate = find_best_scoring_teammate(obs, player_pos)
            if best_teammate != -1:
                mate_pos = obs['left_team'][best_teammate]
                if mate_pos[0] > player_pos[0] + 0.15:
                    return action_name_to_id['long_pass']
                return action_name_to_id['short_pass']

        # Sprint in counter-attack space
        if dist_to_goal > 0.4 and random.random() < dribble_preference:
            return action_name_to_id['sprint']
        
        return action_name_to_id['dribble']

    else:
        # DEFENSIVE LOGIC
        dist_to_ball = get_distance(player_pos, ball_pos)

        # Quick ball recovery
        if dist_to_ball < tackle_threshold:
            return action_name_to_id['sliding']
        elif dist_to_ball < 0.18:
            return action_name_to_id['sprint']

        # Compact defense - move toward goal-to-ball line
        our_goal = (-1, 0)
        ball_vector = np.array(ball_pos) - np.array(our_goal)
        target_def_pos = np.array(our_goal) + ball_vector * 0.2
        move_vector = target_def_pos - np.array(player_pos)

        # Stay compact behind the ball
        if get_distance(player_pos, target_def_pos) > 0.05:
            if abs(move_vector[1]) > abs(move_vector[0]):
                return action_name_to_id['right'] if move_vector[1] > 0 else action_name_to_id['left']
            return action_name_to_id['top'] if move_vector[0] > 0 else action_name_to_id['bottom']
        
        return action_name_to_id['idle']

def defensive_policy(obs, env, action_name_to_id,
                     clearance_threshold=-0.4,
                     pressure_distance=0.15,
                     defensive_depth=0.25,
                     press_intensity=0.1,
                     shooting_aggression=0.9):

    player_pos = get_controlled_player_pos(obs)
    ball_pos = obs['ball'][:2]
    our_goal = (-1, 0)
    dist_to_ball = get_distance(player_pos, ball_pos)

    # ===== OFFENSIVE PHASE =====
    if is_controlled_player_with_ball(obs):
        dist_to_goal = get_distance(player_pos, (1, 0))

        # More aggressive shooting logic
        if player_pos[0] > 0.35 and abs(player_pos[1]) < 0.25:
            return action_name_to_id['shot']
        if dist_to_goal < 0.4 and random.random() < shooting_aggression:
            return action_name_to_id['shot']

        # Under pressure deep — clear or launch attack
        if player_pos[0] < clearance_threshold:
            if any(get_distance(player_pos, opp) < pressure_distance for opp in obs['right_team']):
                return action_name_to_id['long_pass']

            most_advanced = find_most_advanced_teammate(obs)
            if most_advanced != -1:
                teammate_pos = obs['left_team'][most_advanced]
                if teammate_pos[0] > player_pos[0] + 0.2:
                    return action_name_to_id['long_pass']
            return action_name_to_id['long_pass']  # Default: safety-first

        # Find teammate in best scoring position
        best_teammate = find_best_scoring_teammate(obs, player_pos)
        if best_teammate != -1:
            mate_pos = obs['left_team'][best_teammate]
            if mate_pos[0] > player_pos[0] + 0.1:
                return action_name_to_id['short_pass']
        
        # Dribble into space or toward goal
        if dist_to_goal > 0.3:
            return action_name_to_id['sprint']
        return action_name_to_id['dribble']

    # ===== DEFENSIVE PHASE =====
    else:
        # Immediate ball pressure
        if dist_to_ball < press_intensity:
            return action_name_to_id['sliding']
        if dist_to_ball < 0.2:
            return action_name_to_id['sprint']

        # Defend compact zone if ball is in own half
        if ball_pos[0] < 0:
            ball_to_goal = np.array(our_goal) - np.array(ball_pos)
            defensive_pos = np.array(ball_pos) + ball_to_goal * defensive_depth
            move_vector = defensive_pos - np.array(player_pos)

            if np.linalg.norm(move_vector) > 0.05:
                if abs(move_vector[1]) > abs(move_vector[0]):
                    return action_name_to_id['right'] if move_vector[1] > 0 else action_name_to_id['left']
                else:
                    return action_name_to_id['top'] if move_vector[0] > 0 else action_name_to_id['bottom']
            else:
                if dist_to_ball < 0.25:
                    return action_name_to_id['sprint']
                return action_name_to_id['idle']

        # Press in opponent half to win high
        if dist_to_ball < 0.25:
            return action_name_to_id['sprint']

        # Push up to support pressing block
        if player_pos[0] < 0.4:
            return action_name_to_id['top']
        return action_name_to_id['idle']


def passer_policy(obs, env, action_name_to_id,
                  base_pass_probability=0.7,
                  pressure_pass_modifier=0.3,
                  short_pass_threshold=0.22,
                  movement_randomness=0.1,
                  shooting_aggression=0.95):
    player_pos = get_controlled_player_pos(obs)
    ball_pos = obs['ball'][:2]

    # === OFFENSE ===
    if is_controlled_player_with_ball(obs):
        dist_to_goal = get_distance(player_pos, (1, 0))

        # 1. Opportunistic shooting — more frequent & flexible
        if player_pos[0] > 0.35 and abs(player_pos[1]) < 0.3:
            if random.random() < shooting_aggression:
                return action_name_to_id['shot']
        if dist_to_goal < 0.35 and random.random() < 0.6 * shooting_aggression:
            return action_name_to_id['shot']

        # 2. Handle pressure: increase pass urgency
        nearby_opponents = sum(1 for opp in obs['right_team']
                               if get_distance(player_pos, opp) < 0.12)
        pass_probability = base_pass_probability + nearby_opponents * pressure_pass_modifier

        # 3. Avoid risky pass if offside
        if is_offside(obs):
            return action_name_to_id['dribble']

        # 4. Pass to most advanced OR closest with forward movement
        most_advanced = find_most_advanced_teammate(obs)
        closest_mate = find_closest_teammate(obs)

        if random.random() < min(pass_probability, 1.0):
            if most_advanced != -1:
                mate_pos = obs['left_team'][most_advanced]
                if mate_pos[0] > player_pos[0] + 0.2:
                    return action_name_to_id['long_pass']

            if closest_mate != -1:
                mate_pos = obs['left_team'][closest_mate]
                if get_distance(player_pos, mate_pos) < short_pass_threshold:
                    return action_name_to_id['short_pass']
                elif mate_pos[0] > player_pos[0]:
                    return action_name_to_id['long_pass']
                else:
                    return action_name_to_id['short_pass']

        # 5. Final third fallback shooting
        if player_pos[0] > 0.4:
            return action_name_to_id['shot']

        # 6. Movement: attack space
        open_space = find_open_space(obs)
        if random.random() < movement_randomness:
            return action_name_to_id[random.choice(['dribble', 'top'])]
        else:
            if abs(open_space[1]) > abs(open_space[0]):
                return action_name_to_id['right'] if open_space[1] > 0 else action_name_to_id['left']
            else:
                return action_name_to_id['sprint'] if open_space[0] > 0 else action_name_to_id['bottom']

    # === DEFENSE ===
    else:
        dist_to_ball = get_distance(player_pos, ball_pos)

        # 1. Aggressively press when near
        if dist_to_ball < 0.06:
            return action_name_to_id['sliding']
        elif dist_to_ball < 0.15:
            return action_name_to_id['sprint']

        # 2. Teammate has ball — support position
        if is_ball_owned_by_team(obs, 'left'):
            ball_carrier_id = obs['ball_owned_player']
            carrier_pos = obs['left_team'][ball_carrier_id]
            offset_y = 0.12 * (1 if random.random() > 0.5 else -1)

            target_pos = np.array([carrier_pos[0] + 0.22, carrier_pos[1] + offset_y])
            move_vector = target_pos - np.array(player_pos)

            if abs(move_vector[1]) > abs(move_vector[0]):
                return action_name_to_id['right'] if move_vector[1] > 0 else action_name_to_id['left']
            else:
                return action_name_to_id['top'] if move_vector[0] > 0 else action_name_to_id['bottom']

        # 3. Ball lost — immediate counterpress or fallback
        if player_pos[0] > ball_pos[0] + 0.05:
            return action_name_to_id['bottom']
        else:
            move_vector = np.array(ball_pos) - np.array(player_pos)
            if abs(move_vector[1]) > abs(move_vector[0]):
                return action_name_to_id['right'] if move_vector[1] > 0 else action_name_to_id['left']
            else:
                return action_name_to_id['sprint'] if move_vector[0] > 0 else action_name_to_id['bottom']

def possession_policy(obs, env, action_name_to_id,
                     pressure_distance=0.15,        # Distance to count opponents as pressure
                     high_pressure_threshold=2,     # Number of opponents to consider "high pressure"
                     pass_under_pressure_prob=0.5,  # Probability to pass under moderate pressure
                     support_distance=0.15,         # Distance for support positioning
                     shooting_aggression=0.6):      # New parameter for shooting aggression
    """
    Possession-focused strategy that emphasizes:
    - Ball retention
    - Patient build-up play
    - Controlled passing
    - Positional awareness
    - Taking quality shooting opportunities
    
    Parameters:
    - pressure_distance (0-0.2): Distance to count opponents as applying pressure
    - high_pressure_threshold (1-3): Number of nearby opponents to trigger high pressure response
    - pass_under_pressure_prob (0-1): Probability of passing under moderate pressure
    - support_distance (0-0.3): Distance to position for support
    - shooting_aggression (0-1): Higher values mean more likely to shoot
    """
    player_pos = get_controlled_player_pos(obs)
    ball_pos = obs['ball'][:2]
    
    # Check if we have the ball
    if is_controlled_player_with_ball(obs):
        # --- ENHANCED SHOOTING LOGIC ---
        # Much more aggressive shooting decisions compared to previous versions
        
        # Quick shooting check - if in good position, shoot immediately
        if should_shoot(obs, aggression=shooting_aggression):
            return action_name_to_id['shot']
        
        # Additional opportunistic shooting - even when not in ideal position
        # This makes the AI more willing to take shots from various positions
        if player_pos[0] > 0.4:  # In opponent's half
            # Higher chance of shooting when further forward
            shoot_chance = min(0.7, player_pos[0] * shooting_aggression)
            if random.random() < shoot_chance:
                return action_name_to_id['shot']
        
        # Count opponents nearby (pressure analysis)
        opponents_nearby = 0
        for opp_pos in obs['right_team']:
            if get_distance(player_pos, opp_pos) < pressure_distance:
                opponents_nearby += 1
        
        # Handle high pressure situations
        if opponents_nearby >= high_pressure_threshold:
            # Under high pressure, prioritize keeping possession
            
            # Find teammate to pass to
            closest_mate = find_closest_teammate(obs)
            if closest_mate != -1:
                # Almost always pass under high pressure
                if random.random() < 0.9:
                    return action_name_to_id['short_pass']
            
            # No good passing option or small chance - attempt a shot if in attack position
            if player_pos[0] > 0.2:
                return action_name_to_id['shot']
            
            # Otherwise try to dribble away from pressure
            return action_name_to_id['dribble']
        
        # Handle moderate pressure situations
        elif opponents_nearby > 0:
            # Under moderate pressure
            
            # Look for good passing options first
            closest_mate = find_closest_teammate(obs)
            if closest_mate != -1:
                # Pass with probability affected by pressure
                if random.random() < pass_under_pressure_prob:
                    teammate_pos = obs['left_team'][closest_mate]
                    
                    # Long pass if teammate is far ahead
                    if teammate_pos[0] - player_pos[0] > 0.3:
                        return action_name_to_id['long_pass']
                    else:
                        return action_name_to_id['short_pass']
            
            # Secondary shooting check - shoot under pressure
            if player_pos[0] > 0.3 and random.random() < 0.4 * shooting_aggression:
                return action_name_to_id['shot']
            
            # Try to maintain possession by dribbling
            return action_name_to_id['dribble']
        
        else:
            # No pressure - build up play carefully
            
            # First, look for opportunities to advance
            closest_mate = find_closest_teammate(obs)
            if closest_mate != -1:
                teammate_pos = obs['left_team'][closest_mate]
                
                # If teammate is in a much better position, pass
                if teammate_pos[0] > player_pos[0] + 0.25:
                    # Use long pass for significant position advantage
                    return action_name_to_id['long_pass']
                elif abs(teammate_pos[1] - player_pos[1]) > 0.2:
                    # Use short pass for positional switches
                    return action_name_to_id['short_pass']
            
            # If in a good position but no passing option, try a shot
            if player_pos[0] > 0.5 and random.random() < 0.3:
                return action_name_to_id['shot']
            
            # Otherwise, carefully advance with the ball
            # Use dribble for better control vs sprint for speed
            if player_pos[0] < 0.2:  # In our half
                return action_name_to_id['sprint']  # Move forward quickly
            else:
                return action_name_to_id['dribble']  # More control in attack
    
    else:
        # We don't have the ball
        
        # If very close to the ball, try to get it
        dist_to_ball = get_distance(player_pos, ball_pos)
        if dist_to_ball < 0.03:
            return action_name_to_id['sliding']
        elif dist_to_ball < 0.1:
            return action_name_to_id['sprint']
        
        # If our team has the ball, get in support position
        if is_ball_owned_by_team(obs, 'left'):
            ball_carrier_id = obs['ball_owned_player']
            carrier_pos = obs['left_team'][ball_carrier_id]
            
            # Calculate support position - create triangular passing options
            # We want to be ahead of the ball carrier and slightly to the side
            # This creates passing lanes and options
            
            # Calculate position based on carrier's position and movement
            carrier_direction = obs['left_team_direction'][ball_carrier_id]
            
            # Position slightly ahead and to the side of carrier's direction
            target_pos = np.array([
                carrier_pos[0] + carrier_direction[0] * support_distance * 2,
                carrier_pos[1] + carrier_direction[1] * support_distance
            ])
            
            # Make sure we stay in bounds
            target_pos[1] = max(min(target_pos[1], 0.95), -0.95)
            
            # Move towards target position
            vector = target_pos - np.array(player_pos)
            
            # Determine direction
            if abs(vector[1]) > abs(vector[0]):
                if vector[1] > 0:
                    return action_name_to_id['right']
                else:
                    return action_name_to_id['left']
            else:
                if vector[0] > 0:
                    return action_name_to_id['top']
                else:
                    return action_name_to_id['bottom']
        
        else:
            # Opponent has the ball or ball is free
            
            # Calculate position between ball and our goal
            our_goal = (-1, 0)
            ball_to_goal = np.array(our_goal) - np.array(ball_pos)
            ideal_def_pos = np.array(ball_pos) + ball_to_goal * 0.2
            
            # Move towards ideal defensive position
            vector = ideal_def_pos - np.array(player_pos)
            
            # Determine direction
            if abs(vector[1]) > abs(vector[0]):
                if vector[1] > 0:
                    return action_name_to_id['right']
                else:
                    return action_name_to_id['left']
            else:
                if vector[0] > 0:
                    return action_name_to_id['top']
                else:
                    return action_name_to_id['bottom']


def counter_attack_policy(obs, env, action_name_to_id,
                         counter_opportunity_threshold=6,   # Opponents in our half to trigger counter
                         forward_pass_threshold=0.3,        # Position difference to trigger forward pass
                         interception_prediction=0.2,       # Factor for anticipating ball movement
                         press_intensity=0.03,              # Threshold for pressing vs. positioning
                         shooting_aggression=0.8):          # High aggression for counter-attacks
    """
    Counter-attacking strategy that focuses on:
    - Quick transitions from defense to attack
    - Direct forward passes
    - Fast breaks when gaining possession
    - Exploiting space behind opponent defense
    - Taking early shots to catch defense off-guard
    
    Parameters:
    - counter_opportunity_threshold (3-8): Number of opponents in our half to consider counter opportunity
    - forward_pass_threshold (0.1-0.5): Position difference to trigger forward passes
    - interception_prediction (0.1-0.5): How far ahead to anticipate ball movement
    - press_intensity (0.01-0.1): Distance threshold for pressing the ball
    - shooting_aggression (0-1): Higher values mean more likely to shoot
    """
    player_pos = get_controlled_player_pos(obs)
    player_direction = get_controlled_player_direction(obs)
    ball_pos = obs['ball'][:2]
    
    # Check if counter-attack opportunity exists
    # Count opponents in our half
    opponents_in_our_half = sum(1 for opp_pos in obs['right_team'] if opp_pos[0] < 0)
    counter_opportunity = opponents_in_our_half >= counter_opportunity_threshold
    
    # Check if we have the ball
    if is_controlled_player_with_ball(obs):
        # --- ENHANCED COUNTER-ATTACK SHOOTING LOGIC ---
        # Counter-attack policy needs to be extremely aggressive with shooting
        
        # First check - standard shooting assessment
        if should_shoot(obs, aggression=shooting_aggression):
            return action_name_to_id['shot']
        
        # Counter-attack specific shooting logic - shoot earlier and more often
        # This is important because the opponent defense will be out of position
        if counter_opportunity:
            # Very aggressive shooting when on counter-attack
            # Even from distance if in opponent half
            if player_pos[0] > 0.1:  # Just past midfield
                shoot_chance = 0.3 + (player_pos[0] * 0.4)  # Higher chance further forward
                if random.random() < shoot_chance:
                    return action_name_to_id['shot']
        
        # If we're moving at speed and have space ahead, shoot
        player_speed = np.linalg.norm(player_direction)
        if player_speed > 0.01 and player_pos[0] > 0.4:
            defenders_ahead = count_defenders_between(obs, (1, 0))
            if defenders_ahead <= 1 and random.random() < 0.5:
                return action_name_to_id['shot']
        
        # Counter-attack passing logic - look for direct forward passes
        
        # Find most advanced teammate
        most_advanced_mate = -1
        max_x_pos = player_pos[0]
        
        for i, teammate_pos in enumerate(obs['left_team']):
            if i == obs['active']:  # Skip active player
                continue
                
            if teammate_pos[0] > max_x_pos:
                max_x_pos = teammate_pos[0]
                most_advanced_mate = i
        
        # If we have an advanced teammate and significant positional advantage, make a forward pass
        if most_advanced_mate != -1:
            mate_pos = obs['left_team'][most_advanced_mate]
            if mate_pos[0] - player_pos[0] > forward_pass_threshold:
                # Long pass for significant advantage
                return action_name_to_id['long_pass']
        
        # Find any teammate that's significantly ahead
        for i, teammate_pos in enumerate(obs['left_team']):
            if i == obs['active']:  # Skip active player
                continue
                
            if teammate_pos[0] - player_pos[0] > forward_pass_threshold:
                # Direct forward pass
                return action_name_to_id['long_pass']
        
        # If no good passing option and we're in good position, shoot
        if player_pos[0] > 0.3:
            if random.random() < 0.3 * shooting_aggression:
                return action_name_to_id['shot']
        
        # If in counter-attack, sprint forward with the ball
        if counter_opportunity:
            return action_name_to_id['sprint']
        else:
            # Normal situation - dribble forward
            return action_name_to_id['dribble']
    
    else:
        # We don't have the ball - try to win it back quickly
        
        # Calculate distance to ball
        dist_to_ball = get_distance(player_pos, ball_pos)
        
        # Anticipate ball movement for interception
        ball_direction = np.array(obs['ball_direction'][:2])  # x,y components
        anticipated_ball_pos = np.array(ball_pos) + ball_direction * interception_prediction
        dist_to_anticipated = get_distance(player_pos, anticipated_ball_pos)
        
        # If very close to the ball or anticipated position, try to get it
        if dist_to_ball < press_intensity or dist_to_anticipated < press_intensity:
            return action_name_to_id['sliding']
        elif dist_to_ball < 0.1 or dist_to_anticipated < 0.08:
            return action_name_to_id['sprint']
        
        # If we're in opponent's half, try to get back quickly
        if player_pos[0] > 0 and not is_ball_owned_by_team(obs, 'left'):
            # Get back to defend
            return action_name_to_id['sprint']
        
        # If our team has the ball, get in position for counter
        if is_ball_owned_by_team(obs, 'left'):
            # Position for counter-attack - make forward run
            if player_pos[0] < 0.7:  # Not too far forward
                return action_name_to_id['top']  # Run forward
            else:
                # Already forward, adjust position laterally to create space
                if abs(player_pos[1]) > 0.4:
                    # Move toward center if wide
                    if player_pos[1] > 0:
                        return action_name_to_id['left']
                    else:
                        return action_name_to_id['right']
                else:
                    # Already central, wait for ball
                    return action_name_to_id['idle']
        
        # Opponent has the ball - focus on winning it back
        # Calculate opponent with ball position
        if obs['ball_owned_team'] == 1:  # Opponent has ball
            opp_id = obs['ball_owned_player']
            opp_pos = obs['right_team'][opp_id]
            
            # If opponent with ball is close, press aggressively
            if get_distance(player_pos, opp_pos) < 0.1:
                return action_name_to_id['sliding']
            
            # Move towards opponent with ball
            vector = np.array(opp_pos) - np.array(player_pos)
            
            # Determine direction
            if abs(vector[1]) > abs(vector[0]):
                if vector[1] > 0:
                    return action_name_to_id['right']
                else:
                    return action_name_to_id['left']
            else:
                if vector[0] > 0:
                    return action_name_to_id['top']
                else:
                    return action_name_to_id['bottom']
        
        # Ball is free - move towards it
        vector = anticipated_ball_pos - np.array(player_pos)
        
        # Determine direction
        if abs(vector[1]) > abs(vector[0]):
            if vector[1] > 0:
                return action_name_to_id['right']
            else:
                return action_name_to_id['left']
        else:
            if vector[0] > 0:
                return action_name_to_id['top']
            else:
                return action_name_to_id['bottom']