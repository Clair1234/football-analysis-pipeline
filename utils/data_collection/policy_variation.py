import random
import numpy as np
import functools

from .policies import aggressive_policy, defensive_policy, passer_policy, possession_policy, counter_attack_policy

def create_policy_variations(num_variations=3):
    """
    Creates multiple variations of each policy by randomly sampling parameters.
    
    Args:
        num_variations: Number of variations to create for each policy type
        
    Returns:
        List of (name, policy_function) tuples
    """
    policy_variations = []
    
    # Create aggressive policy variations
    for i in range(num_variations):
        # Sample parameters with descriptive names
        if i == 0:
            # Create a balanced aggressive policy
            name = "Balanced Aggressive"
            shooting_distance = 0.7
            tackle_threshold = 0.03
            pass_probability = 0.2
            dribble_preference = 0.7
        else:
            # Random variations
            shooting_style = random.choice(["Long-range", "Close-range"])
            tackle_style = random.choice(["Cautious", "Risky"])
            passing_style = random.choice(["Minimal", "Balanced", "Frequent"])
            
            # Shooting distance (lower = closer shots only)
            if shooting_style == "Long-range":
                shooting_distance = random.uniform(0.5, 0.65)
            else:  # Close-range
                shooting_distance = random.uniform(0.75, 0.9)
            
            # Tackle threshold (higher = more tackles)
            if tackle_style == "Cautious":
                tackle_threshold = random.uniform(0.01, 0.025)
            else:  # Risky
                tackle_threshold = random.uniform(0.035, 0.05)
            
            # Pass probability
            if passing_style == "Minimal":
                pass_probability = random.uniform(0.1, 0.15)
            elif passing_style == "Balanced":
                pass_probability = random.uniform(0.2, 0.3)
            else:  # Frequent
                pass_probability = random.uniform(0.3, 0.4)
            
            # Dribble preference (higher = more dribbling)
            dribble_preference = random.uniform(0.6, 0.9)
            
            # Create descriptive name
            name = f"{shooting_style} {tackle_style} Aggressive"
        
        # Create the policy function with these parameters
        policy_func = functools.partial(
            aggressive_policy,
            shooting_distance=shooting_distance,
            tackle_threshold=tackle_threshold,
            pass_probability=pass_probability,
            dribble_preference=dribble_preference
        )
        
        policy_variations.append((name, policy_func))
    
    # Create defensive policy variations
    for i in range(num_variations):
        if i == 0:
            # Create a balanced defensive policy
            name = "Balanced Defensive"
            clearance_threshold = -0.6
            pressure_distance = 0.1
            defensive_depth = 0.3
            press_intensity = 0.05
        else:
            # Random variations
            clearance_style = random.choice(["Deep Clearance", "Conservative"])
            pressure_style = random.choice(["Tight", "Loose"])
            depth_style = random.choice(["Deep", "High Line"])
            
            # Clearance threshold (lower = deeper clearances)
            if clearance_style == "Deep Clearance":
                clearance_threshold = random.uniform(-0.8, -0.7)
            else:  # Conservative
                clearance_threshold = random.uniform(-0.5, -0.3)
            
            # Pressure distance (higher = pressure from further away)
            if pressure_style == "Tight":
                pressure_distance = random.uniform(0.05, 0.08)
            else:  # Loose
                pressure_distance = random.uniform(0.12, 0.15)
            
            # Defensive depth (higher = deeper)
            if depth_style == "Deep":
                defensive_depth = random.uniform(0.35, 0.5)
            else:  # High Line
                defensive_depth = random.uniform(0.15, 0.25)
            
            # Press intensity (higher = more pressing)
            press_intensity = random.uniform(0.03, 0.08)
            
            # Create descriptive name
            name = f"{depth_style} {pressure_style} Defensive"
        
        # Create the policy function with these parameters
        policy_func = functools.partial(
            defensive_policy,
            clearance_threshold=clearance_threshold,
            pressure_distance=pressure_distance,
            defensive_depth=defensive_depth,
            press_intensity=press_intensity
        )
        
        policy_variations.append((name, policy_func))
    
    # Create passer policy variations
    for i in range(num_variations):
        if i == 0:
            # Create a balanced passer policy
            name = "Balanced Passer"
            base_pass_probability = 0.7
            pressure_pass_modifier = 0.2
            short_pass_threshold = 0.2
            movement_randomness = 0.2
        else:
            # Random variations
            pass_freq = random.choice(["Measured", "Excessive"])
            pass_type = random.choice(["Short", "Mixed", "Long"])
            movement_style = random.choice(["Disciplined", "Creative"])
            
            # Base pass probability (higher = more passes)
            if pass_freq == "Measured":
                base_pass_probability = random.uniform(0.5, 0.65)
            else:  # Excessive
                base_pass_probability = random.uniform(0.75, 0.9)
            
            # Pressure pass modifier (higher = more likely to pass under pressure)
            pressure_pass_modifier = random.uniform(0.15, 0.3)
            
            # Short pass threshold (higher = more short passes)
            if pass_type == "Short":
                short_pass_threshold = random.uniform(0.25, 0.35)
            elif pass_type == "Mixed":
                short_pass_threshold = random.uniform(0.15, 0.25)
            else:  # Long
                short_pass_threshold = random.uniform(0.05, 0.15)
            
            # Movement randomness (higher = more random movement)
            if movement_style == "Disciplined":
                movement_randomness = random.uniform(0.1, 0.2)
            else:  # Creative
                movement_randomness = random.uniform(0.3, 0.5)
            
            # Create descriptive name
            name = f"{pass_type} {pass_freq} Passer"
        
        # Create the policy function with these parameters
        policy_func = functools.partial(
            passer_policy,
            base_pass_probability=base_pass_probability,
            pressure_pass_modifier=pressure_pass_modifier,
            short_pass_threshold=short_pass_threshold,
            movement_randomness=movement_randomness
        )
        
        policy_variations.append((name, policy_func))
    
    # Create possession policy variations
    for i in range(num_variations):
        if i == 0:
            # Create a balanced possession policy
            name = "Balanced Possession"
            pressure_distance = 0.15
            high_pressure_threshold = 2
            pass_under_pressure_prob = 0.5
            support_distance = 0.15
        else:
            # Random variations
            pressure_sensitivity = random.choice(["Calm", "Anxious"])
            support_style = random.choice(["Compact", "Wide"])
            
            # Pressure distance (higher = senses pressure from further away)
            pressure_distance = random.uniform(0.1, 0.2)
            
            # High pressure threshold (lower = more sensitive to pressure)
            if pressure_sensitivity == "Calm":
                high_pressure_threshold = random.choice([2, 3])
                pass_under_pressure_prob = random.uniform(0.4, 0.6)
            else:  # Anxious
                high_pressure_threshold = 1
                pass_under_pressure_prob = random.uniform(0.6, 0.8)
            
            # Support distance (higher = wider spacing)
            if support_style == "Compact":
                support_distance = random.uniform(0.1, 0.15)
            else:  # Wide
                support_distance = random.uniform(0.2, 0.3)
            
            # Create descriptive name
            name = f"{pressure_sensitivity} {support_style} Possession"
        
        # Create the policy function with these parameters
        policy_func = functools.partial(
            possession_policy,
            pressure_distance=pressure_distance,
            high_pressure_threshold=high_pressure_threshold,
            pass_under_pressure_prob=pass_under_pressure_prob,
            support_distance=support_distance
        )
        
        policy_variations.append((name, policy_func))
    
    # Create counter-attack policy variations
    for i in range(num_variations):
        if i == 0:
            # Create a balanced counter-attack policy
            name = "Balanced Counter"
            counter_opportunity_threshold = 6
            forward_pass_threshold = 0.3
            interception_prediction = 0.2
            press_intensity = 0.03
        else:
            # Random variations
            counter_trigger = random.choice(["Patient", "Eager"])
            pass_style = random.choice(["Direct", "Buildup"])
            press_style = random.choice(["Aggressive", "Conservative"])
            
            # Counter opportunity threshold (lower = more counter attacks)
            if counter_trigger == "Patient":
                counter_opportunity_threshold = random.randint(6, 8)
            else:  # Eager
                counter_opportunity_threshold = random.randint(3, 5)
            
            # Forward pass threshold (lower = more forward passes)
            if pass_style == "Direct":
                forward_pass_threshold = random.uniform(0.15, 0.25)
            else:  # Buildup
                forward_pass_threshold = random.uniform(0.35, 0.45)
            
            # Interception prediction (higher = more anticipation)
            interception_prediction = random.uniform(0.15, 0.3)
            
            # Press intensity (higher = more pressing)
            if press_style == "Aggressive":
                press_intensity = random.uniform(0.04, 0.08)
            else:  # Conservative
                press_intensity = random.uniform(0.01, 0.03)
            
            # Create descriptive name
            name = f"{counter_trigger} {pass_style} Counter"
        
        # Create the policy function with these parameters
        policy_func = functools.partial(
            counter_attack_policy,
            counter_opportunity_threshold=counter_opportunity_threshold,
            forward_pass_threshold=forward_pass_threshold,
            interception_prediction=interception_prediction,
            press_intensity=press_intensity
        )
        
        policy_variations.append((name, policy_func))
    
    return policy_variations