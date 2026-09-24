"""Scene-only parameters for the ten strawberry-harvesting experiments.

Keep planning and controller settings in harvest_pipeline.py unchanged between
trials.  This file intentionally contains only scene variations so the ten
results remain comparable.
"""

from copy import deepcopy


DEFAULT_SCENE = {
    "fruit_radius": 0.030,
    # Orientations use Isaac Sim quaternion order: [w, x, y, z].
    "fruit_orientation": [1.0, 0.0, 0.0, 0.0],
    "plant_base_position": [0.550, 0.000, 0.060],
    "plant_base_scale": [0.280, 0.280, 0.120],
    "stem_position": [0.550, 0.000, 0.290],
    "stem_scale": [0.025, 0.025, 0.360],
    "leaf_0_scale": [0.170, 0.060, 0.018],
    "leaf_0_orientation": [1.0, 0.0, 0.0, 0.0],
    "leaf_1_scale": [0.160, 0.065, 0.016],
    "leaf_1_orientation": [1.0, 0.0, 0.0, 0.0],
    "leaf_2_enabled": False,
    "leaf_2_position": [0.0, 0.0, 0.0],
    "leaf_2_scale": [0.160, 0.060, 0.016],
    "leaf_2_orientation": [1.0, 0.0, 0.0, 0.0],
    "neighbor_radius": 0.030,
}


TRIAL_CONFIGS = {
    1: {
        "description": "baseline moderate occlusion",
        "difficulty": "easy",
        "purpose": "Reference success case with a centered target.",
        "fruit_position": [0.550, 0.000, 0.500],
        "leaf_0_position": [0.485, 0.025, 0.545],
        "leaf_1_position": [0.595, -0.020, 0.540],
        "neighbor_position": [0.555, -0.105, 0.490],
    },
    2: {
        "description": "target left and slightly lower",
        "difficulty": "moderate",
        "purpose": "Preserve the validated scene used during development.",
        "fruit_position": [0.520, 0.030, 0.480],
        "leaf_0_position": [0.485, 0.025, 0.525],
        "leaf_1_position": [0.595, -0.020, 0.530],
        "neighbor_position": [0.555, -0.105, 0.490],
    },
    3: {
        "description": "target shifted right",
        "difficulty": "moderate",
        "purpose": "Test lateral workspace variation on the stem side.",
        "fruit_position": [0.545, -0.030, 0.490],
        "leaf_0_position": [0.485, 0.025, 0.535],
        "leaf_1_position": [0.610, -0.020, 0.535],
        "neighbor_position": [0.570, -0.110, 0.485],
    },
    4: {
        "description": "target between upper and lower leaves with neighbor on left",
        "difficulty": "moderate",
        "purpose": "Test vertical obstacle arrangement and left-side neighbor clearance.",

    # Target strawberry
        "fruit_position": [0.530, 0.020, 0.500],

    # Leaf directly above the target strawberry
        "leaf_0_position": [0.530, 0.020, 0.555],
        "leaf_0_scale": [0.175, 0.070, 0.014],
        "leaf_0_orientation": [1.0, 0.0, 0.0, 0.0],

    # Leaf below the target, with sufficient vertical clearance
        "leaf_1_position": [0.530, 0.020, 0.415],
        "leaf_1_scale": [0.165, 0.065, 0.014],
        "leaf_1_orientation": [1.0, 0.0, 0.0, 0.0],

    # Neighboring green strawberry placed on the left
        "neighbor_position": [0.445, 0.020, 0.500],
    },
    5: {
        "description": "low target with an angled leaf on its left",
        "difficulty": "moderate-hard",
        "purpose": "Test low-height reachability with side-leaf occlusion.",
        "fruit_position": [0.530, 0.020, 0.430],
        "leaf_0_position": [0.480, 0.030, 0.480],
        "leaf_1_position": [0.600, -0.020, 0.480],
        # Extra leaf on the target's left/front side, tilted +35 deg about X.
        "leaf_2_enabled": True,
        "leaf_2_position": [0.500, 0.075, 0.465],
        "leaf_2_scale": [0.170, 0.070, 0.016],
        "leaf_2_orientation": [0.953717, 0.300706, 0.0, 0.0],
        "neighbor_position": [0.560, -0.105, 0.440],
    },
    6: {
        "description": "target tangent to neighboring fruit",
        "difficulty": "hard",
        "purpose": "Test grasping and extraction with tangent strawberries.",
        "fruit_position": [0.530, -0.045, 0.490],
        "leaf_0_position": [0.480, 0.020, 0.535],
        "leaf_1_position": [0.600, -0.030, 0.535],
        # Both radii are 0.03 m; the 0.06 m center distance is exact tangency.
        # The neighbor is on the +Y side, leaving the +60-degree route open.
        "neighbor_position": [0.530, 0.015, 0.490],
    },
    7: {
        "description": "strawberry tilted around the x axis",
        "difficulty": "hard",
        "purpose": "Test a strawberry hanging diagonally from the plant.",
        "fruit_position": [0.520, 0.030, 0.480],
        # Target strawberry tilted +35 deg about X.
        "fruit_orientation": [0.953717, 0.300706, 0.0, 0.0],
        "leaf_0_position": [0.485, 0.025, 0.522],
        "leaf_0_scale": [0.180, 0.075, 0.014],
        "leaf_1_position": [0.600, -0.020, 0.535],
        "neighbor_position": [0.560, -0.110, 0.490],
    },
    8: {
    "description": "target upper half enclosed by two tilted leaves",
    "difficulty": "very hard",
    "purpose": "Test grasp planning when two strongly tilted leaves enclose the upper half of the target.",

    "fruit_position": [0.535, 0.000, 0.510],

    # Front/left leaf: moved closer to the fruit and tilted inward by +65 deg.
    "leaf_0_position": [0.515, 0.032, 0.530],
    "leaf_0_scale": [0.195, 0.085, 0.016],
    "leaf_0_orientation": [0.843391, -0.537300, 0.0, 0.0],

    # Back/right leaf: moved closer to the fruit and tilted inward by -65 deg.
    "leaf_1_position": [0.555, -0.032, 0.530],
    "leaf_1_scale": [0.195, 0.085, 0.016],
    "leaf_1_orientation": [0.843391, 0.537300, 0.0, 0.0],

    "neighbor_position": [0.570, -0.115, 0.490],
    },
    
    9: {
        "description": "neighbor blocks preferred minus-60-degree approach",
        "difficulty": "very hard",
        "purpose": "Test whether global comparison selects a different angle.",
        "fruit_position": [0.520, 0.030, 0.480],
        "leaf_0_position": [0.485, 0.025, 0.525],
        "leaf_1_position": [0.595, -0.020, 0.530],
        "neighbor_position": [0.480, 0.085, 0.480],
    },
    10: {
        "description": "dense overhead canopy failure challenge",
        "difficulty": "failure challenge",
        "purpose": "Provide a severe occlusion case for failure analysis.",
        "fruit_position": [0.550, 0.000, 0.470],
        "leaf_0_position": [0.550, 0.000, 0.510],
        "leaf_0_scale": [0.280, 0.280, 0.020],
        "leaf_1_position": [0.575, -0.030, 0.515],
        "leaf_1_scale": [0.200, 0.120, 0.016],
        "neighbor_position": [0.505, -0.055, 0.470],
    },
}


def get_trial_config(trial_id):
    """Return one merged, independent configuration dictionary."""
    if trial_id not in TRIAL_CONFIGS:
        valid = ", ".join(str(value) for value in sorted(TRIAL_CONFIGS))
        raise ValueError(f"Unknown trial_id={trial_id}. Valid IDs: {valid}")

    config = deepcopy(DEFAULT_SCENE)
    config.update(deepcopy(TRIAL_CONFIGS[trial_id]))
    return config


def available_trial_ids():
    return sorted(TRIAL_CONFIGS)

