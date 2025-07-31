import sys as _sys
import os

current_path = os.path.abspath(__file__)

split = current_path.split("sgi_igl")
if len(split)<2:
    print("Please rename the repository 'sgi_igl'")
    raise ValueError
path_to_python_scripts = os.path.join(split[0], "sgi_igl/python/")
_sys.path.insert(0, path_to_python_scripts)

import numpy as np
import torch
from utils import axis_angle_to_quaternion

def return_snake_obstacle_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    settings_dict = {}
    
    settings_dict['maxiter'] = 300
    settings_dict['n_ts'] = 80
    settings_dict['rho'] = 1.0e-2
    settings_dict['eps'] = 2.0e-2
    settings_dict['close_gait'] = True
    settings_dict['n_pts'] = 11
    settings_dict['n_angles'] = settings_dict['n_pts'] - 2
    settings_dict['edges'] = [[idv, idv+1] for idv in range(settings_dict['n_pts']-1)]
    settings_dict['snake_length'] = 1.0
    settings_dict['init_perturb_magnitude'] = 1.0e-2
    
    settings_dict['min_turning_angle'], settings_dict['max_turning_angle'] = - 0.35 * np.pi, 0.35 * np.pi
    
    settings_dict['broken_joint_ids'] = []  
    settings_dict['broken_joint_angles'] = []

    # First the center of the obstacle then the radius
    settings_dict['obstacle_params'] = [
        0.75 * settings_dict['snake_length'], 0.0, 0.0, 
        0.25 * settings_dict['snake_length']
    ]

    target_translation = np.array([1.5 * settings_dict['snake_length'], 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    settings_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    settings_dict['w_fit'] = 1.0e1
    settings_dict['w_energy'] = 1.0e1
    settings_dict['w_obstacle'] = 1.0e1

    return settings_dict