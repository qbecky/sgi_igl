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

def return_snake_broken_joint_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    settings_dict = {}
    
    settings_dict['maxiter'] = 1000
    settings_dict['n_cp'] = 15
    settings_dict['n_ts'] = 100
    settings_dict['rho'] = 1.0e-2
    settings_dict['eps'] = 1.0e-1
    if trial_number in [7]:
        settings_dict['eps'] = 1.0e-2
    elif trial_number in [8]:
        settings_dict['eps'] = 1.0e-3
    settings_dict['close_gait'] = True
    if trial_number in [0, 1, 2, 3, 4, 5, 7, 8]:
        settings_dict['n_pts'] = 11
    elif trial_number in [6]:
        settings_dict['n_pts'] = 42
    settings_dict['n_angles'] = settings_dict['n_pts'] - 2
    settings_dict['edges'] = [[idv, idv+1] for idv in range(settings_dict['n_pts']-1)]
    settings_dict['snake_length'] = 1.0
    settings_dict['w_fit'] = 1.0e2
    settings_dict['init_perturb_magnitude'] = 1.0e-2
    
    settings_dict['min_turning_angle'], settings_dict['max_turning_angle'] = - 0.35 * np.pi, 0.35 * np.pi
    
    if trial_number in [0, 2, 6, 7, 8]:
        settings_dict['broken_joint_ids'] = []
        settings_dict['broken_joint_angles'] = []
    elif trial_number in [1, 3]:
        settings_dict['broken_joint_ids'] = [int((settings_dict['n_angles']) // 2)]
        settings_dict['broken_joint_angles'] = [np.pi / 6.0]
    elif trial_number in [4, 5]:
        settings_dict['broken_joint_ids'] = [int((settings_dict['n_angles']) // 2)]
        settings_dict['broken_joint_angles'] = [np.pi / 4.0]
    
    if trial_number in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        target_translation = np.array([1.3 * settings_dict['snake_length'], 0.0, 0.0])
        target_rotation_axis = torch.tensor([0.0, 0.0, 0.0])
        
    if trial_number in [0, 1, 4]:
        settings_dict['w_energy'] = 0.0e1
    elif trial_number in [2, 3, 5, 6, 7, 8]:
        settings_dict['w_energy'] = 1.0e1

    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    settings_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    return settings_dict