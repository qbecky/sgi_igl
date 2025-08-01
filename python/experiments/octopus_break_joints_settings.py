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

def return_octopus_break_joints_experiment_settings(trial_number):
    '''
    Args:
        trial_number: int, the experiment number
    '''
    settings_dict = {}
    
    settings_dict['maxiter'] = 1000
    settings_dict['n_ts'] = 80
    settings_dict['rho'] = 1.0e-2
    settings_dict['eps'] = 4.0e-2
    settings_dict['close_gait'] = True
    settings_dict['init_perturb_magnitude'] = 1.0e-2

    # Octopus geometry
    settings_dict['tail_lengths'] = [0.7, 0.7, 0.7]
    settings_dict['tail_angles'] = [2.0 * np.pi / 3.0, -2.0 * np.pi / 3.0]
    settings_dict['n_angles_per_curve'] = [7, 7, 7]

    settings_dict['n_angles'] = sum(settings_dict['n_angles_per_curve'])
    n_tails = len(settings_dict['n_angles_per_curve'])
    settings_dict['n_pts'] = settings_dict['n_angles'] + 2 * n_tails

    n_vertices_per_curve = [n_angle + 2 for n_angle in settings_dict['n_angles_per_curve']]
    cumulative_n_vertices = np.cumsum([0] + n_vertices_per_curve)
    settings_dict['edges'] = []
    for crv_id in range(n_tails):
        settings_dict['edges'] += [
            [int(cumulative_n_vertices[crv_id]+idv), int(cumulative_n_vertices[crv_id]+idv+1)] for idv in range(n_vertices_per_curve[crv_id]-1)
        ]

    settings_dict['min_turning_angle'], settings_dict['max_turning_angle'] = - 0.35 * np.pi, 0.35 * np.pi
    
    settings_dict['broken_joint_ids'] = []  
    settings_dict['broken_joint_angles'] = []
    
    target_translation = np.array([0.5, 0.0, 0.0])
    target_rotation_axis = torch.tensor([0.0, 0.0, np.pi / 2.0])
    target_quaternion = axis_angle_to_quaternion(target_rotation_axis).numpy()
    settings_dict['gt'] = np.concatenate([target_quaternion, target_translation], axis=0).tolist()

    settings_dict['w_fit'] = 1.0e1
    settings_dict['w_energy'] = 1.0e1

    return settings_dict