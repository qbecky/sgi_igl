import os
import sys as _sys

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
PATH_TO_CUBICSPLINES = os.path.join(os.path.dirname(SCRIPT_PATH), 'ext/torchcubicspline')
_sys.path.append(PATH_TO_CUBICSPLINES)

import numpy as np
import torch
from torchcubicspline import (
    natural_cubic_spline_coeffs, NaturalCubicSpline
)
from utils import register_points_torch_2d, register_points_no_translation_torch_2d, vmap_rotate_about_axis

def snake_angles_generation(
    operational_angles, snake_length, broken_joint_ids, broken_joint_angles,
    example_pos_, n_ts, close_gait=False, flip_snake=False, return_discretized=False,
):
    '''Generate a snake parameterized by turning angles evolving in time
    
    Args:
        operational_angles: (n_ts-close_gait, n_operational_angles) tensor containing all the operational angles
        snake_length: float representing the length of the snake
        broken_joint_ids: list of int representing the indices of the broken joints
        broken_joint_angles: (n_ts-close_gait, n_broken_angles) tensor representing the turning angles of the broken joints
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        close_gait: bool telling whether the gait should be periodic
        flip_snake: bool representing whether the snake should be flipped or not
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''
    
    n_s = example_pos_.shape[1]
    n_angles = n_s - 2
    edge_length = snake_length / (n_s - 1)
    
    if close_gait:
        operational_angles = torch.cat([operational_angles, operational_angles[0].reshape(1, -1)])
        broken_joint_angles = torch.cat([broken_joint_angles, broken_joint_angles[0].reshape(1, -1)])

    operational_joint_ids = [i for i in range(n_angles) if i not in broken_joint_ids]
    all_angles = torch.zeros(size=(n_ts, n_angles))
    all_angles[:, operational_joint_ids] = operational_angles
    all_angles[:, broken_joint_ids] = broken_joint_angles
    
    pos_ = integrate_snake_angles_constant_edge_length(all_angles, edge_length, n_ts)
    pos3d_ = torch.zeros_like(example_pos_)
    
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_s, 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_snake) * snake_length * (torch.linspace(0.0, 1.0, n_s) - 0.5)
    registration_points[:, 1] = 0.01 * snake_length * torch.randn(size=(n_s,))
    pos3d_[..., :2] = register_points_torch_2d(pos_, registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False)
    
    if return_discretized:
        return pos3d_, operational_angles
    else:
        return pos3d_
    
def octopus_angles_generation(
    operational_angles, tail_lengths, tail_angles, broken_joint_ids, broken_joint_angles, n_angles_per_curve,
    example_pos_, n_ts, close_gait=False, flip_geometry=False, return_discretized=False,
):
    '''Generate a snake parameterized by turning angles evolving in time
    
    Args:
        operational_angles: (n_ts-close_gait, n_operational_angles) tensor containing all the operational angles
        tail_lengths: (n_tails,) tensor representing the length of each tail
        tail_angles: (n_tails-1,) tensor representing the angle of each tail to the first one
        broken_joint_ids: list of int representing the indices of the broken joints
        broken_joint_angles: (n_ts-close_gait, n_broken_angles) tensor representing the turning angles of the broken joints
        n_angles_per_curve: list of integers representing the number of angles per curve
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        close_gait: bool telling whether the gait should be periodic
        flip_geometry: bool representing whether the geometry should be flipped or not
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the octopus
    '''
    
    if close_gait:
        operational_angles = torch.cat([operational_angles, operational_angles[0].reshape(1, -1)])
        broken_joint_angles = torch.cat([broken_joint_angles, broken_joint_angles[0].reshape(1, -1)])

    n_angles = sum(n_angles_per_curve)
    operational_joint_ids = [i for i in range(n_angles) if i not in broken_joint_ids]
    all_angles = torch.zeros(size=(n_ts, n_angles))
    all_angles[:, operational_joint_ids] = operational_angles
    all_angles[:, broken_joint_ids] = broken_joint_angles

    tail_angles = torch.cat([torch.tensor([0.0]).reshape(1,), tail_angles], dim=0)
    c_tail_angles = torch.cos(tail_angles)
    s_tail_angles = torch.sin(tail_angles)
    rotation_matrices = torch.stack([
        torch.stack([c_tail_angles, -s_tail_angles], dim=-1),
        torch.stack([s_tail_angles, c_tail_angles], dim=-1)
    ], dim=-2)  # shape (n_tails, 2, 2)

    n_vertices_per_curve = [n_angle + 2 for n_angle in n_angles_per_curve]
    slice_angles = np.cumsum([0] + n_angles_per_curve)
    slice_vertices = np.cumsum([0] + n_vertices_per_curve)
    
    pos_ = torch.zeros_like(example_pos_)[..., :2]
    pos3d_ = torch.zeros_like(example_pos_)
    for id_tail, (id_angle_start, id_angle_end, id_pos_start, id_pos_end, tail_length) in enumerate(zip(slice_angles[:-1], slice_angles[1:], slice_vertices[:-1], slice_vertices[1:], tail_lengths)):
        n_vertices = id_pos_end - id_pos_start
        edge_length = tail_length / (n_vertices - 1)
        pos_[:, id_pos_start:id_pos_end] = torch.einsum('jk, tik -> tij', rotation_matrices[id_tail], integrate_snake_angles_constant_edge_length(all_angles[:, id_angle_start:id_angle_end], edge_length, n_ts))

    # aligns the first tail and rotates the rest accordingly around the central vertex
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_vertices_per_curve[0], 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_geometry) * tail_lengths[0] * (torch.linspace(0.0, 1.0, n_vertices_per_curve[0]) - 0.5)
    registration_points[:, 1] = 0.01 * tail_lengths[0] * torch.randn(size=(n_vertices_per_curve[0],))
    _, rot = register_points_no_translation_torch_2d(pos_[:, :n_vertices_per_curve[0]], registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False, return_rotation=True)
    pos3d_[..., :2] = torch.einsum("tij, taj -> tai", rot, pos_)
    
    if return_discretized:
        return pos3d_, operational_angles
    else:
        return pos3d_

def snake_angles_generation_cubic_splines(
    control_points, snake_length, broken_joint_ids, broken_joint_angles,
    example_pos_, n_ts, n_cp, close_gait=False, flip_snake=False, return_discretized=False,
):
    '''Generate a snake parameterized by turning angles evolving in time
    
    Args:
        control_points: (n_cp, n_operational_angles) tensor representing the control points of the cubic spline, it
        snake_length: float representing the length of the snake
        broken_joint_ids: list of int representing the indices of the broken joints
        broken_joint_angles: (n_ts, n_broken_angles) tensor representing the turning angles of the broken joints
        example_pos_: (n_ts, n_s, 3) tensor that is needed when vmapping the function
        n_ts: int representing the number of time steps
        n_cp: int representing the number of control points
        flip_snake: bool representing whether the snake should be flipped or not
        return_discretized: Bool, whether to return the discretized spline or not.
    
    Returns:
        (n_ts, n_s, 3) tensor representing the positions of the snake
    '''
    
    n_s = control_points.shape[1] + len(broken_joint_ids) + 2
    n_angles = n_s - 2
    edge_length = snake_length / (n_s - 1)
    
    t = torch.linspace(0.0, 1.0, n_ts)
    s = torch.linspace(0.0, 1.0, n_s-1)
    ts_cp = torch.linspace(0.0, 1.0, n_cp+close_gait)

    if close_gait:
        control_points = torch.cat([control_points, control_points[0].reshape(1, -1)])
    spline_coeffs = natural_cubic_spline_coeffs(ts_cp, control_points)
    spline = NaturalCubicSpline(spline_coeffs)
    operational_angles = spline.evaluate(t) # shape (n_ts, n_operational_angles)
    
    operational_joint_ids = [i for i in range(n_angles) if i not in broken_joint_ids]
    all_angles = torch.zeros(size=(n_ts, n_angles))
    all_angles[:, operational_joint_ids] = operational_angles
    all_angles[:, broken_joint_ids] = broken_joint_angles
    
    pos_ = integrate_snake_angles_constant_edge_length(all_angles, edge_length, n_ts)
    pos3d_ = torch.zeros_like(example_pos_)
    
    torch.manual_seed(0)
    registration_points = torch.zeros(size=(n_s, 2))
    registration_points[:, 0] = (1.0 - 2.0 * flip_snake) * snake_length * (torch.linspace(0.0, 1.0, n_s) - 0.5)
    registration_points[:, 1] = 0.01 * snake_length * torch.randn(size=(n_s,))
    pos3d_[..., :2] = register_points_torch_2d(pos_, registration_points.unsqueeze(0).repeat(n_ts, 1, 1), allow_flip=False)
    
    if return_discretized:
        return pos3d_, operational_angles
    else:
        return pos3d_

def integrate_snake_angles_constant_edge_length(
    all_angles, edge_length, n_ts,
):
    '''Integrate the turning angles of the snake to get the positions
    
    Args:
        all_angles: (n_ts, n_angles) tensor representing the turning angles of the snake
        edge_length: float representing the length of the edges of the snake
        n_ts: int representing the number of time steps
    
    Returns:
        (n_ts, n_angles+2, 2) tensor representing the positions of the snake
    '''
    
    n_ts, n_angles = all_angles.shape
    all_cumulated_angles = torch.cumsum(torch.cat([torch.zeros(size=(n_ts, 1)), all_angles], dim=-1), dim=-1)
    
    pos_ = torch.zeros(size=(n_ts, n_angles+2, 2))
    pos_[..., 1:, 0] = edge_length * torch.cumsum(torch.cos(all_cumulated_angles), dim=-1)
    pos_[..., 1:, 1] = edge_length * torch.cumsum(torch.sin(all_cumulated_angles), dim=-1)
    
    return pos_
