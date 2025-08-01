#import MeshFEM
#import py_newton_optimizer

import numpy as np
from obstacle_implicits import ImplicitFunction
from physics_quantities_torch_edge import vmap_energy_torch
import torch
from utils import vmap_euc_transform_torch, vmap_euc_transform_T_torch, vmap_qmultiply_torch, vmap_qbar_torch, register_points_torch_2d, qmultiply_torch, qbar_torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

## Can be useful for a generic gradient computation
def generate_grad_fun(fun):
    '''
    Args:
        fun: a function with signature (pos_, g) -> torch.tensor representing a scalar
        
    Returns:
        grad_fun: a function with signature (pos_, g) -> grad_obj_g, grad_obj_pos_ representing the gradients
    '''

    def grad_fun(pos_, g):
        pos_torch_ = torch.tensor(pos_)
        pos_torch_.requires_grad = True
        
        g_torch = torch.tensor(g)
        g_torch.requires_grad = True

        obj = fun(pos_torch_, g_torch)
        obj.backward(torch.ones_like(obj))
        
        if g_torch.grad is None:
            grad_g = torch.zeros_like(g_torch)
        else:
            grad_g = g_torch.grad
        if pos_torch_.grad is None:
            grad_pos_ = torch.zeros_like(pos_torch_)
        else:
            grad_pos_ = pos_torch_.grad
        return grad_g.numpy(), grad_pos_.numpy()
    
    return grad_fun


## Objective functions
def compare_last_translation(g, gt):
    '''Compares the translation to the target translation
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        
    Returns:
        scalar representing the objective
    '''
    return np.sum((g[-1, 4:] - gt[4:]) ** 2) / 2.0

def grad_compare_last_translation(g, gt):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    grad = np.zeros_like(g)
    grad[-1, 4:] = g[-1, 4:] - gt[4:]
    return grad

def compare_last_orientation(g, gt):
    '''Compares the rotation to the target rotation

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    # since the quaternions are normalized, the inverse is the conjugate
    diff_quat = qmultiply_torch(g_torch[-1, :4], qbar_torch(gt_torch[:4]))
    angle = 2.0 * torch.atan2(torch.linalg.norm(diff_quat[:3]), diff_quat[3])

    return 0.5 * torch.minimum(
        angle, torch.tensor(2.0 * np.pi) - angle
    ) ** 2

def grad_compare_last_orientation(g, gt):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    gt_torch = torch.tensor(gt)

    obj = compare_last_orientation(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()

def compare_last_registration(g, gt):
    '''Compares the registration to the target registration
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (7,) array representing the target rigid transformation
        
    Returns:
        scalar representing the objective
    '''
    return compare_last_translation(g, gt) + compare_last_orientation(g, gt)

def grad_compare_last_registration(g, gt):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    return grad_compare_last_translation(g, gt) + grad_compare_last_orientation(g, gt)

def compare_all_translations(g, gt):
    '''Compares the translation to the target translation
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    return np.sum((g[:, 4:] - gt[:, 4:]) ** 2) / 2.0

def grad_compare_all_translations(g, gt):
    '''Gradient of the objective with respect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    grad = np.zeros_like(g)
    grad[:, 4:] = g[:, 4:] - gt[:, 4:]
    return grad

def compare_all_orientations(g, gt):
    '''Compares the rotations to the target rotations

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    # since the quaternions are normalized, the inverse is the conjugate
    diff_quats = vmap_qmultiply_torch(g_torch[:, :4], vmap_qbar_torch(gt_torch[:, :4]))
    angles = 2.0 * torch.atan2(torch.linalg.norm(diff_quats[:, :3], dim=1), diff_quats[:, 3])

    return torch.sum(torch.minimum(
        angles, torch.tensor(2.0 * np.pi) - angles
    ) ** 2) / 2.0

def grad_compare_all_orientations(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_orientations(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_translation_increments(g, gt):
    '''Compares the rotations increments to the target rotations increments

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    trans_increments = g_torch[1:, 4:] - g_torch[:-1, 4:]
    transt_increments = gt_torch[1:, 4:] - gt_torch[:-1, 4:]

    return torch.sum((trans_increments - transt_increments) ** 2) / 2.0

def grad_compare_all_translation_increments(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_translation_increments(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_orientation_increments(g, gt):
    '''Compares the rotations increments to the target rotations increments

    Note: this assumes the quaternions are normalized, i.e., the first four elements of g and gt are quaternions
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gt) is np.ndarray:
        gt_torch = torch.tensor(gt)
    else:
        gt_torch = gt

    quats_increments = vmap_qmultiply_torch(g_torch[1:, :4], vmap_qbar_torch(g_torch[:-1, :4]))
    quatst_increments = vmap_qmultiply_torch(gt_torch[1:, :4], vmap_qbar_torch(gt_torch[:-1, :4]))
    diff_quats = vmap_qmultiply_torch(quats_increments, vmap_qbar_torch(quatst_increments))

    angles = 2.0 * torch.atan2(
        torch.linalg.norm(diff_quats[:, :3], dim=1),
        diff_quats[:, 3]
    )

    return torch.sum(torch.minimum(
        angles, torch.tensor(2.0 * np.pi) - angles
    ) ** 2) / 2.0

def grad_compare_all_orientation_increments(g, gt):
    '''Gradient of the objective with respect to its first input

    Args:
        Same as above

    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    gt_torch = torch.tensor(gt)

    obj = compare_all_orientation_increments(g_torch, gt_torch)
    obj.backward(torch.ones_like(obj))
    
    return g_torch.grad.numpy()

def compare_all_registrations(g, gt):
    '''Compares the registration to the target registration

    Note: use them separately since the two terms would need different normalizations
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gt: (n_steps, 7) array representing the target rigid transformations
        
    Returns:
        scalar representing the objective
    '''
    return compare_all_translations(g, gt) + compare_all_orientations(g, gt).detach().item()

def grad_compare_all_registrations(g, gt):
    '''Gradient of the objective with respect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    return grad_compare_all_translations(g, gt) + grad_compare_all_orientations(g, gt)

def pass_checkpoints(g, gcp):
    '''Makes sure the rigid transformations pass through the checkpoints
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gcp: (n_cp, 7) array representing the checkpoints
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gcp) is np.ndarray:
        gcp_torch = torch.tensor(gcp)
    else:
        gcp_torch = gcp
    
    dists_to_cp = torch.sum((g_torch.unsqueeze(0)[..., 4:] - gcp_torch.unsqueeze(1)[..., 4:]) ** 2, dim=-1) # (n_steps, n_cp)
    min_dists_to_cp = torch.mean(torch.min(dists_to_cp, dim=1)[0])
    return min_dists_to_cp / 2.0

def grad_pass_checkpoints(g, gcp):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    gcp_torch = torch.tensor(gcp)

    obj = pass_checkpoints(g_torch, gcp_torch)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()

def pass_checkpoints_smooth(g, gcp, temperature=1.0):
    '''Makes sure the rigid transformations pass through the checkpoints
    
    Args:
        g: (n_steps, 7) array representing rigid transformations
        gcp: (n_cp, 7) array representing the checkpoints
        temperature: scalar representing the temperature of the smooth maximum
        
    Returns:
        scalar representing the objective
    '''
    if type(g) is np.ndarray:
        g_torch = torch.tensor(g)
    else:
        g_torch = g
    if type(gcp) is np.ndarray:
        gcp_torch = torch.tensor(gcp)
    else:
        gcp_torch = gcp
    
    dists_to_cp = torch.sum((g_torch.unsqueeze(0)[..., 4:] - gcp_torch.unsqueeze(1)[..., 4:]) ** 2, dim=-1) # (n_steps, n_cp)
    smooth_min = - temperature * torch.mean(torch.logsumexp(- dists_to_cp / temperature, dim=0)) - temperature * torch.mean(torch.logsumexp(- dists_to_cp / temperature, dim=1))
    return smooth_min / 2.0

def grad_pass_checkpoints_smooth(g, gcp, temperature=1.0):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
    '''
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True
    gcp_torch = torch.tensor(gcp)

    obj = pass_checkpoints_smooth(g_torch, gcp_torch, temperature=temperature)
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy()

def energy_path(pos_, g, masses, a_weight, b_weight, edges, fun_anisotropy_dir):
    '''Compute the energy of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) array representing the rigid transformation
        masses: (n_steps, n_points, 1) array of the masses of the system
        a_weight: scalar or (n_points, 1) array representing the a parameters for anisotropy of local dissipations metrics
        b_weight: scalar or (n_points, 1) array representing the b parameters for anisotropy of local dissipations metrics
        edges: (n_edges, 2) array of the connectivity of the lines object
        fun_anisotropy_dir: callable that computes the anisotropy direction of the system
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
    if not isinstance(masses, torch.Tensor):
        masses = torch.tensor(masses)
    if not isinstance(a_weight, torch.Tensor):
        a_weight = torch.tensor(a_weight)
    if not isinstance(b_weight, torch.Tensor):
        b_weight = torch.tensor(b_weight)
    if not isinstance(edges, torch.Tensor):
        edges = torch.tensor(edges)

    tangents_ = fun_anisotropy_dir(pos_)
    pos = vmap_euc_transform_torch(g, pos_)
    tangents = vmap_euc_transform_T_torch(g, tangents_)

    energies = vmap_energy_torch(
        pos[:-1], pos[1:], tangents[:-1], tangents[1:], 
        masses[:-1], masses[1:], a_weight[:-1], a_weight[1:], 
        b_weight[:-1], b_weight[1:], edges,
    )

    return torch.sum(energies)

def grad_energy_path(pos_, g, masses, a_weight, b_weight, edges, fun_anisotropy_dir):
    '''Gradient of the objective with repect to its first input
    
    Args:
        Same as above
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
    pos_torch_ = torch.tensor(pos_, requires_grad=True)
    g_torch = torch.tensor(g, requires_grad=True)
    masses_torch = torch.tensor(masses)
    a_weight_torch = torch.tensor(a_weight)
    b_weight_torch = torch.tensor(b_weight)
    edges_torch = torch.tensor(edges)

    obj = energy_path(
        pos_torch_, g_torch, 
        masses_torch, a_weight_torch, b_weight_torch, edges_torch, fun_anisotropy_dir
    )
    obj.backward(torch.ones_like(obj))
    return g_torch.grad.numpy(), pos_torch_.grad.numpy()

def avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction):
    '''Compute the avoidance of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        
    Returns:
        scalar representing the objective
    '''
    
    if not isinstance(pos_, torch.Tensor):
        pos_ = torch.tensor(pos_)
    if not isinstance(g, torch.Tensor):
        g = torch.tensor(g)
        
    pos = vmap_euc_transform_torch(g, pos_)
    sdfs = implicit.evaluate_implicit_function(pos.reshape(-1, 3))
    obj = torch.mean(torch.maximum(torch.tensor(0.0), - sdfs))
    return obj

def grad_avoid_implicit(pos_: torch.Tensor, g: torch.Tensor, implicit: ImplicitFunction):
    '''Compute the avoidance of the path
    
    Args:
        pos_: (n_steps, n_points, 3) array of the positions of the system
        g: (n_steps, 7) tensor reprenting the registration per time step
        implicit (instance of an ImplicitFunction): the obstacle to avoid
        
    Returns:
        (n_steps, 7) array representing the gradient of the objective with respect to the rigid transformations
        (n_steps, n_points, 3) array representing the gradient of the objective with respect to the unregistered shapes
    '''
        
    pos_torch_ = torch.tensor(pos_)
    pos_torch_.requires_grad = True
    
    g_torch = torch.tensor(g)
    g_torch.requires_grad = True

    obj = avoid_implicit(pos_, g_torch, implicit)
    obj.backward(torch.ones_like(obj))
    
    if g_torch.grad is None:
        grad_g = torch.zeros_like(g_torch)
    else:
        grad_g = g_torch.grad
    if pos_torch_.grad is None:
        grad_pos_ = torch.zeros_like(pos_torch_)
    else:
        grad_pos_ = pos_torch_.grad
    return grad_g.numpy(), grad_pos_.numpy()
