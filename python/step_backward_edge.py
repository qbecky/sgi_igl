import numpy as np
from physics_quantities_edge import momentum_dot, dot_momentum_, dot_momentum_material, dot_momentum_material_constant, dot_momentum_material_pos_
from scipy.linalg import lstsq
import torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def compute_last_adjoint(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, grad_obj):
    '''
    Compute the previous adjoint vector at time T

    Args:
        pos_s: (n_verts, 3) array the start positions of the system (including rigid transformation), time T-1
        pos_e: (n_verts, 3) array the end positions of the system (including rigid transformation), time T
        tang_s: (n_edges, 3) array the start tangents of the system (including rigid transformation), time T-1
        tang_e: (n_edges, 3) array the end tangents of the system (including rigid transformation), time T
        a_weight: anisotropy of local dissipations metrics in the direction of the tangent
        b_weight: b_weight for anisotropy of local dissipations metrics
        edges: (n_edges, 2) array of connectivity between the vertices (allows for closed polylines)
        ge: (7,) array representing a rigid transformation at the end of the time interval, time T
        grad_obj: (7,) gradient of the objective with respect to the final rigid transformation, time T

    Returns:
        we: (6,) the adjoint vector at the end of the time interval
    '''
    dμ_dge = np.array([momentum_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, ei, wrt_end=True) for ei in np.identity(7)]).T
    we = lstsq(dμ_dge.T, - grad_obj)[0]
    return we

def compute_last_gradient_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we):
    '''
    Compute the gradient of the objective wrt the shape at time T

    Args:
        Same as above
        pos_s_: (n_verts, 3) array the start positions of the system (excluding rigid transformation), time T-1
        pos_e_: (n_verts, 3) array the end positions of the system (excluding rigid transformation), time T
        gs: (7,) array representing a rigid transformation at the start of the time interval, time T-1
        ge: (7,) array representing a rigid transformation at the end of the time interval, time T
        we: (6,) the adjoint vector at the start of the time interval, time T

    Returns:
        dJ_dpos_e_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time T
    '''
    return (
        dot_momentum_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, wrt_end=True)
    )
    
def compute_last_gradient_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we):
    '''
    Compute the gradient of the objective wrt the material parameters at time T

    Args:
        Same as above

    Returns:
        dJ_dmass_e: (n_edges, 1) array the gradient of the cost wrt the masses, time T
        dJ_dae: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time T
        dJ_dbe: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time T
    '''
    return (
        dot_momentum_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, wrt_end=True)
    )
    
def compute_last_gradient_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, fun_anisotropy_dir):
    '''
    Compute the gradient of the objective wrt the material parameters and the shape at time T

    Args:
        Same as above
        fun_anisotropy_dir: function that returns the tangents/normals of the system at time 0

    Returns:
        dJ_dmass_e: (n_edges, 1) array the gradient of the cost wrt the masses, time T
        dJ_dae: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time T
        dJ_dbe: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time T
        dJ_dpos_e_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time T
    '''
    dm_mass, dm_a, dm_b, dm_pos_ = dot_momentum_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, fun_anisotropy_dir, wrt_end=True)
    return (dm_mass, dm_a, dm_b, dm_pos_)
    
def compute_first_gradient_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we):
    '''
    Compute the gradient of the objective wrt the shape at time 0

    Args:
        Same as above

    Returns:
        dJ_dpos_s_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time 0
    '''
    return (
        dot_momentum_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, wrt_end=False)
    )

def compute_first_gradient_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we):
    '''
    Compute the gradient of the objective wrt the material parameters at time T

    Args:
        Same as above

    Returns:
        dJ_dmass_s: (n_edges, 1) array the gradient of the cost wrt the masses, time 0
        dJ_das: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time 0
        dJ_dbs: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time 0
    '''
    return (
        dot_momentum_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, wrt_end=False)
    )
    
def compute_first_gradient_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, fun_anisotropy_dir):
    '''
    Compute the gradient of the objective wrt the material parameters and the untransformed positions at time T

    Args:
        Same as above
        fun_anisotropy_dir: function that returns the tangents/normals of the system at time 0

    Returns:
        dJ_dmass_s: (n_edges, 1) array the gradient of the cost wrt the masses, time 0
        dJ_das: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time 0
        dJ_dbs: (n_edges, 1) array the gradient of the cost wrt the last material parameters, time 0
        dJ_dpos_s_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time 0
    '''
    dm_mass, dm_a, dm_b, dm_pos_ = dot_momentum_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, we, fun_anisotropy_dir, wrt_end=False)
    return (dm_mass, dm_a, dm_b, dm_pos_)

def step_backward_adjoint(pos_s, pos_e, pos_e_next, tang_s, tang_e, tang_e_next, mass_s, mass_e, mass_e_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, edges, ge, we, grad_obj):
    '''
    Compute the previous adjoint vector at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        pos_e_next: (n_verts, 3) array the end positions of the system (including rigid transformation), time t+1
        tang_e_next: (n_edges, 3) array the end tangents of the system (including rigid transformation), time t+1
        mass_e_next: (n_edges, 1) array the masses time t+1
        a_weight_e_next: (n_edges, 1) array anisotropy of local dissipations metrics in the direction of the tangent, time t+1
        b_weight_e_next: (n_edges, 1) array b_weight for anisotropy of local dissipations metrics, time t+1
        we: (6,) the adjoint vector at the end of the time interval, time t+1
        grad_obj: (7,) gradient of the objective with respect to the rigid transformation, time t

    Returns:
        ws: (6,) the adjoint vector at the start of the time interval, time t
    '''

    dμ_dge = np.array([momentum_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, ei, wrt_end=True) for ei in np.identity(7)]).T
    dμ_dgs_next = np.array([momentum_dot(pos_e, pos_e_next, tang_e, tang_e_next, mass_e, mass_e_next, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, edges, ge, ei, wrt_end=False) for ei in np.identity(7)]).T
    
    ws = lstsq(dμ_dge.T, - dμ_dgs_next.T @ we - grad_obj)[0]
    
    return ws

def step_backward_grad_pos_(pos_s_, pos_e_, pos_e_next_, mass_s, mass_e, mass_e_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, edges, gs, ge, ge_next, we, we_next):
    '''
    Compute the gradient at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        gs: (7,) array representing a rigid transformation at the start of the time interval, time t-1
        we: (6,) the adjoint vector at the start of the time interval, time t
        we_next: (6,) the adjoint vector at the end of the time interval, time t+1

    Returns:
        dJ_dpos_e_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time t
    '''
    return (
        dot_momentum_(pos_e_, pos_e_next_, mass_e, mass_e_next, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, ge_next, we_next, wrt_end=False) +
        dot_momentum_(pos_s_, pos_e_, mass_s, mass_e, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, edges, gs, ge, we, wrt_end=True)
    )
    
def step_backward_grad_material(pos_s, pos_e, pos_e_next, tang_s, tang_e, tang_e_next, mass_s, mass_e, mass_e_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, edges, gs, ge, ge_next, we, we_next):
    '''
    Compute the gradient of the objective wrt the material parameters at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        gs: (7,) array representing a rigid transformation at the start of the time interval, time t-1
        we: (6,) the adjoint vector at the start of the time interval, time t
        we_next: (6,) the adjoint vector at the end of the time interval, time t+1

    Returns:
        dJ_dae: (n_edges, 1) array the gradient of the cost wrt the untransformed positions, time t
        dJ_dbe: (n_edges, 1) array the gradient of the cost wrt the untransformed positions, time t
    '''
    
    dJ_das_next, dJ_dbs_next = dot_momentum_material(pos_e, pos_e_next, tang_e, tang_e_next, mass_e, mass_e_next, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, ge_next, we_next, wrt_end=False)
    dJ_dae, dJ_dbe = dot_momentum_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, edges, gs, ge, we, wrt_end=True)
    
    return (
        dJ_das_next + dJ_dae,
        dJ_dbs_next + dJ_dbe
    )
    
def step_backward_grad_material_pos_(pos_s_, pos_e_, pos_e_next_, mass_s, mass_e, mass_e_next, a_weight_s, a_weight_e, a_weight_e_next, b_weight_s, b_weight_e, b_weight_e_next, edges, gs, ge, ge_next, we, we_next, fun_anisotropy_dir):
    '''
    Compute the gradient at time t

    Args:
        Same as above, except that positions and tangents are now transformed
        gs: (7,) array representing a rigid transformation at the start of the time interval, time t-1
        we: (6,) the adjoint vector at the start of the time interval, time t
        we_next: (6,) the adjoint vector at the end of the time interval, time t+1

    Returns:
        dJ_dmass_e: (n_edges, 1) array the gradient of the cost wrt the masses, time t
        dJ_dae: (n_edges, 1) array the gradient of the cost wrt the untransformed positions, time t
        dJ_dbe: (n_edges, 1) array the gradient of the cost wrt the untransformed positions, time t
        dJ_dpos_e_: (n_verts, 3) array the gradient of the cost wrt the untransformed positions, time t
    '''
    dJ_dmass_next, dJ_das_next, dJ_dbs_next, dJ_dpos_next = dot_momentum_material_pos_(pos_e_, pos_e_next_, mass_e, mass_e_next, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, ge, ge_next, we_next, fun_anisotropy_dir, wrt_end=False)
    dJ_dmass_e, dJ_dae, dJ_dbe, dJ_dpose = dot_momentum_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_e, a_weight_e_next, b_weight_e, b_weight_e_next, edges, gs, ge, we, fun_anisotropy_dir, wrt_end=True)
    
    return (
        dJ_dmass_next + dJ_dmass_e,
        dJ_das_next + dJ_dae,
        dJ_dbs_next + dJ_dbe,
        dJ_dpos_next + dJ_dpose,
    )

def multiple_steps_backward_material_pos_(pos_, pos, tangents, masses, a_weights, b_weights, edges, g, grad_obj_g, grad_obj_pos_, fun_anisotropy_dir):
    '''
    Now computes the gradients with respect to the positions and the materials parameters at once
    
    Args:
        Same as above
        grad_obj_g: the gradient of the objective with respect to the rigid transformations, ∂J/∂P.∂P/∂g or ∂J/∂g if J is directly a function of the rigid transformation, shape (n_steps, 7)
        grad_obj_pos_: the partial gradient of the objective with respect to the unregistered positions, ∂J/∂P.∂P/∂P_, shape (n_steps, 3*n_points) flattened from (n_steps, n_points, 3)
        
    Returns:
        grad_masses: (n_step, n_edges) the gradient of the objective with respect to the masses
        grad_a_obj: (n_step, n_edges) the gradient of the objective with respect to the a weights
        grad_b_obj: (n_step, n_edges) the gradient of the objective with respect to the b weights
        grad_mu0_obj: (6,) the gradient of the objective with respect to the initial momentum
        grad_shape_obj: (n_step, 3*n_verts) the gradient of the objective with respect to the non rigidly transformed points
    '''
    n_steps = pos.shape[0]

    # Compute the last adjoint
    adjoints = np.zeros(shape=(n_steps, 6))
    adjoints[-1] = compute_last_adjoint(
        pos[-2], pos[-1], tangents[-2], tangents[-1], 
        masses[-2].reshape(-1, 1), masses[-1].reshape(-1, 1), 
        a_weights[-2].reshape(-1, 1), a_weights[-1].reshape(-1, 1), 
        b_weights[-2].reshape(-1, 1), b_weights[-1].reshape(-1, 1), 
        edges, g[-1], grad_obj_g[-1]
    )
    
    # Compute the gradients
    grad_masses = np.zeros(shape=(n_steps, edges.shape[0]))
    grad_a_obj = np.zeros(shape=(n_steps, edges.shape[0]))
    grad_b_obj = np.zeros(shape=(n_steps, edges.shape[0]))
    grad_shape_obj = np.zeros(shape=(n_steps, 3 * pos.shape[1]))
    
    grad_mass_obj_tmp, grad_a_obj_tmp, grad_b_obj_tmp, grad_shape_obj_tmp = compute_last_gradient_material_pos_(
        pos_[-2], pos_[-1], 
        masses[-2].reshape(-1, 1), masses[-1].reshape(-1, 1), 
        a_weights[-2].reshape(-1, 1), a_weights[-1].reshape(-1, 1), b_weights[-2].reshape(-1, 1), b_weights[-2].reshape(-1, 1),
        edges, g[-2], g[-1], adjoints[-1], fun_anisotropy_dir,
    )

    grad_masses[-1] = grad_mass_obj_tmp.reshape(-1,)
    grad_a_obj[-1] = grad_a_obj_tmp.reshape(-1,)
    grad_b_obj[-1] = grad_b_obj_tmp.reshape(-1,)
    grad_shape_obj[-1] = grad_shape_obj_tmp.reshape(-1,)

    # Loop over time
    for step in np.arange(1, n_steps-1)[::-1]:
        
        adjoints[step] = step_backward_adjoint(
            pos[step-1], pos[step], pos[step+1], 
            tangents[step-1], tangents[step], tangents[step+1],
            masses[step-1].reshape(-1, 1), masses[step].reshape(-1, 1), masses[step+1].reshape(-1, 1),
            a_weights[step-1].reshape(-1, 1), a_weights[step].reshape(-1, 1), a_weights[step+1].reshape(-1, 1), 
            b_weights[step-1].reshape(-1, 1), b_weights[step].reshape(-1, 1), b_weights[step+1].reshape(-1, 1), 
            edges, g[step], adjoints[step+1], grad_obj_g[step]
        )
        
        grad_mass_obj_tmp, grad_a_obj_tmp, grad_b_obj_tmp, grad_shape_obj_tmp = step_backward_grad_material_pos_(
            pos_[step-1], pos_[step], pos_[step+1], 
            masses[step-1].reshape(-1, 1), masses[step].reshape(-1, 1), masses[step+1].reshape(-1, 1),
            a_weights[step-1].reshape(-1, 1), a_weights[step].reshape(-1, 1), a_weights[step+1].reshape(-1, 1), 
            b_weights[step-1].reshape(-1, 1), b_weights[step].reshape(-1, 1), b_weights[step+1].reshape(-1, 1), 
            edges, g[step-1], g[step], g[step+1], adjoints[step], adjoints[step+1], fun_anisotropy_dir,
        )
        grad_masses[step] = grad_mass_obj_tmp.reshape(-1,)
        grad_a_obj[step] = grad_a_obj_tmp.reshape(-1,)
        grad_b_obj[step] = grad_b_obj_tmp.reshape(-1,)
        grad_shape_obj[step] = grad_shape_obj_tmp.reshape(-1,)
    
    grad_mass_obj_tmp, grad_a_obj_tmp, grad_b_obj_tmp, grad_shape_obj_tmp = compute_first_gradient_material_pos_(
        pos_[0], pos_[1], 
        masses[0].reshape(-1, 1), masses[1].reshape(-1, 1), 
        a_weights[0].reshape(-1, 1), a_weights[1].reshape(-1, 1), 
        b_weights[0].reshape(-1, 1), b_weights[1].reshape(-1, 1), 
        edges, g[0], g[1], adjoints[1], fun_anisotropy_dir,
    )
    grad_masses[0] = grad_mass_obj_tmp.reshape(-1,)
    grad_a_obj[0] = grad_a_obj_tmp.reshape(-1,)
    grad_b_obj[0] = grad_b_obj_tmp.reshape(-1,)
    grad_shape_obj[0] = grad_shape_obj_tmp.reshape(-1,)
    
    grad_shape_obj = grad_shape_obj + grad_obj_pos_
    grad_mu0_obj = - np.sum(adjoints, axis=0)

    return grad_masses, grad_a_obj, grad_b_obj, grad_mu0_obj, grad_shape_obj
