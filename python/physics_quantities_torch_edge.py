import torch
from utils import dot_vec_torch, qmultiply_torch, qinv_torch

TORCH_DTYPE = torch.float64
torch.set_default_dtype(TORCH_DTYPE)

def P_dot_torch(g, P, g_dot):
    '''Compute the sensitivities of the positions P wrt g
    
    Args:
        g: (7,) tensor representing a rigid transformation
        P: (N, 3) tensor the points the transformed points using g
        g_dot: (7,) tensor representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) tensor the sensitivities of the points
    '''
    q_dot = g_dot[:4]
    b_dot = g_dot[4:]
    
    q = g[:4]
    ω_dot = 2.0 * qmultiply_torch(q_dot, qinv_torch(q))[:3].unsqueeze(0)
    
    Pdot = torch.cross(ω_dot, P, dim=-1) + b_dot
    
    return Pdot

def energy_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the energy between time t-1 and t acting on the shape given the shape change
    
    Args:
        pos_s: (n_verts, 3) torch tensor of current positions of the system (including rigid transformation)
        pos_e: (n_verts, 3) torch tensor of next positions of the system (including rigid transformation)
        tang_s: (n_edges, 3) torch tensor of current tangents of the system (including rigid transformation)
        tang_e: (n_edges, 3) torch tensor of next tangents of the system (including rigid transformation)
        mass_s: (n_edges, 1) torch tensor of masses of the current prim gamma_curr
        mass_e: (n_edges, 1) torch tensor of masses of the next prim gamma_next
        a_weight_*: scalar or (n_edges, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (n_edges, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        edges: (n_edges, 2) torch tensor of connectivity between the vertices (allows for closed polylines)
        
    Returns:
        (3,) torch tensor of the force acting on the shape
    '''
    midpoint_s_edges = torch.mean(pos_s[edges], dim=1)
    midpoint_e_edges = torch.mean(pos_e[edges], dim=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges
    midpoint_disp_squared = dot_vec_torch(midpoint_disp, midpoint_disp) # shape (n_edges, 1)

    e1 = torch.sum(mass_s * (a_weight_s * midpoint_disp_squared + b_weight_s * dot_vec_torch(midpoint_disp, tang_s) ** 2))
    e2 = torch.sum(mass_e * (a_weight_e * midpoint_disp_squared + b_weight_e * dot_vec_torch(midpoint_disp, tang_e) ** 2))
    e_avg = 0.5 * (e1 + e2)
    
    return 0.5 * e_avg

vmap_energy_torch = torch.vmap(energy_torch, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None))

def force_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the force acting on the shape given the shape change
    
    Args:
        pos_s: (n_verts, 3) torch tensor of current positions of the system (including rigid transformation)
        pos_e: (n_verts, 3) torch tensor of next positions of the system (including rigid transformation)
        tang_s: (n_edges, 3) torch tensor of current tangents of the system (including rigid transformation)
        tang_e: (n_edges, 3) torch tensor of next tangents of the system (including rigid transformation)
        mass_s: (n_edges, 1) torch tensor of masses of the current prim gamma_curr
        mass_e: (n_edges, 1) torch tensor of masses of the next prim gamma_next
        a_weight_*: scalar or (n_edges, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (n_edges, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        edges: (n_edges, 2) torch tensor of connectivity between the vertices (allows for closed polylines)
        
    Returns:
        (3,) torch tensor of the force acting on the shape
    '''
    midpoint_s_edges = torch.mean(pos_s[edges], dim=1)
    midpoint_e_edges = torch.mean(pos_e[edges], dim=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges

    F1 = - mass_s * (a_weight_s * midpoint_disp + b_weight_s * dot_vec_torch(midpoint_disp, tang_s) * tang_s)
    F2 = - mass_e * (a_weight_e * midpoint_disp + b_weight_e * dot_vec_torch(midpoint_disp, tang_e) * tang_e)
    F_avg = 0.5 * (F1 + F2)
    
    return torch.sum(F_avg, dim=0)

def torque_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the torque acting on the shape given the shape change
    
    Args:
        Same as force
        
    Returns:
        (3,) torch tensor of the torque acting on the shape
    '''
    midpoint_s_edges = torch.mean(pos_s[edges], dim=1)
    midpoint_e_edges = torch.mean(pos_e[edges], dim=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges

    T1 = mass_s * (a_weight_s * midpoint_disp + b_weight_s * dot_vec_torch(midpoint_disp, tang_s) * tang_s)
    T1 = torch.cross(T1, midpoint_e_edges, dim=-1)
    T2 = mass_e * (a_weight_e * midpoint_disp + b_weight_e * dot_vec_torch(midpoint_disp, tang_e) * tang_e)
    T2 = torch.cross(T2, midpoint_s_edges, dim=-1)
    T_avg = 0.5 * (T1 + T2)
    
    return torch.sum(T_avg, dim=0)

def momentum_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev):
    '''Compute the Δforce and Δtorque acting on the shape given the shape change
    
    Args:
        Same as force and torque
        force_prev: (3, 1) torch tensor of the Δforce acting on the shape in the previous iteration
        torque_prev: (3, 1) torch tensor of the Δtorque acting on the shape in the previous iteration
        
    Returns:
        (6,) torch tensor of the Δforce and Δtorque acting on the shape
    '''
    force_ = force_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges) - force_prev 
    torque_ = torque_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges) - torque_prev
    return torch.cat((force_, torque_), dim=0)
