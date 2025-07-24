import numpy as np
import torch

from physics_quantities_torch_edge import momentum_torch
from scipy.spatial.transform import Rotation as R
from utils import (dot_vec, euc_transform_torch, qmultiply, qinv)

########################    
##### Base Quantities
########################

def energy(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the energy between time t-1 and t acting on the shape given the shape change
    
    Args:
        pos_s: (n_verts, 3) array of current positions of the system (including rigid transformation)
        pos_e: (n_verts, 3) array of next positions of the system (including rigid transformation)
        tang_s: (n_edges, 3) array of current tangents of the system (including rigid transformation)
        tang_e: (n_edges, 3) array of next tangents of the system (including rigid transformation)
        mass_s: (n_edges, 1) array of masses of the current position
        mass_e: (n_edges, 1) array of masses of the next position
        a_weight_*: scalar or (n_edges, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (n_edges, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        edges: (n_edges, 2) array of connectivity between the vertices (allows for closed polylines)
        
    Returns:
        (3,) array of the force acting on the shape
    '''
    midpoint_s_edges = np.mean(pos_s[edges], axis=1)
    midpoint_e_edges = np.mean(pos_e[edges], axis=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges
    midpoint_disp_squared = dot_vec(midpoint_disp, midpoint_disp) # shape (n_edges, 1)

    e1 = np.sum(mass_s * (a_weight_s * midpoint_disp_squared + b_weight_s * dot_vec(midpoint_disp, tang_s) ** 2))
    e2 = np.sum(mass_e * (a_weight_e * midpoint_disp_squared + b_weight_e * dot_vec(midpoint_disp, tang_e) ** 2))
    e_avg = 0.5 * (e1 + e2)
    
    return 0.5 * e_avg

def force(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the force between time t-1 and t acting on the shape given the shape change
    
    Args:
        pos_s: (n_verts, 3) array of current positions of the system (including rigid transformation)
        pos_e: (n_verts, 3) array of next positions of the system (including rigid transformation)
        tang_s: (n_edges, 3) array of current tangents of the system (including rigid transformation)
        tang_e: (n_edges, 3) array of next tangents of the system (including rigid transformation)
        mass_s: (n_edges, 1) array of masses of the current prim gamma_curr
        mass_e: (n_edges, 1) array of masses of the next prim gamma_next
        a_weight_*: scalar or (n_edges, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (n_edges, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        edges: (n_edges, 2) array of connectivity between the vertices (allows for closed polylines)

    Returns:
        (3,) array of the force acting on the shape
    '''
    midpoint_s_edges = np.mean(pos_s[edges], axis=1)
    midpoint_e_edges = np.mean(pos_e[edges], axis=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges

    F1 = - mass_s * (a_weight_s * midpoint_disp + b_weight_s * dot_vec(midpoint_disp, tang_s) * tang_s)
    F2 = - mass_e * (a_weight_e * midpoint_disp + b_weight_e * dot_vec(midpoint_disp, tang_e) * tang_e)
    F_avg = 0.5 * (F1 + F2)
    
    return np.sum(F_avg, axis=0)

def torque(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges):
    '''Compute the torque acting on the shape given the shape change
    
    Args:
        Same as force
        
    Returns:
        (3,) array of the torque acting on the shape
    '''
    midpoint_s_edges = np.mean(pos_s[edges], axis=1)
    midpoint_e_edges = np.mean(pos_e[edges], axis=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges

    T1 = mass_s * (a_weight_s * midpoint_disp + b_weight_s * dot_vec(midpoint_disp, tang_s) * tang_s)
    T1 = np.cross(T1, midpoint_e_edges, axis=-1)
    T2 = mass_e * (a_weight_e * midpoint_disp + b_weight_e * dot_vec(midpoint_disp, tang_e) * tang_e)
    T2 = np.cross(T2, midpoint_s_edges, axis=-1)
    T_avg = 0.5 * (T1 + T2)
    
    return np.sum(T_avg, axis=0)

def momentum(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev):
    '''Compute the Δforce and Δtorque acting on the shape given the shape change
    
    Args:
        Same as force and mtorque
        force_prev: (3, 1) array of the Δforce acting on the shape in the previous iteration
        torque_prev: (3, 1) array of the Δtorque acting on the shape in the previous iteration
        
    Returns:
        (6,) array of the Δforce and Δtorque acting on the shape
    '''
    force_ = force(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges) - force_prev 
    torque_ = torque(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges) - torque_prev
    return np.concatenate((force_, torque_), axis=0)

############################    
##### First-Order Quantities
############################

def P_dot(g, P, g_dot):
    '''Compute the sensitivities of the positions P wrt g
    
    Args:
        g: (7,) array representing a rigid transformation
        P: (N, 3) array the points the transformed points using g
        g_dot: (7,) array representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    q_dot = g_dot[:4]
    b_dot = g_dot[4:]
    
    q = g[:4]
    b = g[4:].reshape(1, 3)
    ω_dot = 2.0 * qmultiply(q_dot, qinv(q))[:3]
    
    Pdot = np.cross(ω_dot, P - b, axis=-1) + b_dot
    
    return Pdot

def dot_P(g, P_, P_dot):
    '''Compute the pullback of P_dot wrt g: P_dot^T.∂P/∂g
    TODO: Replace with analytical computation
    
    Args:
        g: (7,) array representing a rigid transformation
        P: (N, 3) array the points the transformed points using g
        P_dot: (7,) array representing the time derivative of the rigid transformation
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    
    P_torch_ = torch.tensor(P_)
    P_torch_.requires_grad = True
    P = euc_transform_torch(torch.tensor(g), P_torch_)
    P.backward(P_dot)
    
    return P_torch_.grad.numpy()

def P0_dot(g, P0, P0_dot):
    '''Compute the sensitivities of the positions P wrt P_0
    
    Args:
        g: (7,) array representing a rigid transformation
        P0: (N, 3) array the points to be transformed
        P0_dot: (N, 3) array the change in the points to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the points
    '''
    rot = R.from_quat(g[:4])
    return rot.apply(P0_dot)

def dot_P0(g, P, P_dot):
    '''Compute the pullback of P_dot wrt P0
    
    Args:
        g: (7,) array representing a rigid transformation
        P0: (N, 3) array the points to be transformed
        P_dot: (N, 3) array the vector field to be pulled
        
    Returns:
        (N, 3) array the pullback of the points
    '''
    rot = R.from_quat(qinv(g[:4])) # rot.T
    return rot.apply(P_dot)
    
def T_dot(g, T, g_dot):
    '''Compute the sensitivities of the tangents T wrt g
    
    Args:
        Same as P_dot
        T: (N, 3) array the tangents to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the tangents
    '''
    q_dot = g_dot[:4]
    
    q = g[:4]
    ω_dot = 2.0 * qmultiply(q_dot, qinv(q))[:3]
    
    Tdot = np.cross(ω_dot, T, axis=-1)
    
    return Tdot

def T0_dot(g, T0, T0_dot):
    '''Compute the sensitivities of the positions P wrt P_0
    
    Args:
        g: (7,) array representing a rigid transformation
        T0: (N, 3) array the tangents to be transformed
        T0_dot: (N, 3) array the change in the tangents to be transformed
        
    Returns:
        (N, 3) array the sensitivities of the tangents
    '''
    rot = R.from_quat(g[:4])
    return rot.apply(T0_dot)

def force_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, P_dot, T_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=True):
    '''Compute the sensitivities of the force acting on the shape given the shape change
    
    Args:
        Same as force and torque
        P_dot: (n_verts, 3) array of the changes of the positions of the system
        T_dot: (n_edges, 3) array of the changes of the tangents of the system
        wrt_end: whether P_dot and T_dot represent changes at the begining or at the end of the time interval
        
    Returns:
        (3,) array of the time derivative of the force acting on the shape
    '''

    midpoint_s_edges = np.mean(pos_s[edges], axis=1)
    midpoint_e_edges = np.mean(pos_e[edges], axis=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges
    midpoint_dot = np.mean(P_dot[edges], axis=1)
    
    if wrt_end:
        F1_dot = a_weight_s * midpoint_dot + b_weight_s * dot_vec(midpoint_dot, tang_s) * tang_s
        F2_dot = a_weight_e * midpoint_dot + b_weight_e * dot_vec(midpoint_dot, tang_e) * tang_e
        F2_dot += b_weight_e * (dot_vec(midpoint_disp, T_dot) * tang_e + dot_vec(midpoint_disp, tang_e) * T_dot)
    else:
        F1_dot = - a_weight_s * midpoint_dot - b_weight_s * dot_vec(midpoint_dot, tang_s) * tang_s
        F1_dot += b_weight_s * (dot_vec(midpoint_disp, T_dot) * tang_s + dot_vec(midpoint_disp, tang_s) * T_dot)
        F2_dot = - a_weight_e * midpoint_dot - b_weight_e * dot_vec(midpoint_dot, tang_e) * tang_e
    F1_dot = - mass_s * F1_dot
    F2_dot = - mass_e * F2_dot
    F_avg_dot = 0.5 * (F1_dot + F2_dot)
    
    return np.sum(F_avg_dot, axis=0)
    
        
def torque_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, P_dot, T_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=True):
    '''Compute the sensitivities of the torque acting on the shape given the shape change
    
    Args:
        Same as force and torque
        P_dot: (n_verts, 3) array of the changes of the positions of the system
        T_dot: (n_edges, 3) array of the changes of the tangents of the system
        wrt_end: whether P_dot and T_dot represent changes at the begining or at the end of the time interval
        
    Returns:
        (3,) array of the time derivative of the torque acting on the shape
    '''
    midpoint_s_edges = np.mean(pos_s[edges], axis=1)
    midpoint_e_edges = np.mean(pos_e[edges], axis=1)
    midpoint_disp = midpoint_e_edges - midpoint_s_edges
    midpoint_dot = np.mean(P_dot[edges], axis=1)

    ΔP = pos_e - pos_s

    if wrt_end:
        t1 = a_weight_s * midpoint_disp + b_weight_s * dot_vec(midpoint_disp, tang_s) * tang_s
        t1_dot = a_weight_s * midpoint_dot + b_weight_s * dot_vec(midpoint_dot, tang_s) * tang_s
        T1_dot = mass_s * (np.cross(t1, midpoint_dot, axis=-1) + np.cross(t1_dot, midpoint_e_edges, axis=-1))
        
        T2_dot = a_weight_e * midpoint_dot + b_weight_e * dot_vec(midpoint_dot, tang_e) * tang_e
        T2_dot += b_weight_e * (dot_vec(midpoint_disp, T_dot) * tang_e + dot_vec(midpoint_disp, tang_e) * T_dot)
        T2_dot = mass_e * np.cross(T2_dot, midpoint_s_edges, axis=-1)
    else:
        T1_dot = - a_weight_s * midpoint_dot - b_weight_s * dot_vec(midpoint_dot, tang_s) * tang_s
        T1_dot += b_weight_s * (dot_vec(midpoint_disp, T_dot) * tang_s + dot_vec(midpoint_disp, tang_s) * T_dot)
        T1_dot = mass_s * np.cross(T1_dot, midpoint_e_edges, axis=-1)

        t2 = a_weight_e * midpoint_disp + b_weight_e * dot_vec(midpoint_disp, tang_e) * tang_e
        t2_dot = - a_weight_e * midpoint_dot - b_weight_e * dot_vec(midpoint_dot, tang_e) * tang_e
        T2_dot = mass_e * (np.cross(t2, midpoint_dot, axis=-1) + np.cross(t2_dot, midpoint_s_edges, axis=-1))
    T_avg_dot = 0.5 * (T1_dot + T2_dot)

    return np.sum(T_avg_dot, axis=0)

def momentum_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g, g_dot, wrt_end=True):
    '''Compute the JVP of the momentum at time t given the positioning change: ∂μ/∂g.g_dot at time t-1 or t
    
    In particular, we interpret g and g_dot as the rigid transformation at time t-1 or t-1 and the
    inifinitesimal change in the rigid transformation at time t-1 or t. The momentum is always evaluated
    in the time interval t-1 to t.
    
    Args:
        Same as force and torque
        g: (7,) array representing a rigid transformation
        g_dot: (7,) array representing the infinitesimal change of the rigid transformation
        wrt_end: whether g and g_dot represent the positioning at the begining or at the end of the time interval
        
    Returns:
        (6,) array of the JVP of the momentum
    '''
    if wrt_end:
        Pse_dot = P_dot(g, pos_e, g_dot)
        Tse_dot = T_dot(g, tang_e, g_dot)
    else:
        Pse_dot = P_dot(g, pos_s, g_dot)
        Tse_dot = T_dot(g, tang_s, g_dot)
    
    force_dot_ = force_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, Pse_dot, Tse_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=wrt_end)
    torque_dot_ = torque_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, Pse_dot, Tse_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=wrt_end)
    
    return np.concatenate((force_dot_ , torque_dot_), axis=0)  

def momentum_dot_(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g, Pe0_dot, Te0_dot, wrt_end=True):
    '''Compute the JVP of the momentum at time t given the shape variation at time t-1 or t'''
    Pe_dot = P0_dot(g, None, Pe0_dot)
    Te_dot = T0_dot(g, None, Te0_dot)
    
    force_dot_ = force_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, Pe_dot, Te_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=wrt_end)
    torque_dot_ = torque_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, Pe_dot, Te_dot, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, wrt_end=wrt_end)
    
    return np.concatenate((force_dot_ , torque_dot_), axis=0)  

def dot_momentum_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ, wrt_end=True):
    '''
    Compute the VJP of a momentum change (δμ^T.dμ/dP_)^T with respect to the reference positions.
    The momentum is evaluated at time t, and differentiated against the position at time t-1 or t depending on wrt_end.
    '''
    
    pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ = map(torch.tensor, (pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ))
    if wrt_end: 
        pos_e_.requires_grad = True
    else:
        pos_s_.requires_grad = True
        
    pos_s = euc_transform_torch(gs, pos_s_)
    pos_e = euc_transform_torch(ge, pos_e_)
        
    tang_s = torch.zeros_like(pos_s)
    tang_s[:-1] = pos_s[1:] - pos_s[:-1]
    tang_s[-1] = pos_s[-1] - pos_s[-2]
    tang_s = tang_s / torch.linalg.norm(tang_s, dim=-1, keepdims=True)
    
    tang_e = torch.zeros_like(pos_e)
    tang_e[:-1] = pos_e[1:] - pos_e[:-1]
    tang_e[-1] = pos_e[-1] - pos_e[-2]
    tang_e = tang_e / torch.linalg.norm(tang_e, dim=-1, keepdims=True)
    
    dot_momentum = δμ @ momentum_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, torch.zeros(size=(3,)), torch.zeros(size=(3,)))
    dot_momentum.backward(torch.ones_like(dot_momentum))
    
    if wrt_end: 
        return pos_e_.grad.numpy()
    else:
        return pos_s_.grad.numpy()
    

def dot_momentum_material(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ, wrt_end=True):
    '''
    Compute the VJP of a momentum change (δμ^T.dμ/dab)^T with respect to the material parameters a_weight and b_weight
    The momentum is evaluated at time t, and differentiated against the material parameters a_weight and b_weight
    '''
    
    pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ = map(torch.tensor, (pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ))
    if wrt_end: 
        mass_e.requires_grad = True
        a_weight_e.requires_grad = True
        b_weight_e.requires_grad = True
    else:
        mass_s.requires_grad = True
        a_weight_s.requires_grad = True
        b_weight_s.requires_grad = True
    
    dot_momentum = δμ @ momentum_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, torch.zeros(size=(3,)), torch.zeros(size=(3,)))
    dot_momentum.backward(torch.ones_like(dot_momentum))
    
    if wrt_end: 
        return mass_e.grad.numpy(), a_weight_e.grad.numpy(), b_weight_e.grad.numpy()
    else:
        return mass_s.grad.numpy(), a_weight_s.grad.numpy(), b_weight_s.grad.numpy()
    
def dot_momentum_material_pos_(pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ, fun_anisotropy_dir, wrt_end=True):
    '''
    Compute the VJP of a momentum change (δμ^T.dμ/dP_)^T with respect to the reference positions and the material parameters ab.
    The momentum is evaluated at time t, and differentiated against the position at time t-1 or t depending on wrt_end.
    '''
    
    pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ = map(torch.tensor, (pos_s_, pos_e_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, gs, ge, δμ))
    if wrt_end: 
        pos_e_.requires_grad = True
        mass_e.requires_grad = True
        a_weight_e.requires_grad = True
        b_weight_e.requires_grad = True
    else:
        pos_s_.requires_grad = True
        mass_s.requires_grad = True
        a_weight_s.requires_grad = True
        b_weight_s.requires_grad = True
        
    pos_s = euc_transform_torch(gs, pos_s_)
    pos_e = euc_transform_torch(ge, pos_e_)

    tang_s = fun_anisotropy_dir(pos_s)
    tang_e = fun_anisotropy_dir(pos_e)
    
    dot_momentum = δμ @ momentum_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, torch.zeros(size=(3,)), torch.zeros(size=(3,)))
    dot_momentum.backward(torch.ones_like(dot_momentum))
    
    if wrt_end: 
        return mass_e.grad.numpy(), a_weight_e.grad.numpy(), b_weight_e.grad.numpy(), pos_e_.grad.numpy()
    else:
        return mass_s.grad.numpy(), a_weight_s.grad.numpy(), b_weight_s.grad.numpy(), pos_s_.grad.numpy()

def dot_momentum_material_constant(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight, b_weight, edges, gs, ge, δμ):
    '''
    Compute the VJP of a momentum change (δμ^T.dμ/dab)^T with respect to the material parameters a_weight and b_weight.
    The momentum is evaluated at time t, and differentiated against the material parameters a_weight and b_weight.
    As opposed to the previous function, the material parameters are constant.
    '''
    
    pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight, b_weight, edges, gs, ge, δμ = map(torch.tensor, (pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight, b_weight, edges, gs, ge, δμ))
    a_weight.requires_grad = True
    b_weight.requires_grad = True
    
    dot_momentum = δμ @ momentum_torch(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight, a_weight, b_weight, b_weight, edges, torch.zeros(size=(3,)), torch.zeros(size=(3,)))
    dot_momentum.backward(torch.ones_like(dot_momentum))

    return a_weight.grad.numpy(), b_weight.grad.numpy()
