from functools import partial
import numpy as np
from scipy import optimize
from physics_quantities_edge import (momentum, momentum_dot)
from utils import (dot_vec, euc_transform, euc_transform_T)

def cons_dot(g, g_dot):
    '''Compute the time derivative of the quaternion constraint q1^2 + q2^2 + q3^2 + q0^2 = 1'''
    return 2.0 * np.dot(g[:4], g_dot[:4]) 
    
########################    
##### Actual computation
########################
        
def F(pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev, g):
    '''The function we want to find the root of'''
    pos_e = euc_transform(g, pos_e_)
    tang_e = euc_transform_T(g, Te_)

    ELequations = momentum(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev)
    qconstraint = np.array([np.dot(g[:4], g[:4]) - 1.0])
    return np.concatenate((ELequations, qconstraint))
    
def F_dot(pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g, g_dot):
    '''The JVP of the function we want to find the root of'''
    pos_e = euc_transform(g, pos_e_)
    tang_e = euc_transform_T(g, Te_)

    consdot = np.array([cons_dot(g, g_dot)])
    return np.concatenate((momentum_dot(pos_s, pos_e, tang_s, tang_e, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g, g_dot), consdot), axis=0)
    
def F_Jac(pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g):
    '''The Jacobian of the function we want to find the root of'''
    Jac = np.array([F_dot(pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, g, ei) for ei in np.identity(7)])
    return Jac.T

def print_unsuccessful(sol):
    print("====== Unsuccessful root finding =====")
    print("Status: {} ({} function evaluations, {} Jacobian evaluations)".format(sol.status, sol.nfev, sol.njev))
    with np.printoptions(precision=3, suppress=True):
        print("Solution (quaternion norm): {} ({:.2e})".format(sol.x, np.linalg.norm(sol.x[:4])))
        print("Function value (norm): {} ({:.2e})\n".format(sol.fun, np.linalg.norm(sol.fun)))

def step_forward(pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev, g_curr, g_guess=None, options=None):
    '''
    Compute the next rigid transformation ge = [q1, q2, q3, q0, b1, b2, b3]

    Args:
        pos_s: (N, 3) array the current positions of the system (including rigid transformation)
        pos_e_: (N, 3) array the next positions of the system (excluding rigid transformation)
        tang_s: (N, 3) array the current tangents of the system (including rigid transformation)
        Te_: (N, 3) array the next tangents of the system (excluding rigid transformation)
        mass_s: (N, 1) array the masses of the current prim gamma_curr
        mass_e: (N, 1) array the masses of the next prim gamma_next
        a_weight_*: scalar or (N, 1) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weight_*: scalar or (N, 1) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        force_prev: (3,) array of the Δforce acting on the shape in the previous iteration
        torque_prev: (3,) array of the Δtorque acting on the shape in the previous iteration
        g_curr: (7,) array representing a rigid transformation
        g_guess: (7,) array representing a guess for the next rigid transformation (may come from a previous optimization step)
        options: a dictionary of options for the root finder

    Returns:
        pos_e: (N, 3) array representing the next positions
        tang_e: (N, 3) array representing the next tangents
        ge: (7,) array representing the next rigid transformation
    '''
    if options is None:
        # options = {'ftol': 1.0e-3}
        options = {'maxfev': 1000}

    F_curr = partial(F, pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges, force_prev, torque_prev)
    F_Jac_curr = partial(F_Jac, pos_s, pos_e_, tang_s, Te_, mass_s, mass_e, a_weight_s, a_weight_e, b_weight_s, b_weight_e, edges)

    successful_guess = False
    sol_guess = None
    fun_guess = np.inf
    if g_guess is not None:
        sol_guess = optimize.root(F_curr, g_guess, jac=F_Jac_curr, method='hybr', options=options)
        successful_guess = sol_guess.success
        if not successful_guess:
            print_unsuccessful(sol_guess)
            print("Trying with the registration from the previous step...")
            fun_guess = np.linalg.norm(sol_guess.fun)
        else:
            ge = sol_guess.x

    successful_curr = False
    sol_curr = None
    fun_curr = np.inf
    if not successful_guess:
        sol_curr = optimize.root(F_curr, g_curr, jac=F_Jac_curr, method='hybr', options=options)
        successful_curr = sol_curr.success
        if not successful_curr:
            print_unsuccessful(sol_curr)
            fun_curr = np.linalg.norm(sol_curr.fun)
        else:
            ge = sol_curr.x
    
    if not successful_guess and not successful_curr:
        if fun_guess < fun_curr:
            print("Using the guess.")
            ge = sol_guess.x
        else:
            print("Using the current.")
            ge = sol_curr.x

    ge[:4] = ge[:4] / np.linalg.norm(ge[:4]) # unnecessary, but just to be sure
    
    pos_e = euc_transform(ge, pos_e_) # new positions
    tang_e = euc_transform_T(ge, Te_) # new tangents
    return pos_e, tang_e, ge

def multiple_steps_forward(pos_, tangents_, masses, a_weights, b_weights, edges, force_0, torque_0, g0=None, g_guess=None, options=None):
    '''
    Use step_forward multiple times and return the resulting rigidly transformed geometry
    
    Args:
        pos_: (n_steps, n_points, 3) the positions of the non rigidly transformed points
        tangents_: (n_steps, n_points, 3) the tangents of the non rigidly transformed points
        masses: (n_steps, n_points) the mass allocated to each point through time
        a_weights: (n_steps) or (n_steps, n_points) array representing the a parameters for anisotropy of local dissipations metrics, current or next
        b_weights: (n_steps) or (n_steps, n_points) array representing the b parameters for anisotropy of local dissipations metrics, current or next
        force_0: (3,) array of the Δforce acting on the shape in the first iteration
        torque_0: (3,) array of the Δtorque acting on the shape in the first iteration
        g0: (7,) array representing a guess for the first rigid transformation
        g_guess: (n_steps, 7) array representing a guess for the rigid transformations of the current step
        options: a dictionary of options for the root finder for each step
        
    Returns:
        pos: (n_steps, n_points, 3) the positions of the rigidly transformed points
        tangents: (n_steps, n_points, 3) the tangents of the rigidly transformed points
        g: (n_steps, 7) the corresponding optimal rigid transformations
    '''
    pos = pos_.copy()
    n_steps = pos_.shape[0]
    tangents = tangents_.copy()
    g = np.zeros(shape=(n_steps, 7))
    g[:, 3] = 1.0
    if g0 is not None:
        g = np.repeat(g0.reshape(1, -1), n_steps, axis=0)
        pos[0] = euc_transform(g0, pos_[0]) # new positions
        tangents[0] = euc_transform(g0, tangents_[0]) # new tangents

    for step in range(n_steps-1):
        
        if g_guess is not None:
            g_guess_step = g_guess[step+1]
        else:
            g_guess_step = None
        
        pos_e, tang_e, ge = step_forward(
            pos[step], pos_[step+1], tangents[step], tangents_[step+1], 
            masses[step].reshape(-1, 1), masses[step+1].reshape(-1, 1), 
            a_weights[step].reshape(-1, 1), a_weights[step+1].reshape(-1, 1), 
            b_weights[step].reshape(-1, 1), b_weights[step+1].reshape(-1, 1), 
            edges, force_0, torque_0, g[step], g_guess=g_guess_step, options=options,
        )

        pos[step+1] = pos_e.copy()
        tangents[step+1] = tang_e.copy()
        g[step+1] = ge.copy()
    
    # Make sure the first transformation is the identity
    #g[0] = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0 ,0.0])
        
    return pos, tangents, g
