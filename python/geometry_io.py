import json
import torch

def export_snakes_to_json(
    pos_, g, pos, force_0, torque_0, save_path, edges=None,
    weights_optim=None, quantities_per_vertex=None,
    quantities_per_edge=None, target_final_g=None,
    target_checkpoints_g=None, obstacle=None, 
    optimization_settings=None, optimization_duration=None,
    optimization_evolution=None, mass_data=None,
):
    '''
    Args:
        pos_: torch.tensor/np.array of shape (n_ts, N, 3)
        g: torch.tensor/np.array of shape (n_ts, 7)
        pos: torch.tensor/np.array of shape (n_ts, N, 3)
        force_0: torch.tensor/np.array of shape (3,)
        torque_0: torch.tensor/np.array of shape (3,)
        edges: torch.tensor/np.array of shape (M, 2)
        save_path: str
        weights_optim: dict of optimization weights
        quantities_per_vertex: dict of lists of shapes (n_ts, N)
        quantities_per_edge: dict of lists of shapes (n_ts, M)
        target_final_g: torch.tensor/np.array of shape (7,)
        target_checkpoints_g: torch.tensor/np.array of shape (n_checkpoints, 7)
        obstacle: an object of type ImplicitFunction
        optimization_settings: dict of optimization settings
        optimization_duration: float, the duration of the optimization
        optimization_evolution: dict of optimization evolution
        mass_data: dict of masses, may contain logits/min/max masses
    '''
    if edges is None:
        edges = torch.stack([
            torch.arange(0, pos.shape[1]-1, dtype=torch.long),
            torch.arange(1, pos.shape[1], dtype=torch.long),
        ], dim=1)
    
    if weights_optim is None:
        weights_optim = {}
        
    if quantities_per_vertex is None:
        quantities_per_vertex = {}
        
    if quantities_per_edge is None:
        quantities_per_edge = {}
        
    if target_final_g is None:
        target_final_g = torch.tensor([])
        
    if target_checkpoints_g is None:
        target_checkpoints_g = torch.tensor([])
        
    if obstacle is None:
        obstacle_ser = None
    else:
        obstacle_ser = obstacle.serialize()
        
    if optimization_settings is None:
        optimization_settings = {}
        
    if optimization_duration is None:
        optimization_duration = -1.0
        
    if optimization_evolution is None:
        optimization_evolution = {}

    if mass_data is None:
        mass_data = {}

    with open(save_path, 'w') as jsonFile:
        json.dump({
            'pos_': pos_.tolist(),
            'g': g.tolist(),
            'pos': pos.tolist(),
            'force_0': force_0.tolist(),
            'torque_0': torque_0.tolist(),
            'edges': edges.tolist(),
            'weights_optim': weights_optim,
            'quantities_per_vertex': quantities_per_vertex,
            'quantities_per_edge': quantities_per_edge,
            'target_final_g': target_final_g.tolist(),
            'target_checkpoints_g': target_checkpoints_g.tolist(),
            'obstacle': obstacle_ser,
            'optimization_settings': optimization_settings,
            'optimization_duration': optimization_duration,
            'optimization_evolution': optimization_evolution,
            'mass_data': mass_data,
        }, jsonFile, indent=4)
