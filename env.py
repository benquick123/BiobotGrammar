import numpy as np
from utils import apply_action_clipping_sim


class SimEnvWrapper:
    
    def __init__(self, make_sim_and_task_fn, load=False):
        self.env, self.task = make_sim_and_task_fn()
        
        if load:
            self.env.restore_state_from_file("tmp.bullet")
        else:
            self.env.save_state_to_file("tmp.bullet")
        
        self.observation_dim = (None, )
        self.action_dim = self.env.get_robot_dof_count(0)
        self.seed = None
        self.real_step = False
        self.sim_joint_positions = np.zeros(self.action_dim)
    
    def step(self, action, torques=None):
        r = 0
        
        if torques is not None:
            self.env.get_joint_positions(0, self.sim_joint_positions)
            _, over_limits = apply_action_clipping_sim(self.sim_joint_positions, return_over_limit=True)
            torques = torques[:len(over_limits)]
            torques[over_limits] = 1
            
            # torques = np.zeros_like(torques)
            self.env.update_torques(0, torques)
        
        for k in range(self.task.interval):
            self.env.set_joint_targets(0, action.reshape(-1, 1))
            self.task.add_noise(self.env, (self.task.interval * self.seed + k) % (2 ** 32))
            self.env.step()
            
            r += self.task.get_objective_fn()(self.env) # , neural_input)
        
        if self.real_step:
            self.env.save_state_to_file("tmp.bullet")
        
        return None, r, None, None
    
    def reset(self, seed=None):
        if self.seed is None and seed is not None:
            self.seed = seed
    
    def set_seed(self, seed=None):
        self.seed = seed
    
    def set_env_state(self):
        self.env.restore_state()
    
    def real_env_step(self, is_real):
        self.real_step = is_real
        if not is_real:
            self.env.save_state()