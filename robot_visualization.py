import traceback
from time import time, sleep

import pyrobotdesign as rd
import numpy as np

from env import SimEnvWrapper
from mppi import MPPI
from neurons import NeuronStream
from tasks import FlatTerrainTask
from utils import (build_normalized_robot, finalize_robot, convert_joint_angles,
                   get_make_sim_and_task_fn, make_graph, presimulate, apply_action_clipping_sim, get_experiment_config)
# from constants import *
from view import prepare_viewer, viewer_step


if __name__ == "__main__":
    
    experiment_config = get_experiment_config(path="configs/experiment_onesided.json")
    
    # initialize task
    task = FlatTerrainTask()
    graphs = rd.load_graphs(experiment_config["grammar_filepath"])
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in experiment_config["rule_sequence"]]
        
    graph = make_graph(rules, rule_sequence)
    # initialize robot
    robot = build_normalized_robot(graph)
    finalize_robot(robot)
    robot_init_pos, _ = presimulate(robot)
    
    # initialize env
    make_sim_and_task_fn = get_make_sim_and_task_fn(task, robot, robot_init_pos=robot_init_pos)
    main_env, _ = make_sim_and_task_fn()
    env = SimEnvWrapper(make_sim_and_task_fn)
    
    dof_count = main_env.get_robot_dof_count(0)
    objective_fn = task.get_objective_fn()
    n_samples = 512 // experiment_config["num_threads"]
    
    # initialize rendering
    viewer, tracker = prepare_viewer(main_env)
    
    # this gives me the joint types that should be fixed during movement.
    # TODO: an algorithm that determines the positions of joints given their number and spacing
    # NOTE: 2s in the rule sequence seems to indicative of the number of legs. make sure there are always three.
    sim_joint_types = np.zeros(dof_count)
    main_env.get_joint_types(0, sim_joint_types)
    print(sim_joint_types)
    
    if experiment_config["optimize"]:
        optimizer = MPPI(env, experiment_config["horizon"], n_samples, 
                            num_cpu=experiment_config["num_threads"],
                            kappa=1.0,
                            gamma=task.discount_factor,
                            default_act="mean",
                            filter_coefs=experiment_config["action_filter_coefs"],
                            seed=experiment_config["seed"],
                            joint_types=sim_joint_types,
                            neuron_stream=None)
        
        # search for initial paths
        paths = optimizer.do_rollouts(experiment_config["seed"])
        optimizer.update(paths)
        optimizer.paths_per_cpu = 64 // experiment_config["num_threads"]
    else:
        optimizer = None
    
    current_torques = _current_torques = np.ones(dof_count)
    sim_joint_positions = np.zeros(dof_count)
    
    try:
        prev_time = time()
        step = 0
        while True:
            
            if optimizer is not None:
                paths = optimizer.do_rollouts(experiment_config["seed"] + len(optimizer.sol_act) + 1)
                optimizer.update(paths)
                
                actions = optimizer.act_sequence[0]
                
                optimizer.advance_time()
            else:
                try:
                    with open("actions.csv", "r") as f:
                        actions = np.array(list(map(float, f.read().split(","))))
                    assert len(actions) == dof_count
                except:
                    actions = np.zeros(dof_count)
                
            if viewer is not None:
                # actions_t = apply_action_clipping_sim(actions)
                viewer_step(main_env, task, actions, viewer, tracker, step=step, torques=_current_torques)
                
                main_env.get_joint_positions(0, sim_joint_positions)
                # _, over_limits = apply_action_clipping_sim(sim_joint_positions, return_over_limit=True)
                
                # _current_torques = current_torques.copy() if optimizer is None else optimizer.current_torques.copy()[:len(over_limits)]
                # _current_torques[over_limits] = 1
                
            curr_time = time()
            sleep_time = curr_time - prev_time
            print("step =", step, 
                  "\ttime =", np.round(sleep_time, 4), 
                  "\tactions =", np.round(actions, 2), 
                  "\tsim_positions =", np.round(sim_joint_positions, 2),
                  "\ttorques =", np.round(_current_torques))
            sleep((1 / 15 - sleep_time + 0.01) if sleep_time < 1 / 15 else 0.01)
            prev_time = curr_time
            step += 1
    
    except KeyboardInterrupt:
        pass
    
    except:
        traceback.print_exc()
        
    finally:
        # stop the threaded processes.
        print("STOPPING THE PROCESSES.")
        sleep(3)
