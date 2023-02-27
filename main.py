import traceback
from time import time, sleep
from datetime import datetime
import os

if not os.access('/dev/ttyUSB0', os.R_OK):
    print("Fixing permissions. Input admin password below.")
    os.system("./fix_permissions.sh")
    os.system("setserial /dev/ttyUSB0 low_latency")

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
from controller import Controller


if __name__ == "__main__":
    
    experiment_config = get_experiment_config()
    
    # initialize task
    task = FlatTerrainTask()
    graphs = rd.load_graphs(experiment_config["grammar_filepath"])
    rules = [rd.create_rule_from_graph(g) for g in graphs]
    rule_sequence = [int(s.strip(",")) for s in experiment_config["rule_sequence"]]
    
    # initialize graph
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
    
    # initialize controller
    controller = None # Controller()
    # initialize neuron stream
    if experiment_config["neuron_stream"]:
        neuron_stream = NeuronStream(channels=experiment_config["channels"], 
                                     dt=experiment_config["dt"],
                                     buffer_size=int(experiment_config["neuron_stream_seconds"] / experiment_config["dt"]))
        neuron_stream.start()
    else:
        neuron_stream = None
        
    # initialize rendering
    viewer, tracker = prepare_viewer(main_env)
    
    
    if experiment_config["optimize"]:
        optimizer = MPPI(env, experiment_config["horizon"], n_samples, 
                            num_cpu=experiment_config["num_threads"],
                            kappa=1.0,
                            gamma=task.discount_factor,
                            default_act="mean",
                            filter_coefs=experiment_config["action_filter_coefs"],
                            seed=experiment_config["seed"],
                            neuron_stream=neuron_stream)
        
        # search for initial paths
        paths = optimizer.do_rollouts(experiment_config["seed"])
        optimizer.update(paths)
        optimizer.paths_per_cpu = 64 // experiment_config["num_threads"]
    else:
        optimizer = None
    
    if experiment_config["input_action_sequence"] is not None:
        action_sequence = np.load(experiment_config["input_action_sequence"])
        experiment_config["save_action_sequence"] = False
    
    if experiment_config["save_action_sequence"]:
        action_sequence = []
    
    current_torques = _current_torques = np.zeros(dof_count)
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
            elif experiment_config["input_action_sequence"] is not None:
                actions = action_sequence[step % len(action_sequence)]
            else:
                try:
                    with open("actions.csv", "r") as f:
                        actions = np.array(list(map(float, f.read().split(","))))
                    assert len(actions) == dof_count
                except:
                    actions = np.zeros(dof_count)
            
            if experiment_config["save_action_sequence"]:
                action_sequence.append(actions)
                
            if viewer is not None:
                actions_t = apply_action_clipping_sim(actions)
                viewer_step(main_env, task, actions_t, viewer, tracker, step=step, torques=_current_torques)
                
                main_env.get_joint_positions(0, sim_joint_positions)
                _, over_limits = apply_action_clipping_sim(sim_joint_positions, return_over_limit=True)
                
                _current_torques = current_torques.copy() if optimizer is None else optimizer.current_torques.copy()[:len(over_limits)]
                _current_torques[over_limits] = 1
            
            if controller is not None:
                actions_t = sim_joint_positions if viewer is not None else actions
                actions_t = convert_joint_angles(actions_t)
                controller.move(actions_t)
            
            curr_time = time()
            sleep_time = curr_time - prev_time
            print("step =", step, "\ttime =", np.round(sleep_time, 4), "\tactions =", np.round(actions, 2), "\tsim_positions =", np.round(sim_joint_positions, 2))
            sleep((1 / 15 - sleep_time + 0.01) if sleep_time < 1 / 15 else 0.01)
            prev_time = curr_time
            step += 1
            
            # if step % 50 == 0:
            #     current_torques = 1 - current_torques
    
    except KeyboardInterrupt:
        pass
    
    except:
        traceback.print_exc()
        
    finally:
        # stop the threaded processes.
        print("STOPPING THE PROCESSES.")
        if experiment_config["save_action_sequence"]:
            output_path = os.path.join("output", str(datetime.now()).split(".")[0].replace(" ", "_").replace(":", "-") + ".npy")
            np.save(output_path, action_sequence)
        
        if controller is not None:
            controller.stop()
        if neuron_stream is not None:
            neuron_stream.stop()