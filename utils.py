import json

import numpy as np
import pyrobotdesign as rd

from tasks import FlatTerrainTask
from constants import *


JOINT_BASELINES_ANGLES_292 = np.pi * np.array([0, 0, 0, 0, 60, 120, 0, 0, -120, 60, 0]) / 180
MOTOR_CONFIG = json.load(open("configs/292_motors.json", "r"))
MIN_LIMITS = np.array([MOTOR_CONFIG["min_limits"][str(limit)] for limit in sorted(map(int, MOTOR_CONFIG["min_limits"].keys()))] + [0])
MAX_LIMITS = np.array([MOTOR_CONFIG["max_limits"][str(limit)] for limit in sorted(map(int, MOTOR_CONFIG["max_limits"].keys()))] + [2 ** 12])


def convert_joint_angles(action, joint_baseline_angles=None):
    if joint_baseline_angles is None:
        joint_baseline_angles = JOINT_BASELINES_ANGLES_292
    assert len(action) == len(joint_baseline_angles)
    return action + joint_baseline_angles


def apply_action_clipping_sim(action, return_over_limit=False):
    joint_baseline_angles = JOINT_BASELINES_ANGLES_292
    # receives action in radians
    action = np.array(action)
    # do the transform into real-world radians
    action += joint_baseline_angles
    
    # do the transform into real-world steps
    action += np.pi
    action *= 2 ** 12
    action /= 2 * np.pi

    # clip the actions
    action_clipped = np.clip(action, MIN_LIMITS, MAX_LIMITS)
    over_limits = action_clipped != action
    
    # convert back to radians
    action_clipped *= 2 * np.pi
    action_clipped /= 4096
    action_clipped -= np.pi
    
    # subtract baseline
    action_clipped -= joint_baseline_angles
    if return_over_limit:
        return action_clipped, over_limits
    else:
        return action_clipped


def stack_tensor_dict_list(tensor_dict_list):
    """
    Stack a list of dictionaries of {tensors or dictionary of tensors}.
    :param tensor_dict_list: a list of dictionaries of {tensors or dictionary of tensors}.
    :return: a dictionary of {stacked tensors or dictionary of stacked tensors}
    """
    keys = list(tensor_dict_list[0].keys())
    ret = dict()
    for k in keys:
        example = tensor_dict_list[0][k]
        if isinstance(example, dict):
            v = stack_tensor_dict_list([x[k] for x in tensor_dict_list])
        else:
            v = np.array([x[k] for x in tensor_dict_list])
        ret[k] = v
    return ret


def get_make_sim_and_task_fn_without_args():
    
    def fn():
        task = FlatTerrainTask()
        
        graphs = rd.load_graphs(GRAMMAR_FILEPATH)
        rules = [rd.create_rule_from_graph(g) for g in graphs]
        rule_sequence = [int(s.strip(",")) for s in RULE_SEQUENCE]
        graph = make_graph(rules, rule_sequence)
        robot = build_normalized_robot(graph)
        finalize_robot(robot)
        
        robot_init_pos, _ = presimulate(robot)
        
        sim = rd.BulletSimulation(task.time_step)
        
        task.add_terrain(sim)
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        return sim, task
    
    return fn


def get_make_sim_and_task_fn(task, robot, robot_init_pos):
    
    def fn():
        sim = rd.BulletSimulation(task.time_step)
        task.add_terrain(sim)
        # Rotate 180 degrees around the y axis, so the base points to the right
        sim.add_robot(robot, robot_init_pos, rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
        return sim, task
    
    return fn


def get_applicable_matches(rule, graph):
    """Generates all applicable matches for rule in graph."""
    for match in rd.find_matches(rule.lhs, graph):
        if rd.check_rule_applicability(rule, graph, match):
            yield match


def make_initial_graph():
    """Make an initial robot graph."""
    n0 = rd.Node()
    n0.name = 'robot'
    n0.attrs.label = 'robot'
    initial_graph = rd.Graph()
    initial_graph.nodes = [n0]
    return initial_graph


def make_graph(rules, rule_sequence):
    graph = make_initial_graph()
    for r in rule_sequence:
        matches = list(get_applicable_matches(rules[r], graph))
        if matches:
            graph = rd.apply_rule(rules[r], graph, matches[0])
        else:
            raise ValueError("Rule in sequence has no applicable matches")
    return graph


def build_normalized_robot(graph):
    """Build a robot from the graph and normalize the mass of the body links."""
    robot = rd.build_robot(graph)

    body_links = []
    total_body_length = 0.0
    for link in robot.links:
        if np.isclose(link.radius, 0.045):
            # Link is a body link
            body_links.append(link)
            total_body_length += link.length
            target_mass = link.length * link.density

    if body_links:
        body_density = target_mass / total_body_length
        for link in body_links:
            link.density = body_density

    return robot


def finalize_robot(robot):
    for link in robot.links:
        link.label = ""
        link.joint_label = ""
        if link.shape == rd.LinkShape.NONE:
            link.shape = rd.LinkShape.CAPSULE
            link.length = 0.1
            link.radius = 0.025
            link.color = [1.0, 0.0, 1.0]
        if link.joint_type == rd.JointType.NONE:
            link.joint_type = rd.JointType.FIXED
            link.joint_color = [1.0, 0.0, 1.0]


def presimulate(robot):
    """Find an initial position that will place the robot on the ground behind the
    x=0 plane, and check if the robot collides in its initial configuration."""
    temp_sim = rd.BulletSimulation()
    temp_sim.add_robot(robot, np.zeros(3), rd.Quaterniond(0.0, 0.0, 1.0, 0.0))
    temp_sim.step()
    robot_idx = temp_sim.find_robot_index(robot)
    lower = np.zeros(3)
    upper = np.zeros(3)
    temp_sim.get_robot_world_aabb(robot_idx, lower, upper)
    return [-upper[0], -lower[1], 0.0], temp_sim.robot_has_collision(robot_idx)

