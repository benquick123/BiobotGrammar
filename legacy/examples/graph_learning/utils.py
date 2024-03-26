import sys

import numpy as np

# if there's overlap between args_list and commandline input, use commandline input
def solve_argv_conflict(args_list):
    arguments_to_be_removed = []
    arguments_size = []

    for argv in sys.argv[1:]:
        if argv.startswith('-'):
            size_count = 1
            for i, args in enumerate(args_list):
                if args == argv:
                    arguments_to_be_removed.append(args)
                    for more_args in args_list[i+1:]:
                        if not more_args.startswith('-'):
                            size_count += 1
                        else:
                            break
                    arguments_size.append(size_count)
                    break

    for args, size in zip(arguments_to_be_removed, arguments_size):
        args_index = args_list.index(args)
        for _ in range(size):
            args_list.pop(args_index)
            

class JointTypes:
    
    OTHER = 0.0
    BODY = 1.0
    BODY_FIXED = 2.0
    
    
def get_angles_from_points(x):
    angles = []
    for p0, p1, p2 in zip(x, x[(np.arange(1, len(x) + 1) % len(x))], x[(np.arange(2, len(x) + 2) % len(x))]):
        p01, p12 = p0 - p1, p1 - p2
        p01_len = np.linalg.norm(p0 - p1)
        p12_len = np.linalg.norm(p1 - p2)
        angles.append(np.arccos(np.dot(p01, p12) / (p01_len * p12_len)))
        
    return np.array(angles)
    
    
def angle_finder_fn(x, ls):
    x = x.reshape(-1, 2)
    centroid = x.mean(axis=0)
    dist_err = 0
    for p0, p1, dist in zip(x, x[(np.arange(1, len(x) + 1) % len(x))], ls):
        dist_err += np.linalg.norm(np.linalg.norm(p0 - p1) - dist)
    
    angles = get_angles_from_points(x)
    angle_sum = np.sum(angles)
    return dist_err + np.linalg.norm(centroid) + np.linalg.norm(angle_sum - (len(ls) - 2) * np.pi) + np.std(angles)