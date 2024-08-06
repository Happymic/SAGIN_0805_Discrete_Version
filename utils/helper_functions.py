import numpy as np

def euclidean_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def angle_between(pos1, pos2):
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])

def is_in_range(pos1, pos2, range):
    return euclidean_distance(pos1, pos2) <= range

def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm