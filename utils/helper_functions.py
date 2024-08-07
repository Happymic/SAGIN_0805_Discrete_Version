import numpy as np


def euclidean_distance(pos1, pos2):
    """
    Calculate the Euclidean distance between two points.

    Args:
    pos1 (np.array): First position
    pos2 (np.array): Second position

    Returns:
    float: Euclidean distance between pos1 and pos2
    """
    return np.linalg.norm(np.array(pos1) - np.array(pos2))


def angle_between(pos1, pos2):
    """
    Calculate the angle between two positions relative to the x-axis.

    Args:
    pos1 (np.array): First position
    pos2 (np.array): Second position

    Returns:
    float: Angle in radians
    """
    return np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0])


def is_in_range(pos1, pos2, range):
    """
    Check if two positions are within a certain range of each other.

    Args:
    pos1 (np.array): First position
    pos2 (np.array): Second position
    range (float): Maximum distance

    Returns:
    bool: True if positions are within range, False otherwise
    """
    return euclidean_distance(pos1, pos2) <= range


def normalize_vector(vector):
    """
    Normalize a vector to unit length.

    Args:
    vector (np.array): Input vector

    Returns:
    np.array: Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def clip_value(value, min_val, max_val):
    """
    Clip a value between a minimum and maximum.

    Args:
    value (float): Value to clip
    min_val (float): Minimum allowed value
    max_val (float): Maximum allowed value

    Returns:
    float: Clipped value
    """
    return max(min_val, min(value, max_val))


def linear_interpolation(start, end, t):
    """
    Perform linear interpolation between two values.

    Args:
    start (float): Start value
    end (float): End value
    t (float): Interpolation factor (0 to 1)

    Returns:
    float: Interpolated value
    """
    return start + t * (end - start)


def rotate_vector(vector, angle):
    """
    Rotate a 2D vector by a given angle.

    Args:
    vector (np.array): 2D vector to rotate
    angle (float): Angle in radians

    Returns:
    np.array: Rotated vector
    """
    rotation_matrix = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return np.dot(rotation_matrix, vector)


def calculate_heading(velocity):
    """
    Calculate heading angle from a velocity vector.

    Args:
    velocity (np.array): Velocity vector

    Returns:
    float: Heading angle in radians
    """
    return np.arctan2(velocity[1], velocity[0])


def random_point_in_circle(center, radius):
    """
    Generate a random point within a circle.

    Args:
    center (np.array): Center of the circle
    radius (float): Radius of the circle

    Returns:
    np.array: Random point within the circle
    """
    angle = np.random.uniform(0, 2 * np.pi)
    r = radius * np.sqrt(np.random.uniform(0, 1))
    return center + np.array([r * np.cos(angle), r * np.sin(angle)])