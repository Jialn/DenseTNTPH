
import uuid
import numpy as np
import carla

lane_sample_resolution = 1.0
max_lane_sample_distance_when_straight = 5.0
pixels_per_meter = 2
raster_size = [ 224, 224 ]


def lateral_shift(transform,shift):
    transform.rotation.yaw += 90
    return transform.location + shift * transform.get_forward_vector()


def x_to_world_transformation(transform):
    """
    Get the transformation matrix from x(it can be vehicle or sensor)
    coordinates to world coordinate.

    Parameters
    ----------
    transform : carla.Transform
        The transform that contains location and rotation

    Returns
    -------
    matrix : np.ndarray
        The transformation matrx.

    """
    rotation = transform.rotation
    location = transform.location

    # used for rotation matrix
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))

    matrix = np.identity(4)
    # translation matrix
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z

    # rotation matrix
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r

    return matrix


def list_loc2array(list_location):
        """
        Convert list of carla location to np.array
        Parameters
        ----------
        list_location : list
            List of carla locations.

        Returns
        -------
        loc_array : np.array
            Numpy array of shape (N, 3)
        """
        loc_array = np.zeros((len(list_location), 3))
        for (i, carla_location) in enumerate(list_location):
            loc_array[i, 0] = carla_location.x
            loc_array[i, 1] = carla_location.y
            loc_array[i, 2] = carla_location.z
        return loc_array


def list_wpt2array(list_wpt):
    """
    Convert list of carla transform to np.array
    Parameters
    ----------
    list_wpt : list
        List of carla waypoint.

    Returns
    -------
    loc_array : np.array
        Numpy array of shape (N, 3)
    """
    loc_array = np.zeros((len(list_wpt), 3))
    for (i, carla_wpt) in enumerate(list_wpt):
        loc_array[i, 0] = carla_wpt.transform.location.x
        loc_array[i, 1] = carla_wpt.transform.location.y
        loc_array[i, 2] = carla_wpt.transform.location.z
    return loc_array

def distance_wp(wp1, wp2):
    pos1 = np.array([wp1.transform.location.x, wp1.transform.location.y])
    pos2 = np.array([wp2.transform.location.x, wp2.transform.location.y])
    return np.linalg.norm(pos1-pos2)

def get_all_lane_info(carla_map):
    """
    From the topology generate all lane and crosswalk
    information in a dictionary under world's coordinate frame.
    """
    # list of str
    lanes_id = []
    # boundary of each lane for later filtering
    lanes_bounds = np.empty((0, 2, 2), dtype=np.float)
    bound_info = {'lanes': {}}
    lane_info = {}

    topology = [x[0] for x in carla_map.get_topology()]
    # sort by altitude
    topology = sorted(topology, key=lambda w: w.transform.location.z)

    def should_remove_wp(wp0, wp1, wp2):
        if abs(wp0.transform.rotation.yaw - wp1.transform.rotation.yaw) < 1.0 and \
            abs(wp1.transform.rotation.yaw - wp2.transform.rotation.yaw) < 1.0 and \
            distance_wp(wp0, wp2) <  max_lane_sample_distance_when_straight:
            return True
        return False

    # loop all waypoints to get lane information
    for (i, waypoint) in enumerate(topology):
        # unique id for each lane
        lane_id = uuid.uuid4().hex[:6].upper()
        lanes_id.append(lane_id)

        is_intersection = waypoint.is_intersection
        lane_change = waypoint.lane_change
        has_traffic_control = True if waypoint.lane_type == carla.LaneType.Driving else False

        waypoints = [waypoint]
        nxt = waypoint.next(lane_sample_resolution)[0]
        # looping until next lane
        while nxt.road_id == waypoint.road_id \
                and nxt.lane_id == waypoint.lane_id:
            if len(waypoints) > 3:
                if should_remove_wp(waypoints[-2], waypoints[-1], nxt):
                    waypoints.pop()
            waypoints.append(nxt)
            nxt = nxt.next(lane_sample_resolution)[0]

        # waypoint is the centerline, we need to calculate left lane mark
        left_marking = [lateral_shift(w.transform, -w.lane_width * 0.5) for
                        w in waypoints]
        right_marking = [lateral_shift(w.transform, w.lane_width * 0.5) for
                            w in waypoints]
        # convert the list of carla.Location to np.array
        left_marking = list_loc2array(left_marking)
        right_marking = list_loc2array(right_marking)
        mid_lane = list_wpt2array(waypoints)

        # get boundary information
        bound = get_bounds(left_marking, right_marking)
        lanes_bounds = np.append(lanes_bounds, bound, axis=0)
        lane_info.update({lane_id: {'xyz_left': left_marking,
                                    'xyz_right': right_marking,
                                     'xyz_mid': mid_lane,
                                     'is_intersection': is_intersection,
                                     'lane_change': lane_change,
                                     'has_traffic_control': has_traffic_control
                                             }})
        # boundary information
        bound_info['lanes']['ids'] = lanes_id
        bound_info['lanes']['bounds'] = lanes_bounds

    return bound_info, lane_info


def get_lane_ids_in_xy_bbox(x,y,bound_info,query_search_range_manhattan):
    """
    Prune away all lane segments based on Manhattan distance.
    Perform an search based on manhattan distance search radius from a given 2D query point.
    We pre-assign lane segment IDs to indices inside a big lookup array, with precomputed
    hallucinated lane polygon extents.

    Args:
        query_x: representing x coordinate of xy query location
        query_y: representing y coordinate of xy query location
        city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh
        query_search_range_manhattan: search radius along axes

    Returns:
        lane_ids: lane segment IDs that live within a bubble
    """

    #------------------ 
    lane_ids_in_bbox = []
    lane_indices = indices_in_bounds(x,y,bound_info['lanes']['bounds'], query_search_range_manhattan)
    for idx, lane_idx in enumerate(lane_indices):
        lane_idx = bound_info['lanes']['ids'][lane_idx]
        lane_ids_in_bbox.append(lane_idx)

    return lane_ids_in_bbox


def indices_in_bounds(x,y,bounds: np.ndarray,query_search_range_manhattan: float = 5.0) -> np.ndarray:
    """
    Get indices of elements for which the bounding box described by bounds
    intersects the one defined around center (square with side 2*half_side)

    Parameters
    ----------
    bounds :np.ndarray
        array of shape Nx2x2 [[x_min,y_min],[x_max, y_max]]

    query_search_range_manhattan : float
        half the side of the bounding box centered around center

    Returns
    -------
    np.ndarray: indices of elements inside radius from center
    """
    x_center = x
    y_center = y
    x_min_in = x_center > bounds[:, 0, 0] - query_search_range_manhattan
    y_min_in = y_center > bounds[:, 0, 1] - query_search_range_manhattan
    x_max_in = x_center < bounds[:, 1, 0] + query_search_range_manhattan
    y_max_in = y_center < bounds[:, 1, 1] + query_search_range_manhattan
    return np.nonzero(x_min_in & y_min_in & x_max_in & y_max_in)[0]


def get_lane_segment_centerline(lane_id, lane_info):
    """
    We return a 3D centerline for any particular lane segment.

    Args:
        lane_segment_id: unique identifier for a lane segment within a city
        city_name: either 'MIA' or 'PIT' for Miami or Pittsburgh

    Returns:
        lane_centerline: Numpy array of shape (N,3)
    """
    lane_centerline = lane_info[lane_id]['xyz_mid']
    return lane_centerline


def get_lane_segment_polygon(lane_id, lane_info_):
    """
    Generate the lane area poly under rasterization map's center
    coordinate frame.

    Parameters
    ----------
    xyz_left : np.ndarray
        Left lanemarking of a lane, shape: (n, 3).
    xyz_right : np.ndarray
        Right lanemarking of a lane, shape: (n, 3).

    Returns
    -------
    lane_area : np.ndarray
        Combine left and right lane together to form a polygon.
    """
    lane_info = lane_info_[lane_id]
    # left and right lane location
    xyz_left, xyz_right = lane_info['xyz_left'], lane_info['xyz_right']

    lane_area = np.vstack([xyz_right, xyz_left[::-1]])
    lane_area = np.vstack([lane_area, xyz_right[0]])

    return lane_area


def find_local_lane_centerlines(position,bound_info,lane_info,query_search_range_manhattan) -> np.ndarray:
    """
    Find local lane centerline to the specified x,y location

    Args:
        query_x: x-coordinate of map query
        query_y: x-coordinate of map query
        city_name: either 'MIA' for Miami or 'PIT' for Pittsburgh

    Returns
        local_lane_centerlines: Array of arrays, representing an array of lane centerlines, each a polyline
    """
    lane_ids = get_lane_ids_in_xy_bbox(position,bound_info,query_search_range_manhattan)
    local_lane_centerlines = [get_lane_segment_centerline(lane_id,lane_info) for lane_id in lane_ids]
    return np.array(local_lane_centerlines)


def get_bounds(left_lane, right_lane):
    """
    Get boundary information of a lane.

    Parameters
    ----------
    left_lane : np.array
        shape: (n, 3)
    right_lane : np.array
        shape: (n,3)
    Returns
    -------
    bound : np.array
    """
    x_min = min(np.min(left_lane[:, 0]),
                np.min(right_lane[:, 0]))
    y_min = min(np.min(left_lane[:, 1]),
                np.min(right_lane[:, 1]))
    x_max = max(np.max(left_lane[:, 0]),
                np.max(right_lane[:, 0]))
    y_max = max(np.max(left_lane[:, 1]),
                np.max(right_lane[:, 1]))
    bounds = np.asarray([[[x_min, y_min], [x_max, y_max]]])
    return bounds


def world_to_sensor(cords, sensor_transform):
    """
    Transform coordinates from world reference to sensor reference.

    Parameters
    ----------
    cords : np.ndarray
        Coordinates under world reference, shape: (4, n).

    sensor_transform : carla.Transform
        Sensor position in the world.

    Returns
    -------
    sensor_cords : np.ndarray
        Coordinates in the sensor reference.

    """
    sensor_world_matrix = x_to_world_transformation(sensor_transform)
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)

    return sensor_cords


def city_lane_centerlines_dict(lane_id, lane_info):
    lane_centerlines_dict = {}
    lane_centerlines_dict['has_traffic_control'] = lane_info[lane_id]['has_traffic_control']  # True or Fasle

    if lane_info[lane_id]['lane_change'] == carla.LaneChange.Both:
        lane_centerlines_dict['turn_direction'] = 'BOTH'
    elif lane_info[lane_id]['lane_change'] == carla.LaneChange.Left:
        lane_centerlines_dict['turn_direction'] = 'LEFT'
    elif lane_info[lane_id]['lane_change'] == carla.LaneChange.Right:
        lane_centerlines_dict['turn_direction'] = 'RIGHT'
    else:
        lane_centerlines_dict['turn_direction'] = 'NONE'

    lane_centerlines_dict['is_intersection'] = lane_info[lane_id]['is_intersection'] #  True or Fasle

    return lane_centerlines_dict
