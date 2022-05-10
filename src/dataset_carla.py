"""
CarlaEnvironment.
Install carla:
.. code-block:: bash
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.9.tar.gz
    mkdir carla
    tar zxf CARLA_0.9.9.tar.gz -C carla
    cd carla/Import
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.9.tar.gz
    cd ..
    ./ImportAssert.sh
    easy_install PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
    pip install networkx==2.2
Make sure you are using python3.7
"""
import uuid
import carla
import numpy as np
import os
import sys
import copy
import math
import multiprocessing
import os
import pickle
import random
import zlib
from collections import defaultdict
from multiprocessing import Process
from random import choice

import numpy as np
import torch

# from argoverse.map_representation.map_api import ArgoverseMap
# import carla

from tqdm import tqdm

import utils_cython
import utils
from utils import get_name, get_file_name_int, get_angle, logging, rotate, round_value, get_pad_vector, get_dis, get_subdivide_polygons
from utils import get_points_remove_repeated, get_one_subdivide_polygon, get_dis_point_2_polygons, larger, equal, assert_
from utils import get_neighbour_points, get_subdivide_points, get_unit_vector, get_dis_point_2_points

TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3

pixels_per_meter = 2
raster_size = [ 224, 224 ]
lane_sample_resolution = 0.1


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

    # loop all waypoints to get lane information
    for (i, waypoint) in enumerate(topology):
        # unique id for each lane
        lane_id = uuid.uuid4().hex[:6].upper()
        lanes_id.append(lane_id)

        waypoints = [waypoint]
        nxt = waypoint.next(lane_sample_resolution)[0]
        # looping until next lane
        while nxt.road_id == waypoint.road_id \
                and nxt.lane_id == waypoint.lane_id:
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
                                        }})
        # boundary information
        bound_info['lanes']['ids'] = lanes_id
        bound_info['lanes']['bounds'] = lanes_bounds
    return bound_info,lane_info,lanes_bounds


def get_lane_ids_in_xy_bbox(position,bound_info,query_search_range_manhattan):
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
    lane_ids_in_bbox = []
    lane_indices = indices_in_bounds(position,bound_info['lanes']['bounds'], query_search_range_manhattan)
    for idx, lane_idx in enumerate(lane_indices):
        lane_idx = bound_info['lanes']['ids'][lane_idx]
        lane_ids_in_bbox.append(lane_idx)  
    return lane_ids_in_bbox


def indices_in_bounds(position,bounds: np.ndarray,query_search_range_manhattan: float = 5.0) -> np.ndarray:
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
    x_center, y_center = position.location.x, position.location.y

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


def get_lane_segment_polygon(lane_id, lane_info_,position):
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
    center = position
    lane_info = lane_info_[lane_id]
    # left and right lane location
    xyz_left, xyz_right = lane_info['xyz_left'], lane_info['xyz_right']
    lane_area = np.zeros((2, xyz_left.shape[0], 2))
    # convert coordinates to center's coordinate frame
    xyz_left = xyz_left.T
    xyz_left = np.r_[xyz_left, [np.ones(xyz_left.shape[1])]]
    xyz_right = xyz_right.T
    xyz_right = np.r_[xyz_right, [np.ones(xyz_right.shape[1])]]

    # ego's coordinate frame
    xyz_left = world_to_sensor(xyz_left, center).T
    xyz_right = world_to_sensor(xyz_right, center).T

    # to image coordinate frame
    lane_area[0] = xyz_left[:, :2]
    lane_area[1] = xyz_right[::-1, :2]
    # switch x and y
    lane_area = lane_area[..., ::-1]
    # y revert
    lane_area[:, :, 1] = -lane_area[:, :, 1]
    lane_area[:, :, 0] = lane_area[:, :, 0] * pixels_per_meter + raster_size[0] // 2
    lane_area[:, :, 1] = lane_area[:, :, 1] * pixels_per_meter + raster_size[1] // 2
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

@staticmethod
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


def city_lane_centerlines_dict(city_name, lane_id):
    lane_centerlines_dict = {}
    lane_centerlines_dict['has_traffic_control'] = None  # True or Fasle
    lane_centerlines_dict['turn_direction'] = None  # 'RIGHT' or 'LEFT'
    lane_centerlines_dict['is_intersection'] = None #  True or Fasle
    return lane_centerlines_dict


def get_sub_map(args: utils.Args, x, y, city_name, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.
    Only take lanes which are no more than args.max_distance away from (x, y).
    """
    if args.not_use_api:
        pass
    else:
        if 'semantic_lane' in args.other_params:
            lane_ids = get_lane_ids_in_xy_bbox(x, y, city_name, query_search_range_manhattan=args.max_distance)
            # lane_centerline = am.city_lane_centerlines_dict[city_name][lane_ids[0]].centerline
            # print(lane_centerline.shape, am.get_ground_height_at_xy(lane_centerline, city_name))
            local_lane_centerlines = [get_lane_segment_centerline(lane_id, city_name) for lane_id in lane_ids]
            polygons = local_lane_centerlines
            # z = am.get_ground_height_at_xy(np.array([[x, y]]), city_name)[0]

            if args.visualize:
                angle = mapping['angle']
                vis_lanes = [get_lane_segment_polygon(lane_id, city_name) for lane_id in lane_ids]
                t = []
                for each in vis_lanes:
                    for point in each:
                        point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                    num = len(each) // 2
                    t.append(each[:num].copy())
                    t.append(each[num:num * 2].copy())
                vis_lanes = t
                mapping['vis_lanes'] = vis_lanes
        # else:
        #     polygons = am.find_local_lane_centerlines(x, y, city_name,
        #                                               query_search_range_manhattan=args.max_distance)
        polygons = [polygon[:, :2].copy() for polygon in polygons]
        angle = mapping['angle']
        for index_polygon, polygon in enumerate(polygons):
            for i, point in enumerate(polygon):
                point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)
                if 'scale' in mapping:
                    assert 'enhance_rep_4' in args.other_params
                    scale = mapping['scale']
                    point[0] *= scale
                    point[1] *= scale

        if args.use_centerline:
            if 'semantic_lane' in args.other_params:
                local_lane_centerlines = [polygon for polygon in polygons]

        def dis_2(point):
            return point[0] * point[0] + point[1] * point[1]

        def get_dis(point_a, point_b):
            return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

        def get_dis_for_points(point, polygon):
            dis = np.min(np.square(polygon[:, 0] - point[0]) + np.square(polygon[:, 1] - point[1]))
            return np.sqrt(dis)

        def ok_dis_between_points(points, points_, limit):
            dis = np.inf
            for point in points:
                dis = np.fmin(dis, get_dis_for_points(point, points_))
                if dis < limit:
                    return True
            return False

        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

        lane_idx_2_polygon_idx = {}
        for polygon_idx, lane_idx in enumerate(lane_ids):
            lane_idx_2_polygon_idx[lane_idx] = polygon_idx

        if 'goals_2D' in args.other_params:
            points = []
            visit = {}
            point_idx_2_unit_vector = []

            mapping['polygons'] = polygons

            for index_polygon, polygon in enumerate(polygons):
                for i, point in enumerate(polygon):
                    hash = get_hash(point)
                    if hash not in visit:
                        visit[hash] = True
                        points.append(point)

                if 'subdivide' in args.other_params:
                    subdivide_points = get_subdivide_points(polygon)
                    points.extend(subdivide_points)
                    subdivide_points = get_subdivide_points(polygon, include_self=True)

            mapping['goals_2D'] = np.array(points)

        for index_polygon, polygon in enumerate(polygons):
            assert_(2 <= len(polygon) <= 10, info=len(polygon))
            # assert len(polygon) % 2 == 1

            # if args.visualize:
            #     traj = np.zeros((len(polygon), 2))
            #     for i, point in enumerate(polygon):
            #         traj[i, 0], traj[i, 1] = point[0], point[1]
            #     mapping['trajs'].append(traj)

            start = len(vectors)
            if 'semantic_lane' in args.other_params:
                assert len(lane_ids) == len(polygons)
                lane_id = lane_ids[index_polygon]
                lane_segment = city_lane_centerlines_dict(city_name, lane_id)
            assert_(len(polygon) >= 2)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * args.hidden_size
                    vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                    vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                    vector[-5] = 1
                    vector[-6] = i

                    vector[-7] = len(polyline_spans)

                    if 'semantic_lane' in args.other_params:
                        vector[-8] = 1 if lane_segment.has_traffic_control else -1
                        vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                            -1 if lane_segment.turn_direction == 'LEFT' else 0
                        vector[-10] = 1 if lane_segment.is_intersection else -1
                    point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                    if i >= 2:
                        point_pre_pre = polygon[i - 2]
                    vector[-17] = point_pre_pre[0]
                    vector[-18] = point_pre_pre[1]

                    vectors.append(vector)
                point_pre = point

            end = len(vectors)
            if start < end:
                polyline_spans.append([start, end])

    return (vectors, polyline_spans)
    

def preprocess_map(map_dict):
    """
    Preprocess map to calculate potential polylines.
    """

    for city_name in map_dict:
        ways = map_dict[city_name]['way']
        nodes = map_dict[city_name]['node']
        polylines = []
        polylines_dict = {}
        for way in ways:
            polyline = []
            points = way['nd']
            points = [nodes[int(point['@ref'])] for point in points]
            point_pre = None
            for i, point in enumerate(points):
                if i > 0:
                    vector = [float(point_pre['@x']), float(point_pre['@y']), float(point['@x']), float(point['@y'])]
                    polyline.append(vector)
                point_pre = point

            if len(polyline) > 0:
                index_x = round_value(float(point_pre['@x']))
                index_y = round_value(float(point_pre['@y']))
                if index_x not in polylines_dict:
                    polylines_dict[index_x] = []
                polylines_dict[index_x].append(polyline)
                polylines.append(polyline)

        map_dict[city_name]['polylines'] = polylines
        map_dict[city_name]['polylines_dict'] = polylines_dict


def preprocess(args, id2info, mapping):
    """
    This function calculates matrix based on information from get_instance.
    """
    polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    vectors = []
    two_seconds = mapping['two_seconds']
    mapping['trajs'] = []
    mapping['agents'] = []
    for id in keys:
        polyline = {}

        info = id2info[id]
        start = len(vectors)
        if args.no_agents:
            if id != 'AV' and id != 'AGENT':
                break

        agent = []
        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            agent.append((line[X], line[Y]))

        if args.visualize:
            traj = np.zeros([args.hidden_size])
            for i, line in enumerate(info):
                if larger(line[TIMESTAMP], two_seconds):
                    traj = traj[:i * 2].copy()
                    break
                traj[i * 2], traj[i * 2 + 1] = line[X], line[Y]
                if i == len(info) - 1:
                    traj = traj[:(i + 1) * 2].copy()
            traj = traj.reshape((-1, 2))
            mapping['trajs'].append(traj)

        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            x, y = line[X], line[Y]
            if i > 0:
                # print(x-line_pre[X], y-line_pre[Y])
                vector = [line_pre[X], line_pre[Y], x, y, line[TIMESTAMP], line[OBJECT_TYPE] == 'AV',
                          line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS', len(polyline_spans), i]
                vectors.append(get_pad_vector(vector))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != 'AGENT'
        else:
            mapping['agents'].append(np.array(agent))

            polyline_spans.append([start, end])

    assert_(len(mapping['agents']) == len(polyline_spans))

    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping['map_start_polyline_idx'] = len(polyline_spans)
    if args.use_map:
        vectors, polyline_spans = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'],
                                              vectors=vectors,
                                              polyline_spans=polyline_spans, mapping=mapping)

    # logging('len(vectors)', t, len(vectors), prob=0.01)

    matrix = np.array(vectors)
    # matrix = np.array(vectors, dtype=float)
    # del vectors

    # matrix = torch.zeros([len(vectors), args.hidden_size])
    # for i, vector in enumerate(vectors):
    #     for j, each in enumerate(vector):
    #         matrix[i][j].fill_(each)

    labels = []
    info = id2info['AGENT']
    info = info[mapping['agent_pred_index']:]
    if not args.do_test:
        if 'set_predict' in args.other_params:
            pass
        else:
            assert len(info) == 30
    for line in info:
        labels.append(line[X])
        labels.append(line[Y])

    if 'set_predict' in args.other_params:
        if 'test' in args.data_dir[0]:
            labels = [0.0 for _ in range(60)]

    if 'goals_2D' in args.other_params:
        point_label = np.array(labels[-2:])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

        if 'stage_one' in args.other_params:
            stage_one_label = 0
            polygons = mapping['polygons']
            min_dis = 10000.0
            for i, polygon in enumerate(polygons):
                temp = np.min(get_dis(polygon, point_label))
                if temp < min_dis:
                    min_dis = temp
                    stage_one_label = i

            mapping['stage_one_label'] = stage_one_label

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([30, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(args.future_frame_num, dtype=np.int64),
        eval_time=30,
    ))

    return mapping


def argoverse_get_instance(lines, file_name, args):
    """
    Extract polylines from one example file content.
    """

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping['file_name'] = file_name

    for i, line in enumerate(lines):

        line = line.strip().split(',')
        if i == 0:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]
            if 'direction' in args.other_params:
                span = agent_lines[-6:]
                intervals = [2]
                angles = []
                for interval in intervals:
                    for j in range(len(span)):
                        if j + interval < len(span):
                            der_x, der_y = span[j + interval][X] - span[j][X], span[j + interval][Y] - span[j][Y]
                            angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]
    if not args.do_test:
        if 'set_predict' in args.other_params:
            pass
        else:
            assert len(id2info['AGENT']) == 50

    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    if args.do_eval:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info['AGENT'][20:]):
            origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
        mapping['origin_labels'] = origin_labels

    angle = -get_angle(der_x, der_y) + math.radians(90)
    if 'direction' in args.other_params:
        angles = np.array(angles)
        der_x, der_y = np.mean(angles, axis=0)
        angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale
    return preprocess(args, id2info, mapping)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, batch_size, to_screen=True):
        data_dir = args.data_dir
        self.ex_list = []
        self.args = args

        if args.reuse_temp_file:
            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'rb')
            self.ex_list = pickle.load(pickle_file)
            # self.ex_list = self.ex_list[len(self.ex_list) // 2:]
            pickle_file.close()
        else:
            if args.core_num >= 1:
                # TODO
                files = []
                for each_dir in data_dir:
                    root, dirs, cur_files = os.walk(each_dir).__next__()
                    files.extend([os.path.join(each_dir, file) for file in cur_files if
                                  file.endswith("csv") and not file.startswith('.')])
                print(files[:5], files[-5:])

                pbar = tqdm(total=len(files))

                queue = multiprocessing.Queue(args.core_num)
                queue_res = multiprocessing.Queue()

                def calc_ex_list(queue, queue_res, args):
                    res = []
                    dis_list = []
                    while True:
                        file = queue.get()
                        if file is None:
                            break
                        if file.endswith("csv"):
                            with open(file, "r", encoding='utf-8') as fin:
                                lines = fin.readlines()[1:]
                            instance = argoverse_get_instance(lines, file, args)
                            if instance is not None:
                                data_compress = zlib.compress(pickle.dumps(instance))
                                res.append(data_compress)
                                queue_res.put(data_compress)
                            else:
                                queue_res.put(None)

                processes = [Process(target=calc_ex_list, args=(queue, queue_res, args,)) for _ in range(args.core_num)]
                for each in processes:
                    each.start()
                # res = pool.map_async(calc_ex_list, [queue for i in range(args.core_num)])
                for file in files:
                    assert file is not None
                    queue.put(file)
                    pbar.update(1)

                # necessary because queue is out-of-order
                while not queue.empty():
                    pass

                pbar.close()

                self.ex_list = []

                pbar = tqdm(total=len(files))
                for i in range(len(files)):
                    t = queue_res.get()
                    if t is not None:
                        self.ex_list.append(t)
                    pbar.update(1)
                pbar.close()
                pass

                for i in range(args.core_num):
                    queue.put(None)
                for each in processes:
                    each.join()

            else:
                assert False

            pickle_file = open(os.path.join(args.temp_file_dir, get_name('ex_list')), 'wb')
            pickle.dump(self.ex_list, pickle_file)
            pickle_file.close()
        assert len(self.ex_list) > 0
        if to_screen:
            print("valid data size is", len(self.ex_list))
            logging('max_vector_num', max_vector_num)
        self.batch_size = batch_size

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        # file = self.ex_list[idx]
        # pickle_file = open(file, 'rb')
        # instance = pickle.load(pickle_file)
        # pickle_file.close()

        data_compress = self.ex_list[idx]
        instance = pickle.loads(zlib.decompress(data_compress))
        return instance


def post_eval(args, file2pred, file2labels, DEs):
    from argoverse.evaluation import eval_forecasting

    score_file = args.model_recover_path.split('/')[-1]
    for each in args.eval_params:
        each = str(each)
        if len(each) > 15:
            each = 'long'
        score_file += '.' + str(each)
        # if 'minFDE' in args.other_params:
        #     score_file += '.minFDE'
    if args.method_span[0] >= utils.NMS_START:
        score_file += '.NMS'
    else:
        score_file += '.score'

    for method in utils.method2FDEs:
        FDEs = utils.method2FDEs[method]
        miss_rate = np.sum(np.array(FDEs) > 2.0) / len(FDEs)
        if method >= utils.NMS_START:
            method = 'NMS=' + str(utils.NMS_LIST[method - utils.NMS_START])
        utils.logging(
            'method {}, FDE {}, MR {}, other_errors {}'.format(method, np.mean(FDEs), miss_rate, utils.other_errors_to_string()),
            type=score_file, to_screen=True, append_time=True)
    utils.logging('other_errors {}'.format(utils.other_errors_to_string()),
                  type=score_file, to_screen=True, append_time=True)
    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(file2pred, file2labels, 6, 30, 2.0)
    utils.logging(metric_results, type=score_file, to_screen=True, append_time=True)
    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        utils.logging('ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3', score,
                      type=score_file, to_screen=True, append_time=True)

    utils.logging(vars(args), is_json=True,
                  type=score_file, to_screen=True, append_time=True)


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
