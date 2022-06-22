"""
CarlaEnvironment.
Install carla:
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.13.tar.gz
    mkdir carla && \
    tar zxf CARLA_0.9.13.tar.gz -C carla
    cd carla/Import
    wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.13.tar.gz
    cd ..
    ./ImportAssets.sh
    easy_install PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg  or
    pip install PythonAPI/carla/dist/carla-0.9.13-cp37-cp37m-manylinux_2_27_x86_64.whl 
    pip install networkx==2.2
You may need to rename the file to 'carla-0.9.13-cp37-cp37m-manylinux1_x86_64.whl'
Make sure you are using python3.7

To Run:
./CarlaUE4.sh # -RenderOffScreen
# to use less GPU memory, less GPU but possible crash:
./CarlaUE4.sh  -RenderOffScreen -quality-level=Low
# testing:
python src/carla_scripts/carla_with_traffic.py
"""
import os, sys, time
import carla
import logging
from numpy import random
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from utils import rotate, get_subdivide_points, get_dis
from carla_scripts.carla_submap_wrapper import get_lane_ids_in_xy_bbox, get_lane_segment_centerline, city_lane_centerlines_dict, get_all_lane_info


def get_vectornet_mapping(all_vehicles_pos_list, agent_angle, 
        map_bound_info, map_lane_info, frame_cnt,
        mapping):
    '''
    Get the vectornet mapping from vechicles pos list and map data
    Inputs:
        all_vehicles_pos_list: position list of all the vehicles including agent. agent is the first one of all_vehicles_pos_list
        agent_angle: the yaw angle of agent at the current (2s) timestamp
        map_bound_info, map_lane_info: the map info, dict, refer tocarla_submap_wrapper
        frame_cnt: a number, will be add to the 'filename', should be unique in the dataset
        mapping: a dict, can be empty, used for vectornet input of denseTNT and training
    Output: will complete the mapping keys as output, including:
        'cent_x', 'cent_y', 'angle', 'trajs', 'map_start_polyline_idx', 'polygons', 
        'goals_2D', 'matrix', 'polyline_spans', 'origin_labels',  'goals_2D_labels', 'stage_one_label',  'labels', 'labels_is_valid',
        'eval_time', 'file_name'
    '''
    two_second_index = 20
    min_distance_submap = 35
    max_distance_for_agents = 70
    vector_net_hidden_size = 128
    polyline_spans = []
    vectors = []
    relative_trajs = []
    map_start_polyline_idx = None
    agent_pos_list = all_vehicles_pos_list[0]
    agent_loc = agent_pos_list[two_second_index]
    
    def get_pad_vector(li, hidden_size):
        # Pad vector to hidden_size
        assert len(li) <= hidden_size
        li.extend([0] * (hidden_size - len(li)))
        return li
    
    def get_hash(point):
        return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

    # get vehicles' trajectory
    for vhid in range(len(all_vehicles_pos_list)):
        vh_loc = all_vehicles_pos_list[vhid][two_second_index]
        # print(vh_loc, end=",")
        if abs(vh_loc[0] - agent_loc[0]) < max_distance_for_agents and abs(vh_loc[1] - agent_loc[1]) < max_distance_for_agents:
            start = len(vectors)
            # relative_trajs for denseTNT visualize:
            traj = []
            for pos in all_vehicles_pos_list[vhid]:
                x, y = rotate(pos[0]-agent_loc[0], pos[1]-agent_loc[1], agent_angle)
                traj.append([x,y])
            relative_trajs.append(np.array(traj).reshape((-1, 2)))
            # trajectory for prediction
            is_agent = (vhid == 0)  # self.world.get_actor(self.vehicles_list[vhid]).attributes['role_name'] == 'hero'
            is_others, is_av = not is_agent, False
            for i, line in enumerate(traj):
                x, y = line[0], line[1]
                time_stamp = (i - two_second_index) * 0.1
                if i > 0 and i < two_second_index:
                    line_pre = traj[i-1]
                    vector = [line_pre[0], line_pre[1], x, y, time_stamp, is_av,
                            is_agent, is_others, len(polyline_spans), i]
                    vectors.append(get_pad_vector(vector, vector_net_hidden_size))
            # set polyline_spans
            end = len(vectors)
            polyline_spans.append([start, end])
        # end vehicles' trajectory
    map_start_polyline_idx = len(polyline_spans)
    origin_labels = np.zeros([30, 2])
    for i, line in enumerate(all_vehicles_pos_list[0][20:50]):
        origin_labels[i][0], origin_labels[i][1] = line[0], line[1]

    # get sub-map around agent location
    VECTOR_PRE_X = 0
    VECTOR_PRE_Y = 1
    VECTOR_X = 2
    VECTOR_Y = 3
    lane_ids = get_lane_ids_in_xy_bbox(agent_loc[0], agent_loc[1], map_bound_info, min_distance_submap)
    local_lane_centerlines = [get_lane_segment_centerline(lane_id, map_lane_info) for lane_id in lane_ids]
    polygons = []
    for polygon in local_lane_centerlines:
        if len(polygon) > 2: polygons.append(polygon[:, :2].copy())
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            point[0], point[1] = rotate(point[0] - agent_loc[0], point[1] - agent_loc[1], agent_angle)
    local_lane_centerlines = [polygon for polygon in polygons]

    # goals_2D and labels
    points = []
    visit = {}
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            hash = get_hash(point)
            if hash not in visit:
                visit[hash] = True
                points.append(point)
        subdivide_points = get_subdivide_points(polygon)
        points.extend(subdivide_points)
    mapping['goals_2D'] = np.array(points)
    
    labels = []
    for i, line in enumerate(relative_trajs[0][20:50]):
        labels.append(line[0])
        labels.append(line[1])
    point_label = np.array(labels[-2:])
    mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))
    mapping['goals_2D_labels_xy'] = mapping['goals_2D'][mapping['goals_2D_labels']]

    stage_one_label = 0
    min_dis = 10000.0
    for i, polygon in enumerate(polygons):
        temp = np.min(get_dis(polygon, point_label))
        if temp < min_dis:
            min_dis = temp
            stage_one_label = i
    mapping['stage_one_label'] = stage_one_label
    
    for index_polygon, polygon in enumerate(polygons):
        start = len(vectors)
        # semantic_lane 
        lane_id = lane_ids[index_polygon]
        lane_segment_dict = city_lane_centerlines_dict(lane_id, map_lane_info)
        # assert_(len(polygon) >= 2)
        for i, point in enumerate(polygon):
            if i > 0:
                vector = [0] * vector_net_hidden_size # args.hidden_size
                vector[-1 - VECTOR_PRE_X], vector[-1 - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                vector[-1 - VECTOR_X], vector[-1 - VECTOR_Y] = point[0], point[1]
                vector[-5], vector[-6], vector[-7] = 1, i, len(polyline_spans)
                # semantic_lane 
                vector[-8] = 1 if lane_segment_dict['has_traffic_control'] else -1
                vector[-9] = 1 if lane_segment_dict['turn_direction'] == 'RIGHT' else \
                    -1 if lane_segment_dict['turn_direction'] == 'LEFT' else 0
                vector[-10] = 1 if lane_segment_dict['is_intersection'] else -1
                # pre-point
                point_pre_pre = (2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                if i >= 2: point_pre_pre = polygon[i - 2]
                vector[-17], vector[-18] = point_pre_pre[0], point_pre_pre[1]
                # anppend res
                vectors.append(vector)
            point_pre = point
        end = len(vectors)
        if start < end: polyline_spans.append([start, end])
    
    mapping['vis_lanes'] = polygons
    # update dict
    mapping.update(dict(
        matrix=np.array(vectors),
        labels=np.array(labels).reshape([30, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(30, dtype=np.int64),
        eval_time=30, cent_x=agent_loc[0], cent_y=agent_loc[1],
        map_start_polyline_idx=map_start_polyline_idx, polygons=polygons,
        trajs=relative_trajs, angle=agent_angle, origin_labels=origin_labels,
        file_name='carla_'+str(frame_cnt)
    ))


def draw_vectornet_mapping(mapping, win_name="matrix_vis", wait_key=None):
    import cv2
    matrix, polygon_span, map_start_idx = mapping['matrix'], mapping['polyline_spans'], mapping['map_start_polyline_idx'], 
    w, h = 1600, 1600
    offset = (w//2, h//2)
    pix_meter = 0.125
    image = np.zeros((h, w, 3), np.uint8)

    def pts2pix(pts_x, pts_y):
        new_pts = np.array([- pts_x / pix_meter + offset[0], - pts_y / pix_meter + offset[1]]).astype(np.int)
        return (new_pts[0], new_pts[1])
        
    if 'vis.goals_2D' in mapping:
        goals_2d = mapping['vis.goals_2D']  # goals_2D
        score = mapping['vis.scores']
        max_val, min_val = -min(score), -max(score)
        scale = 255 // (max_val - min_val)
        num_goals, _ = goals_2d.shape
        # print(score)
        for j in range(num_goals):
            red = min(255, max(1, score[j]*scale + max_val*scale))
            cv2.circle(image, pts2pix(goals_2d[j, 0], goals_2d[j,1]), 2, (255-red, 0, red), thickness=-1)
    # draw submap
    for i in range(map_start_idx,len(polygon_span)):
        path_span_slice = polygon_span[i]
        for j in range(path_span_slice.start, path_span_slice.stop):
            way_pts_info = matrix[j]
            color = (80, 80, 80)
            cv2.line(image, pts2pix(way_pts_info[-3], way_pts_info[-4]), pts2pix(way_pts_info[-1], way_pts_info[-2]), color, 2)
            cv2.circle(image, pts2pix(way_pts_info[-1], way_pts_info[-2]), 2, (180, 180, 180), thickness=-1)
            if j == path_span_slice.start:
                cv2.putText(image, 'path_seg:'+str(i), pts2pix(way_pts_info[-1], way_pts_info[-2]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    # draw trajectory
    for i in range(map_start_idx):
        traj_span_slice = polygon_span[i]
        for j in range(traj_span_slice.start, traj_span_slice.stop):
            traj_pts_info = matrix[j]
            color = (64, 180, 165)
            # traj_pts_info: line_pre[0], line_pre[1], x, y, time_stamp, is_av, is_agent, is_others, len(polyline_spans), i
            cv2.line(image, pts2pix(traj_pts_info[0], traj_pts_info[1]), pts2pix(traj_pts_info[2], traj_pts_info[3]), color, 2)
            if j == traj_span_slice.start:
                cv2.putText(image, 'traj:'+str(i), pts2pix(traj_pts_info[2], traj_pts_info[3]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    # draw predicted trajectory if not None
    if "vis.predict_trajs" in mapping:
        pred_trajectory = mapping['vis.predict_trajs']
        # pred_trajectory = pred_trajectory.reshape([6, 30, 2])
        num_traj, num_pts, _ = pred_trajectory.shape
        for i in range(num_traj):
            for j in range(1, num_pts-1): # num_pts-1
                color = (32, 165, 185)
                cv2.line(image, pts2pix(pred_trajectory[i,j-1,0], pred_trajectory[i,j-1,1]), pts2pix(pred_trajectory[i,j,0], pred_trajectory[i,j,1]), color, 2)
            cv2.circle(image, pts2pix(pred_trajectory[i,-1,0], pred_trajectory[i,-1,1]), 2, (0, 255, 255), thickness=-1)
    if "labels" in mapping:
        label = mapping['labels']
        num_pts, _ = label.shape
        for j in range(1, num_pts):
            color = (32, 165, 64)
            cv2.line(image, pts2pix(label[j-1,0], label[j-1,1]), pts2pix(label[j,0], label[j,1]), color, 2)
        cv2.circle(image, pts2pix(label[-1,0], label[-1,1]), 3, (0, 255, 0), thickness=-1)
    
    lane_label_idx = mapping['stage_one_label']
    cv2.putText(image, 'lane_label_idx:'+str(lane_label_idx), (20, h-60), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    if 'goals_2D_labels_xy' in mapping:
        cv2.putText(image, 'goal_label_idx:'+str(mapping['goals_2D_labels'])+', goal_label:'+str(mapping['goals_2D_labels_xy']), (20, h-30), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    cv2.imshow(win_name, image)
    cv2.waitKey(wait_key)
    return image


map_index = 1

class CarlaSyncModeWithTraffic(object):
    """
    Carla client manager with traffic
    """

    def __init__(self):
        self.vehicles_list = []
        self.vehicles_pos_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = carla.Client('127.0.0.1', 2000)  # ip and port
        self.client.set_timeout(5.0)
        self.seed = 16000 # 80899 # random seed, None
        self.respawn = False
        self.hybrid = True
        self.filterv = 'vehicle.*'
        self.generationv = 'All'
        self.number_of_vehicles = 20
        self.max_trajectory_size = 51
        random.seed(self.seed if self.seed is not None else int(time.time()))
        self.world = self.client.load_world('Town0' + str(map_index))
        self.world = self.client.get_world()
        self.spectator = self.world.get_spectator()
        self.traffic_manager = None
        self._setup_client()
        self.hero_actor = None
        self.spawned_hero = None
        self.actors_with_transforms = None
        self.tick_cnt = 0
        # self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.map = self.world.get_map()
        # Start hero mode by default
        self._select_hero_actor()
        self.hero_actor.set_autopilot(True)
        # Set traffic_manager paras
        for i in range(len(self.vehicles_list)):
            acotr_i = self.world.get_actor(self.vehicles_list[i])
            self.traffic_manager.auto_lane_change(acotr_i, True)
            self.traffic_manager.ignore_lights_percentage(acotr_i,80)
            self.traffic_manager.ignore_signs_percentage(acotr_i,80)
            self.traffic_manager.ignore_vehicles_percentage(acotr_i,20)
        self.vehicles_pos_list = []
        for i in range(len(self.vehicles_list)):
            self.vehicles_pos_list.append([])
        self.bound_info, self.lane_info = get_all_lane_info(self.map)
        # pre-tick to fill trajectory buffer
        for i in range(self.max_trajectory_size):
            self.tick()

    def _get_actor_blueprints(self, filter, generation):
        bps = self.world.get_blueprint_library().filter(filter)
        if generation.lower() == "all": return bps
        # If the filter returns only one bp, we assume that this one needed and therefore, we ignore the generation
        if len(bps) == 1: return bps
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("Actor Generation is not valid. No actor will be spawned.")
            return []
        
    def _setup_client(self):
        world, client = self.world, self.client
        traffic_manager = client.get_trafficmanager(8000) # Port to communicate with TM (default: 8000)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if self.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if self.hybrid:
            traffic_manager.set_hybrid_physics_mode(False)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if self.seed is not None:
            traffic_manager.set_random_device_seed(self.seed)
        settings = self.world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        self.traffic_manager = traffic_manager
        if not settings.synchronous_mode:
            synchronous_master = True
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
        else:
            synchronous_master = False
        world.apply_settings(settings)
        blueprints = self._get_actor_blueprints(self.filterv, self.generationv)
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if self.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif self.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, self.number_of_vehicles, number_of_spawn_points)
            self.number_of_vehicles = number_of_spawn_points

        # cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # Spawn vehicles
        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= self.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                self.vehicles_list.append(response.actor_id)
        print('spawned %d vehicles, press Ctrl+C to exit.' % (len(self.vehicles_list)))

    def _select_hero_actor(self):
        hero_vehicles = [actor for actor in self.world.get_actors(
        ) if 'vehicle' in actor.type_id and actor.attributes['role_name'] == 'hero']
        if len(hero_vehicles) > 0:
            self.hero_actor = random.choice(hero_vehicles)
            self.hero_transform = self.hero_actor.get_transform()
        else:
            # Get a random blueprint.
            blueprint = random.choice(self.world.get_blueprint_library().filter(self.filterv))
            blueprint.set_attribute('role_name', 'hero')
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            # Spawn the player.
            while self.hero_actor is None:
                spawn_points = self.world.get_map().get_spawn_points()
                spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
                self.hero_actor = self.world.try_spawn_actor(blueprint, spawn_point)
            self.hero_transform = self.hero_actor.get_transform()
            # Save it in order to destroy it when closing program
            self.spawned_hero = self.hero_actor
            self.vehicles_list.insert(0, self.hero_actor.id)

    def tick(self):
        self.world.tick()
        loc = self.hero_transform.location
        self.spectator.set_transform(
            carla.Transform(carla.Location(x=loc.x,y=loc.y,z=60),
            carla.Rotation(yaw=self.hero_transform.rotation.yaw,pitch=-89.9,roll=0)))  # 89.9 to avoid gimbal lock
        # save the trajectory of all vechicles into a list
        for i in range(len(self.vehicles_list)):
            acotr_i = self.world.get_actor(self.vehicles_list[i])
            pos = acotr_i.get_location()  # calra_loc
            self.vehicles_pos_list[i].append(np.array([pos.x, pos.y]))
            if len(self.vehicles_pos_list[i]) > self.max_trajectory_size:
                self.vehicles_pos_list[i].pop(0)
        # save actor's transforms for visualize
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        self.tick_cnt += 1
        # random set options
        if random.randint(1, 75) == 1: 
            # print("force lane change")
            acotr_i = self.world.get_actor(self.vehicles_list[0])
            self.traffic_manager.force_lane_change(acotr_i, bool(random.choice([True, False]))) # direction: True is the one on the right and False is the left one.
        if random.randint(1, 100) == 1: # every 10s on average
            # self.traffic_manager.global_percentage_speed_difference(random.randint(-100, 50)) # from 50% to 200%
            self.traffic_manager.set_global_distance_to_leading_vehicle(1.5 + 6*random.random())
        if random.randint(1, 100) == 1: 
            for i in range(len(self.vehicles_list)):
                acotr_i = self.world.get_actor(self.vehicles_list[i])
                self.traffic_manager.vehicle_percentage_speed_difference(acotr_i,random.randint(-100, 50))

    def get_vectornet_input(self, mapping):
        angle = (-self.hero_transform.rotation.yaw + 90) * 3.14159265359 / 180.0 # TODO: to be confirmed
        get_vectornet_mapping(all_vehicles_pos_list = self.vehicles_pos_list, agent_angle = angle,
            map_bound_info=self.bound_info, map_lane_info=self.lane_info,
            frame_cnt=self.tick_cnt, mapping=mapping)

    def collect_offline_onestep(self):
        angle = (-self.hero_transform.rotation.yaw + 90) * 3.14159265359 / 180.0 # TODO: to be confirmed
        return self.vehicles_pos_list, angle

    def destroy_vechicles(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        # if self.spawned_hero is not None:
        #     self.spawned_hero.destroy()
        time.sleep(0.25)


save_offline_data = False # if True, will save mapping data as npy and trajectory as csv file
offline_data_path = './carla_offline_data'
offline_data_num_killo = 20  # in K, will * 1000

"""
Training Example:
OUTPUT_DIR=models/carla_offline_data/models.densetnt.carla
python3 src/run.py --argoverse --future_frame_num 30   --do_train --data_dir data/carla_offline_data/16000/ \
    --output_dir ${OUTPUT_DIR} --hidden_size 128 --train_batch_size 64 --use_map   --core_num 16 --use_centerline --distributed_training 1 --other_params semantic_lane direction goals_2D enhance_global_graph subdivide lazy_points laneGCN point_sub_graph \
    stage_one stage_one_dynamic=0.95 laneGCN-4 point_level-4-3 complete_traj complete_traj-3 # --reuse_temp_file
"""
if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    carla_client = CarlaSyncModeWithTraffic()
    try:
        if save_offline_data:
            import os
            import time
            if not os.path.exists(offline_data_path): os.system("mkdir " + offline_data_path)
            # TODO: save lane_info and bound_info as npy
            # carla_client.bound_info, carla_client.lane_info
            offline_data_path = offline_data_path+'/' + str(carla_client.seed) # +'_finetune_test/'
            if not os.path.exists(offline_data_path): os.system("mkdir " + offline_data_path)
            os.system("cp bound_info.npy " + offline_data_path)
            os.system("cp lane_info.npy " + offline_data_path)
            mapping = None
            for i in range(offline_data_num_killo):
                vehicles_pos_lists = []
                agent_angles = []
                offline_data_block_size = 1000
                start_time = time.time()
                for j in range(offline_data_block_size):
                    carla_client.tick()
                    vehicles_pos_list, angle = carla_client.collect_offline_onestep()
                    vehicles_pos_lists.append(np.array(vehicles_pos_list))
                    agent_angles.append(angle)
                    # carla_client.get_vectornet_input(mapping)
                    # draw_vectornet_mapping(mapping, wait_key=10)
                append_name = str(((map_index-1)*offline_data_num_killo + i+1)*offline_data_block_size)
                os.system("cp bound_info.npy " + offline_data_path+'bound_info_'+append_name+'.npy')
                os.system("cp lane_info.npy " + offline_data_path+'lane_info_'+append_name+'.npy')
                np.savez_compressed(offline_data_path+'vehicles_pos_list_'+append_name, vehicles_pos_lists=np.array(vehicles_pos_lists), agent_angles=np.array(agent_angles))
                print("1000 samples generated in "+str(time.time()-start_time)+" sec, current data gen index:" + str(i*offline_data_block_size)) 
        else:
            mapping = {}
            while True:
                carla_client.tick()
                carla_client.get_vectornet_input(mapping)
                draw_vectornet_mapping(mapping, wait_key=10)
    finally:
        carla_client.destroy_vechicles()
