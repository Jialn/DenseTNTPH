"""
./CarlaUE4.sh # -RenderOffScreen
# to use less GPU memory:
./CarlaUE4.sh  -RenderOffScreen -quality-level=Low 
# testing:
python src/carla_with_traffic.py

"""
import time
import carla
import logging
import pygame
from numpy import random
import numpy as np
import math
from carla_visualize import *
from utils import rotate, get_subdivide_points, get_dis
from carla_submap_wrapper import get_lane_ids_in_xy_bbox, get_lane_segment_centerline, city_lane_centerlines_dict, get_all_lane_info


def draw_matrix(matrix, polygon_span, map_start_idx, pred_trajectory=None, win_name="matrix_vis", wait_key=None):
    import cv2
    w, h = 1600, 1600
    offset = (w//2, h//2)
    pix_meter = 0.125
    image = np.zeros((h, w, 3), np.uint8)

    def pts2pix(pts_x, pts_y):
        new_pts = np.array([- pts_x / pix_meter + offset[0], - pts_y / pix_meter + offset[1]]).astype(np.int)
        return (new_pts[0], new_pts[1])
        
    # draw submap
    for i in range(map_start_idx,len(polygon_span)):
        path_span_slice = polygon_span[i]
        for j in range(path_span_slice.start, path_span_slice.stop):
            way_pts_info = matrix[j]
            color = (80, 80, 80)
            cv2.line(image, pts2pix(way_pts_info[-3], way_pts_info[-4]), pts2pix(way_pts_info[-1], way_pts_info[-2]), color, 2)
            cv2.circle(image, pts2pix(way_pts_info[-1], way_pts_info[-2]), 2, (0, 128, 128), thickness=-1)
            if j == path_span_slice.start:
                cv2.putText(image, 'path_seg:'+str(i), pts2pix(way_pts_info[-1], way_pts_info[-2]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    # draw trajectory
    for i in range(map_start_idx):
        traj_span_slice = polygon_span[i]
        for j in range(traj_span_slice.start, traj_span_slice.stop):
            traj_pts_info = matrix[j]
            color = (64, 192, 64)
            # traj_pts_info: line_pre[0], line_pre[1], x, y, time_stamp, is_av, is_agent, is_others, len(polyline_spans), i
            cv2.line(image, pts2pix(traj_pts_info[0], traj_pts_info[1]), pts2pix(traj_pts_info[2], traj_pts_info[3]), color, 2)
            if j == traj_span_slice.start:
                cv2.putText(image, 'traj:'+str(i), pts2pix(traj_pts_info[2], traj_pts_info[3]), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), thickness=1)
    # draw predicted trajectory if not None
    if pred_trajectory is not None:
        # pred_trajectory = pred_trajectory.reshape([6, 30, 2])
        num_traj, num_pts, _ = pred_trajectory.shape
        for i in range(num_traj):
            for j in range(1, num_pts): # num_pts-1
                color = (64, 64, 255)
                cv2.line(image, pts2pix(pred_trajectory[i,j-1,0], pred_trajectory[i,j-1,1]), pts2pix(pred_trajectory[i,j,0], pred_trajectory[i,j,1]), color, 2)

    cv2.imshow(win_name, image)
    cv2.waitKey(wait_key)
    return image


class CarlaSyncModeWithTraffic(object):
    """
    Carla client manager with traffic
    """

    def __init__(self):
        self.vehicles_list = []
        self.vehicles_pos_list = []
        self.hero_pos_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = carla.Client('127.0.0.1', 2000)  # ip and port
        self.client.set_timeout(5.0)
        self.seed = 16000 # 80899 # random seed, None
        self.respawn = False
        self.hybrid = False
        self.filterv = 'vehicle.*'
        self.generationv = 'All'
        self.number_of_vehicles = 20
        self.max_trajectory_size = 51
        self.vector_net_hidden_size = 128
        self.visualize_carla = True
        random.seed(self.seed if self.seed is not None else int(time.time()))
        self.world = self.client.get_world()
        # print(self.client.get_available_maps())
        self._setup_client()
        self.hero_actor = None
        self.spawned_hero = None
        self.actors_with_transforms = None
        self.tick_cnt = 0
        # self.world.unload_map_layer(carla.MapLayer.Buildings)
        self.map = self.world.get_map()
        if self.visualize_carla:
            self.width, self.height = 1920, 1080
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("visualizer")

            self.map_image = MapImage(carla_world=self.world,
                carla_map=self.map, pixels_per_meter=PIXELS_PER_METER)
            self.original_surface_size = self.height
            self.surface_size = self.map_image.big_map_surface.get_width()
            # Render Actors
            self.actors_surface = pygame.Surface((self.map_image.surface.get_width(), self.map_image.surface.get_height()))
            self.actors_surface.set_colorkey(COLOR_BLACK)
            self.border_round_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA).convert()
            self.border_round_surface.set_colorkey(COLOR_WHITE)
            self.border_round_surface.fill(COLOR_BLACK)
            center_offset = (int(self.width / 2), int(self.height / 2))
            pygame.draw.circle(self.border_round_surface, COLOR_ALUMINIUM_1, center_offset, int(self.height / 2))
            pygame.draw.circle(self.border_round_surface, COLOR_WHITE, center_offset, int((self.height - 8) / 2))
            scaled_original_size = self.original_surface_size * (1.0 / 0.9)
            self.hero_surface = pygame.Surface((scaled_original_size, scaled_original_size)).convert()
            self.result_surface = pygame.Surface((self.surface_size, self.surface_size)).convert()
            self.result_surface.set_colorkey(COLOR_BLACK)
        # Start hero mode by default
        self._select_hero_actor()
        self.hero_actor.set_autopilot(True)
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
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if self.seed is not None:
            traffic_manager.set_random_device_seed(self.seed)
        settings = self.world.get_settings()
        traffic_manager.set_synchronous_mode(True)
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
        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

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

    def _split_actors(self):
        vehicles, traffic_lights, speed_limits, walkers = [], [], [], []
        for actor_with_transform in self.actors_with_transforms:
            actor = actor_with_transform[0]
            if 'vehicle' in actor.type_id:
                vehicles.append(actor_with_transform)
            elif 'traffic_light' in actor.type_id:
                traffic_lights.append(actor_with_transform)
            elif 'speed_limit' in actor.type_id:
                speed_limits.append(actor_with_transform)
            elif 'walker' in actor.type_id:
                walkers.append(actor_with_transform)
        return (vehicles, traffic_lights, speed_limits, walkers)

    def _render_walkers(self, surface, list_w, world_to_pixel):
        for w in list_w:
            color = COLOR_PLUM_0
            # compute bounding box points
            bb = w[0].bounding_box.extent
            corners = [
                carla.Location(x=-bb.x, y=-bb.y), carla.Location(x=bb.x, y=-bb.y),
                carla.Location(x=bb.x, y=bb.y), carla.Location(x=-bb.x, y=bb.y)]
            w[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            pygame.draw.polygon(surface, color, corners)

    def _render_vehicles(self, surface, list_v, world_to_pixel):
        # print("list_v:")
        # print(len(list_v))
        for v in list_v:
            color = COLOR_SKY_BLUE_0
            if int(v[0].attributes['number_of_wheels']) == 2:
                color = COLOR_CHOCOLATE_1
            if v[0].attributes['role_name'] == 'hero':
                color = COLOR_CHAMELEON_0
            # Compute bounding box points
            bb = v[0].bounding_box.extent
            corners = [carla.Location(x=-bb.x, y=-bb.y), carla.Location(x=bb.x - 0.8, y=-bb.y),
                       carla.Location(x=bb.x, y=0), carla.Location(x=bb.x - 0.8, y=bb.y),
                       carla.Location(x=-bb.x, y=bb.y), carla.Location(x=-bb.x, y=-bb.y)]
            v[1].transform(corners)
            corners = [world_to_pixel(p) for p in corners]
            # print(corners)
            pygame.draw.lines(surface, color, False, corners, int(math.ceil(4.0 * self.map_image.scale)))

    def render_actors(self, surface, vehicles, walkers):
        self._render_vehicles(surface, vehicles, self.map_image.world_to_pixel)
        self._render_walkers(surface, walkers, self.map_image.world_to_pixel)

    def clip_surfaces(self, clipping_rect):
        self.actors_surface.set_clip(clipping_rect)
        self.result_surface.set_clip(clipping_rect)

    def render(self):
        if self.actors_with_transforms is None:
            print("no actors_with_transforms!")
            return
        display = self.display
        self.result_surface.fill(COLOR_BLACK)
        vehicles, traffic_lights, speed_limits, walkers = self._split_actors()
        # render
        self.actors_surface.fill(COLOR_BLACK)
        self.render_actors(self.actors_surface, vehicles, walkers)
        # blit surfaces
        surfaces = ((self.map_image.surface, (0, 0)),
                    (self.actors_surface, (0, 0)))
        angle = self.hero_transform.rotation.yaw + 90.0
        hero_location_screen = self.map_image.world_to_pixel(self.hero_transform.location)
        hero_front = self.hero_transform.get_forward_vector()
        translation_offset = (hero_location_screen[0] - self.hero_surface.get_width() / 2 + hero_front.x * PIXELS_AHEAD_VEHICLE,
            (hero_location_screen[1] - self.hero_surface.get_height() / 2 + hero_front.y * PIXELS_AHEAD_VEHICLE))
        ## apply clipping rect
        clipping_rect = pygame.Rect(translation_offset[0],
                                    translation_offset[1],
                                    self.hero_surface.get_width(),
                                    self.hero_surface.get_height())
        self.clip_surfaces(clipping_rect)
        Util.blits(self.result_surface, surfaces)
        self.border_round_surface.set_clip(clipping_rect)
        self.hero_surface.fill(COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, (-translation_offset[0], -translation_offset[1]))
        rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()
        center = (display.get_width() / 2, display.get_height() / 2)
        rotation_pivot = rotated_result_surface.get_rect(center=center)
        display.blit(rotated_result_surface, rotation_pivot)
        display.blit(self.border_round_surface, (0, 0))
        pygame.display.flip()

    def tick(self):
        self.world.tick()
        # save the trajectory of all vechicles into a list
        for i in range(len(self.vehicles_list)):
            acotr_i = self.world.get_actor(self.vehicles_list[i])
            pos = acotr_i.get_location()  # calra_loc
            self.vehicles_pos_list[i].append(np.array([pos.x, pos.y]))
            if len(self.vehicles_pos_list[i]) > self.max_trajectory_size:
                self.vehicles_pos_list[i].pop(0)
            # print( acotr_i.attributes['role_name'])
            if acotr_i.attributes['role_name'] == 'hero':
                self.hero_pos_list.append(np.array([pos.x, pos.y]))
                if len(self.hero_pos_list) > self.max_trajectory_size:
                    self.hero_pos_list.pop(0)
        # save actor's transforms for visualize
        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()
        # visualize
        if self.visualize_carla:
            self.render()
        self.tick_cnt += 1

    def get_vectornet_input(self, mapping):
        '''
        mapping keys:
            # already impl:
            'cent_x', 'cent_y', 'angle', 'trajs', 'map_start_polyline_idx', 'polygons', 
            'goals_2D', 'matrix', 'polyline_spans', 'origin_labels',  'goals_2D_labels', 'stage_one_label',  'labels', 'labels_is_valid', 'eval_time'
            # not in dataset infer: 'stage_one_scores', 'stage_one_topk', 'set_predict_ans_points', 'vis.predict_trajs', 'file_name', 'agents'
            # not used outside dataset construct: 'start_time' , 'two_seconds', 'city_name', 'agent_pred_index'
        '''
        two_second_index = 20
        min_distance_submap = 35
        max_distance_for_agents = 70
        polyline_spans = []
        vectors = []
        trajs = []
        map_start_polyline_idx = None
        agent_loc = self.hero_pos_list[two_second_index]
        angle = (-self.hero_transform.rotation.yaw + 90) * 3.14159265359 / 180.0 # TODO: to be confirmed

        def get_pad_vector(li, hidden_size):
            # Pad vector to hidden_size
            assert len(li) <= hidden_size
            li.extend([0] * (hidden_size - len(li)))
            return li
        
        def get_hash(point):
            return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

        # get vehicles' trajectory
        for vhid in range(len(self.vehicles_pos_list)):
            vh_loc = self.vehicles_pos_list[vhid][two_second_index]
            # print(vh_loc, end=",")
            if abs(vh_loc[0] - agent_loc[0]) < max_distance_for_agents and abs(vh_loc[1] - agent_loc[1]) < max_distance_for_agents:
                start = len(vectors)
                # traj for denseTNT visualize:
                traj = []
                for pos in self.vehicles_pos_list[vhid]:
                    traj.append(pos[0])
                    traj.append(pos[1])
                traj = np.array(traj).reshape((-1, 2))
                trajs.append(traj)
                # trajectory for prediction
                is_agent = self.world.get_actor(self.vehicles_list[vhid]).attributes['role_name'] == 'hero'
                is_others, is_av = not is_agent, False
                for i, line in enumerate(self.vehicles_pos_list[vhid]):
                    x, y = line[0], line[1]
                    time_stamp = (i - two_second_index) * 0.1
                    # if i > 0: # For testing
                    if i > 0 and i < two_second_index:
                        line_pre = self.vehicles_pos_list[vhid][i-1].copy()
                        # print(x-line_pre[X], y-line_pre[Y])
                        x, y = rotate(x - agent_loc[0], y - agent_loc[1], angle)
                        line_pre[0], line_pre[1]= rotate(line_pre[0] - agent_loc[0], line_pre[1] - agent_loc[1], angle)
                        vector = [line_pre[0], line_pre[1], x, y, time_stamp, is_av,
                                is_agent, is_others, len(polyline_spans), i]
                        vectors.append(get_pad_vector(vector, self.vector_net_hidden_size))
                # set polyline_spans
                end = len(vectors)
                polyline_spans.append([start, end])
            # end vehicles' trajectory
        map_start_polyline_idx = len(polyline_spans)
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(self.vehicles_pos_list[0][20:50]):
            origin_labels[i][0], origin_labels[i][1] = line[0], line[1]

        # get sub-map around agent location
        VECTOR_PRE_X = 0
        VECTOR_PRE_Y = 1
        VECTOR_X = 2
        VECTOR_Y = 3
        lane_ids = get_lane_ids_in_xy_bbox(agent_loc[0], agent_loc[1], self.bound_info, min_distance_submap)
        local_lane_centerlines = [get_lane_segment_centerline(lane_id, self.lane_info) for lane_id in lane_ids]
        polygons = local_lane_centerlines
        polygons = [polygon[:, :2].copy() for polygon in polygons]
        for index_polygon, polygon in enumerate(polygons):
            for i, point in enumerate(polygon):
                point[0], point[1] = rotate(point[0] - agent_loc[0], point[1] - agent_loc[1], angle)
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
            subdivide_points = get_subdivide_points(polygon, include_self=True)
        mapping['goals_2D'] = np.array(points)
        
        labels = []
        for i, line in enumerate(self.vehicles_pos_list[0][20:50]):
            labels.append(line[0])
            labels.append(line[1])
        point_label = np.array(labels[-2:])
        mapping['goals_2D_labels'] = np.argmin(get_dis(mapping['goals_2D'], point_label))

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
            lane_segment_dict = city_lane_centerlines_dict(lane_id, self.lane_info)
            # assert_(len(polygon) >= 2)
            for i, point in enumerate(polygon):
                if i > 0:
                    vector = [0] * self.vector_net_hidden_size # args.hidden_size
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

        # update dict
        mapping.update(dict(
            matrix=np.array(vectors),
            labels=np.array(labels).reshape([30, 2]),
            polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
            labels_is_valid=np.ones(30, dtype=np.int64),
            eval_time=30, cent_x=agent_loc[0], cent_y=agent_loc[1],
            map_start_polyline_idx=map_start_polyline_idx, polygons=polygons,
            traj=traj, angle=angle, origin_labels=origin_labels,
            file_name='carla_'+str(self.tick_cnt)
        ))

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


save_offline_data = True # if True, will save mapping data as npy and trajectory as csv file
offline_data_path = './carla_offline_data'
offline_data_num = 20 * 1000 # 20K


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    carla_client = CarlaSyncModeWithTraffic()
    try:
        if save_offline_data:
            # TODO: save lane_info and bound_info as npy
            # carla_client.bound_info, carla_client.lane_info
            mapping = {}
            for i in range(offline_data_num):
                carla_client.tick()
                carla_client.get_vectornet_input(mapping)
                draw_matrix(mapping['matrix'], mapping['polyline_spans'], mapping['map_start_polyline_idx'], wait_key=10)
                # TODO: save trajectory as csv file
                # carla_client.vehicles_pos_list should do the work, the first one is agent and others are others
                pass
                # after this part done, change dataset_carla.py accrodingly
        else:
            mapping = {}
            while True:
                carla_client.tick()
                carla_client.get_vectornet_input(mapping)
                draw_matrix(mapping['matrix'], mapping['polyline_spans'], mapping['map_start_polyline_idx'], wait_key=10)
    finally:
        carla_client.destroy_vechicles()
