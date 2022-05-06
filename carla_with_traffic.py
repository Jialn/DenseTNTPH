"""
./CarlaUE4.sh # -RenderOffScreen

"""
import time
import carla
import logging
import pygame
from numpy import random
import math
from carla_visualize import *

class CarlaSyncModeWithTraffic(object):
    """
    Carla client manager with traffic
    """

    def __init__(self):
        self.vehicles_list = []
        self.walkers_list = []
        self.all_id = []
        self.client = carla.Client('127.0.0.1', 2000)  # ip and port
        self.client.set_timeout(5.0)
        self.seed = 115200 # random seed, None
        self.respawn = False
        self.hybrid = False
        self.filterv = 'vehicle.*'
        self.generationv = 'All'
        self.number_of_vehicles = 25
        self.visualize_observation = True
        random.seed(self.seed if self.seed is not None else int(time.time()))
        self.world = self.client.get_world()
        print(self.client.get_available_maps())
        self._setup_client()
        self.hero_actor = None
        self.spawned_hero = None
        self.actors_with_transforms = None
        if self.visualize_observation:
            self.width, self.height = 1920, 1080
            pygame.init()
            self.display = pygame.display.set_mode(
                (self.width, self.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("visualizer")

            self.map = self.world.get_map()
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
        print("list_v:")
        print(len(list_v))
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
            print(corners)
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

        #self.border_round_surface.set_clip(clipping_rect)

        self.hero_surface.fill(COLOR_ALUMINIUM_4)
        self.hero_surface.blit(self.result_surface, (-translation_offset[0], -translation_offset[1]))
        rotated_result_surface = pygame.transform.rotozoom(self.hero_surface, angle, 0.9).convert()

        center = (display.get_width() / 2, display.get_height() / 2)
        rotation_pivot = rotated_result_surface.get_rect(center=center)
        display.blit(rotated_result_surface, rotation_pivot)

        # display.blit(self.border_round_surface, (0, 0))
        pygame.display.flip()

    def tick(self):
        self.world.tick()

        actors = self.world.get_actors()
        self.actors_with_transforms = [(actor, actor.get_transform()) for actor in actors]
        if self.hero_actor is not None:
            self.hero_transform = self.hero_actor.get_transform()

        self.render()

    
    def get_step_observation(self):
        map = self.map
        # Nearest waypoint in the center of a Driving or Sidewalk lane.
        # waypoint01 = map.get_waypoint(vehicle.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
        # Nearest waypoint but specifying OpenDRIVE parameters. 
        # waypoint02 = map.get_waypoint_xodr(road_id,lane_id,s)
        if self.visualize_observation:
            pass
        return None

    def destroy_vechicles(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        settings.no_rendering_mode = False
        settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)
        print('\ndestroying %d vehicles' % len(self.vehicles_list))
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.vehicles_list])
        if self.spawned_hero is not None:
            self.spawned_hero.destroy()
        time.sleep(0.25)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    carla_client = CarlaSyncModeWithTraffic()
    try:
        while True:
            carla_client.tick()
            # print(carla_client.get_step_observation())
    finally:
        carla_client.destroy_vechicles()
