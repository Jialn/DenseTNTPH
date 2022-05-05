"""
./CarlaUE4.sh # -RenderOffScreen

"""
import time
import carla
import logging
from numpy import random

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
        self.number_of_vehicles = 50
        self.visualize_observation = True
        random.seed(self.seed if self.seed is not None else int(time.time()))
        self.world = self.client.get_world()
        print(self.client.get_available_maps())
        self._setup_client()

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
        
    def tick(self):
        self.world.tick()

    
    def get_step_observation(self):
        map = self.world.get_map()
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
        time.sleep(0.25)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    carla_client = CarlaSyncModeWithTraffic()
    try:
        while True:
            carla_client.tick()
            print(carla_client.get_step_observation())
    finally:
        carla_client.destroy_vechicles()
