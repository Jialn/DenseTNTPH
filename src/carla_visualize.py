
import carla
import pygame
import math
from pygame.locals import KMOD_CTRL
from pygame.locals import KMOD_SHIFT
from pygame.locals import K_COMMA
from pygame.locals import K_DOWN
from pygame.locals import K_ESCAPE
from pygame.locals import K_F1
from pygame.locals import K_LEFT
from pygame.locals import K_PERIOD
from pygame.locals import K_RIGHT
from pygame.locals import K_SLASH
from pygame.locals import K_SPACE
from pygame.locals import K_TAB
from pygame.locals import K_UP
from pygame.locals import K_a
from pygame.locals import K_d
from pygame.locals import K_h
from pygame.locals import K_i
from pygame.locals import K_m
from pygame.locals import K_p
from pygame.locals import K_q
from pygame.locals import K_s
from pygame.locals import K_w

# Colors

# We will use the color palette used in Tango Desktop Project (Each color is indexed depending on brightness level)
# See: https://en.wikipedia.org/wiki/Tango_Desktop_Project

COLOR_BUTTER_0 = pygame.Color(252, 233, 79)
COLOR_BUTTER_1 = pygame.Color(237, 212, 0)
COLOR_BUTTER_2 = pygame.Color(196, 160, 0)

COLOR_ORANGE_0 = pygame.Color(252, 175, 62)
COLOR_ORANGE_1 = pygame.Color(245, 121, 0)
COLOR_ORANGE_2 = pygame.Color(209, 92, 0)

COLOR_CHOCOLATE_0 = pygame.Color(233, 185, 110)
COLOR_CHOCOLATE_1 = pygame.Color(193, 125, 17)
COLOR_CHOCOLATE_2 = pygame.Color(143, 89, 2)

COLOR_CHAMELEON_0 = pygame.Color(138, 226, 52)
COLOR_CHAMELEON_1 = pygame.Color(115, 210, 22)
COLOR_CHAMELEON_2 = pygame.Color(78, 154, 6)

COLOR_SKY_BLUE_0 = pygame.Color(114, 159, 207)
COLOR_SKY_BLUE_1 = pygame.Color(52, 101, 164)
COLOR_SKY_BLUE_2 = pygame.Color(32, 74, 135)

COLOR_PLUM_0 = pygame.Color(173, 127, 168)
COLOR_PLUM_1 = pygame.Color(117, 80, 123)
COLOR_PLUM_2 = pygame.Color(92, 53, 102)

COLOR_SCARLET_RED_0 = pygame.Color(239, 41, 41)
COLOR_SCARLET_RED_1 = pygame.Color(204, 0, 0)
COLOR_SCARLET_RED_2 = pygame.Color(164, 0, 0)

COLOR_ALUMINIUM_0 = pygame.Color(238, 238, 236)
COLOR_ALUMINIUM_1 = pygame.Color(211, 215, 207)
COLOR_ALUMINIUM_2 = pygame.Color(186, 189, 182)
COLOR_ALUMINIUM_3 = pygame.Color(136, 138, 133)
COLOR_ALUMINIUM_4 = pygame.Color(85, 87, 83)
COLOR_ALUMINIUM_4_5 = pygame.Color(66, 62, 64)
COLOR_ALUMINIUM_5 = pygame.Color(46, 52, 54)


COLOR_WHITE = pygame.Color(255, 255, 255)
COLOR_BLACK = pygame.Color(0, 0, 0)


# Module Defines
MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'

PIXELS_PER_METER = 12

MAP_DEFAULT_SCALE = 0.1
HERO_DEFAULT_SCALE = 1.0

PIXELS_AHEAD_VEHICLE = 150


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

class Util(object):

    @staticmethod
    def blits(destination_surface, source_surfaces, rect=None, blend_mode=0):
        for surface in source_surfaces:
            destination_surface.blit(surface[0], surface[1], rect, blend_mode)

    @staticmethod
    def length(v):
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def get_bounding_box(actor):
        bb = actor.trigger_volume.extent
        corners = [carla.Location(x=-bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=-bb.y),
                   carla.Location(x=bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=bb.y),
                   carla.Location(x=-bb.x, y=-bb.y)]
        corners = [x + actor.trigger_volume.location for x in corners]
        t = actor.get_transform()
        t.transform(corners)
        return corners


class MapImage(object):
    def __init__(self, carla_world, carla_map, pixels_per_meter, show_triggers=False, show_connections=True, show_spawn_points=False):
        self._pixels_per_meter = pixels_per_meter
        self.scale = 1.0
        self.show_triggers = show_triggers
        self.show_connections = show_connections
        self.show_spawn_points = show_spawn_points

        waypoints = carla_map.generate_waypoints(2)
        margin = 50
        max_x = max(waypoints, key=lambda x: x.transform.location.x).transform.location.x + margin
        max_y = max(waypoints, key=lambda x: x.transform.location.y).transform.location.y + margin
        min_x = min(waypoints, key=lambda x: x.transform.location.x).transform.location.x - margin
        min_y = min(waypoints, key=lambda x: x.transform.location.y).transform.location.y - margin

        self.width = max(max_x - min_x, max_y - min_y)
        self._world_offset = (min_x, min_y)

        width_in_pixels = int(self._pixels_per_meter * self.width)

        self.big_map_surface = pygame.Surface((width_in_pixels, width_in_pixels)).convert()
        self.draw_road_map(self.big_map_surface, carla_world, carla_map, self.world_to_pixel, self.world_to_pixel_width)
        self.surface = self.big_map_surface

    def draw_road_map(self, map_surface, carla_world, carla_map, world_to_pixel, world_to_pixel_width):
        map_surface.fill(COLOR_ALUMINIUM_4)
        precision = 0.05

        def lane_marking_color_to_tango(lane_marking_color):
            tango_color = COLOR_BLACK

            if lane_marking_color == carla.LaneMarkingColor.White:
                tango_color = COLOR_ALUMINIUM_2

            elif lane_marking_color == carla.LaneMarkingColor.Blue:
                tango_color = COLOR_SKY_BLUE_0

            elif lane_marking_color == carla.LaneMarkingColor.Green:
                tango_color = COLOR_CHAMELEON_0

            elif lane_marking_color == carla.LaneMarkingColor.Red:
                tango_color = COLOR_SCARLET_RED_0

            elif lane_marking_color == carla.LaneMarkingColor.Yellow:
                tango_color = COLOR_ORANGE_0

            return tango_color

        def draw_solid_line(surface, color, closed, points, width):
            if len(points) >= 2:
                pygame.draw.lines(surface, color, closed, points, width)

        def draw_broken_line(surface, color, closed, points, width):
            broken_lines = [x for n, x in enumerate(zip(*(iter(points),) * 20)) if n % 3 == 0]
            for line in broken_lines:
                pygame.draw.lines(surface, color, closed, line, width)

        def get_lane_markings(lane_marking_type, lane_marking_color, waypoints, sign):
            margin = 0.20
            if lane_marking_type == carla.LaneMarkingType.Broken or (lane_marking_type == carla.LaneMarkingType.Solid):
                marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
                return [(lane_marking_type, lane_marking_color, marking_1)]
            elif lane_marking_type == carla.LaneMarkingType.SolidBroken or lane_marking_type == carla.LaneMarkingType.BrokenSolid:
                marking_1 = [world_to_pixel(lateral_shift(w.transform, sign * w.lane_width * 0.5)) for w in waypoints]
                marking_2 = [world_to_pixel(lateral_shift(w.transform,
                                                          sign * (w.lane_width * 0.5 + margin * 2))) for w in waypoints]
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking_1),
                        (carla.LaneMarkingType.Broken, lane_marking_color, marking_2)]
            elif lane_marking_type == carla.LaneMarkingType.BrokenBroken:
                marking = [world_to_pixel(lateral_shift(w.transform,
                                                        sign * (w.lane_width * 0.5 - margin))) for w in waypoints]
                return [(carla.LaneMarkingType.Broken, lane_marking_color, marking)]
            elif lane_marking_type == carla.LaneMarkingType.SolidSolid:
                marking = [world_to_pixel(lateral_shift(w.transform,
                                                        sign * ((w.lane_width * 0.5) - margin))) for w in waypoints]
                return [(carla.LaneMarkingType.Solid, lane_marking_color, marking)]

            return [(carla.LaneMarkingType.NONE, carla.LaneMarkingColor.Other, [])]

        def draw_lane_marking(surface, waypoints, is_left):
            sign = -1 if is_left else 1
            lane_marking = None

            marking_type = carla.LaneMarkingType.NONE
            previous_marking_type = carla.LaneMarkingType.NONE

            marking_color = carla.LaneMarkingColor.Other
            previous_marking_color = carla.LaneMarkingColor.Other

            waypoints_list = []
            temp_waypoints = []
            current_lane_marking = carla.LaneMarkingType.NONE
            for sample in waypoints:
                lane_marking = sample.left_lane_marking if sign < 0 else sample.right_lane_marking

                if lane_marking is None:
                    continue

                marking_type = lane_marking.type
                marking_color = lane_marking.color

                if current_lane_marking != marking_type:
                    markings = get_lane_markings(
                        previous_marking_type,
                        lane_marking_color_to_tango(previous_marking_color),
                        temp_waypoints,
                        sign)
                    current_lane_marking = marking_type

                    for marking in markings:
                        waypoints_list.append(marking)

                    temp_waypoints = temp_waypoints[-1:]

                else:
                    temp_waypoints.append((sample))
                    previous_marking_type = marking_type
                    previous_marking_color = marking_color

            # Add last marking
            last_markings = get_lane_markings(
                previous_marking_type,
                lane_marking_color_to_tango(previous_marking_color),
                temp_waypoints,
                sign)
            for marking in last_markings:
                waypoints_list.append(marking)

            for markings in waypoints_list:
                if markings[0] == carla.LaneMarkingType.Solid:
                    draw_solid_line(surface, markings[1], False, markings[2], 2)
                elif markings[0] == carla.LaneMarkingType.Broken:
                    draw_broken_line(surface, markings[1], False, markings[2], 2)

        def draw_arrow(surface, transform, color=COLOR_ALUMINIUM_2):
            transform.rotation.yaw += 180
            forward = transform.get_forward_vector()
            transform.rotation.yaw += 90
            right_dir = transform.get_forward_vector()
            end = transform.location
            start = end - 2.0 * forward
            right = start + 0.8 * forward + 0.4 * right_dir
            left = start + 0.8 * forward - 0.4 * right_dir
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        start, end]], 4)
            pygame.draw.lines(
                surface, color, False, [
                    world_to_pixel(x) for x in [
                        left, start, right]], 4)

        def draw_traffic_signs(surface, font_surface, actor, color=COLOR_ALUMINIUM_2, trigger_color=COLOR_PLUM_0):
            transform = actor.get_transform()
            waypoint = carla_map.get_waypoint(transform.location)

            angle = -waypoint.transform.rotation.yaw - 90.0
            font_surface = pygame.transform.rotate(font_surface, angle)
            pixel_pos = world_to_pixel(waypoint.transform.location)
            offset = font_surface.get_rect(center=(pixel_pos[0], pixel_pos[1]))
            surface.blit(font_surface, offset)

            # Draw line in front of stop
            forward_vector = carla.Location(waypoint.transform.get_forward_vector())
            left_vector = carla.Location(-forward_vector.y, forward_vector.x,
                                         forward_vector.z) * waypoint.lane_width / 2 * 0.7

            line = [(waypoint.transform.location + (forward_vector * 1.5) + (left_vector)),
                    (waypoint.transform.location + (forward_vector * 1.5) - (left_vector))]

            line_pixel = [world_to_pixel(p) for p in line]
            pygame.draw.lines(surface, color, True, line_pixel, 2)

        def lateral_shift(transform, shift):
            transform.rotation.yaw += 90
            return transform.location + shift * transform.get_forward_vector()

        def draw_topology(carla_topology, index):
            topology = [x[index] for x in carla_topology]
            topology = sorted(topology, key=lambda w: w.transform.location.z)
            for waypoint in topology:
                # if waypoint.road_id == 150 or waypoint.road_id == 16:
                waypoints = [waypoint]

                nxt = waypoint.next(precision)
                if len(nxt) > 0:
                    nxt = nxt[0]
                    while nxt.road_id == waypoint.road_id:
                        waypoints.append(nxt)
                        nxt = nxt.next(precision)
                        if len(nxt) > 0:
                            nxt = nxt[0]
                        else:
                            break

                # Draw Road
                road_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in waypoints]
                road_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in waypoints]

                polygon = road_left_side + [x for x in reversed(road_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon, 5)
                    pygame.draw.polygon(map_surface, COLOR_ALUMINIUM_5, polygon)

                # Draw Shoulders and Parkings
                PARKING_COLOR = COLOR_ALUMINIUM_4_5
                SHOULDER_COLOR = COLOR_ALUMINIUM_5

                final_color = SHOULDER_COLOR

                # Draw Right
                shoulder = []
                for w in waypoints:
                    r = w.get_right_lane()
                    if r is not None and (
                            r.lane_type == carla.LaneType.Shoulder or r.lane_type == carla.LaneType.Parking):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in shoulder]
                shoulder_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in shoulder]

                polygon = shoulder_left_side + [x for x in reversed(shoulder_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, final_color, polygon, 5)
                    pygame.draw.polygon(map_surface, final_color, polygon)

                draw_lane_marking(
                    map_surface,
                    shoulder,
                    False)

                # Draw Left
                shoulder = []
                for w in waypoints:
                    r = w.get_left_lane()
                    if r is not None and (
                            r.lane_type == carla.LaneType.Shoulder or r.lane_type == carla.LaneType.Parking):
                        if r.lane_type == carla.LaneType.Parking:
                            final_color = PARKING_COLOR
                        shoulder.append(r)

                shoulder_left_side = [lateral_shift(w.transform, -w.lane_width * 0.5) for w in shoulder]
                shoulder_right_side = [lateral_shift(w.transform, w.lane_width * 0.5) for w in shoulder]

                polygon = shoulder_left_side + [x for x in reversed(shoulder_right_side)]
                polygon = [world_to_pixel(x) for x in polygon]

                if len(polygon) > 2:
                    pygame.draw.polygon(map_surface, final_color, polygon, 5)
                    pygame.draw.polygon(map_surface, final_color, polygon)

                draw_lane_marking(
                    map_surface,
                    shoulder,
                    True)

                # Draw Lane Markings and Arrows
                if not waypoint.is_intersection:
                    draw_lane_marking(
                        map_surface,
                        waypoints,
                        True)
                    draw_lane_marking(
                        map_surface,
                        waypoints,
                        False)
                    for n, wp in enumerate(waypoints):
                        if ((n + 1) % 400) == 0:
                            draw_arrow(map_surface, wp.transform)

        topology = carla_map.get_topology()
        draw_topology(topology, 0)
        draw_topology(topology, 1)

        if self.show_spawn_points:
            for sp in carla_map.get_spawn_points():
                draw_arrow(map_surface, sp, color=COLOR_CHOCOLATE_0)

        if self.show_connections:
            dist = 1.5
            to_pixel = lambda wp: world_to_pixel(wp.transform.location)
            for wp in carla_map.generate_waypoints(dist):
                col = (0, 255, 255) if wp.is_intersection else (0, 255, 0)
                for nxt in wp.next(dist):
                    pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(nxt), 2)
                if wp.lane_change & carla.LaneChange.Right:
                    r = wp.get_right_lane()
                    if r and r.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(r), 2)
                if wp.lane_change & carla.LaneChange.Left:
                    l = wp.get_left_lane()
                    if l and l.lane_type == carla.LaneType.Driving:
                        pygame.draw.line(map_surface, col, to_pixel(wp), to_pixel(l), 2)

        actors = carla_world.get_actors()

        # Draw Traffic Signs
        font_size = world_to_pixel_width(1)
        font = pygame.font.SysFont('Arial', font_size, True)

        stops = [actor for actor in actors if 'stop' in actor.type_id]
        yields = [actor for actor in actors if 'yield' in actor.type_id]

        stop_font_surface = font.render("STOP", False, COLOR_ALUMINIUM_2)
        stop_font_surface = pygame.transform.scale(
            stop_font_surface, (stop_font_surface.get_width(), stop_font_surface.get_height() * 2))

        yield_font_surface = font.render("YIELD", False, COLOR_ALUMINIUM_2)
        yield_font_surface = pygame.transform.scale(
            yield_font_surface, (yield_font_surface.get_width(), yield_font_surface.get_height() * 2))

        for ts_stop in stops:
            draw_traffic_signs(map_surface, stop_font_surface, ts_stop, trigger_color=COLOR_SCARLET_RED_1)

        for ts_yield in yields:
            draw_traffic_signs(map_surface, yield_font_surface, ts_yield, trigger_color=COLOR_ORANGE_1)

    def world_to_pixel(self, location, offset=(0, 0)):
        x = self.scale * self._pixels_per_meter * (location.x - self._world_offset[0])
        y = self.scale * self._pixels_per_meter * (location.y - self._world_offset[1])
        return [int(x - offset[0]), int(y - offset[1])]

    def world_to_pixel_width(self, width):
        return int(self.scale * self._pixels_per_meter * width)

    def scale_map(self, scale):
        if scale != self.scale:
            self.scale = scale
            width = int(self.big_map_surface.get_width() * self.scale)
            self.surface = pygame.transform.smoothscale(self.big_map_surface, (width, width))
