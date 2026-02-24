from dataclasses import dataclass
from typing import Dict, List, Tuple, TypedDict

import gymnasium as gym
import numpy as np
import pygame
import shapely
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from shapely.coords import CoordinateSequence

from environment.car_model import CarModel
from environment.car_state import CarState
from utils.geom import Vec2, calculate_curvature, vec_norm
from utils.io import Colors, load_track


class _LtsObservation(TypedDict):
    kinematic_state: NDArray[
        np.float32
    ]  # [local_vx, local_vy, local_ax, local_ay, cos(angle_to_track), sin(angle_to_track), angular_velocity]
    ranges: NDArray[np.float32]
    future_curvatures: NDArray[np.float32]


@dataclass
class RewardCoefficients:
    """Penalties must be negative since they will be added to the reward"""

    progress: float
    parallel_velocity: float
    survival: float
    collision: float


@dataclass
class ResetOptions:
    """Options for resetting the LTS environment"""

    randomize_start: bool = False  # Whether to randomize the starting position along the track
    randomize_track: bool = (
        False  # Whether to randomize the track (if multiple tracks are available)
    )


class LtsEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30, "name": "LtsEnv-v0"}

    def __init__(
        self,
        track_csv_path: str,
        num_rangefinders: int,
        num_future_curvatures: int,
        car_model: CarModel,
        dt: float = 1 / metadata["render_fps"],  # Use the render FPS by default
        reward_coefficients: RewardCoefficients | None = None,
        max_range: float = 100.0,
        render_mode: str | None = None,
        follow_car: bool = True,
    ):
        super().__init__()

        self._car_model = car_model
        self._dt = dt  # Physical simulation timestep in seconds

        # Initialise lap progress
        self._progress = 0.0  # Fraction of the track completed, from 0 to 1
        self._previous_progress = 0.0  # For reward calculation

        # Store reward coefficients, using default values if not provided
        if reward_coefficients is None:
            reward_coefficients = RewardCoefficients(
                progress=5.0,
                parallel_velocity=1.0,
                survival=+0.01,
                collision=-100.0,
            )
        self._rcs = reward_coefficients

        # Initialise track geometry
        track_df = load_track(track_csv_path)

        # Precompute track angles using central differences
        x_c = track_df["x_centreline"]
        y_c = track_df["y_centreline"]
        track_angles = np.arctan2(
            np.roll(y_c, -1) - np.roll(y_c, 1), np.roll(x_c, -1) - np.roll(x_c, 1)
        )
        self._lerp_track_angle = interp1d(
            np.linspace(0, 1, len(track_angles), endpoint=False),
            track_angles,
            fill_value="extrapolate",  # type: ignore
        )

        self._centreline = shapely.LineString(track_df[["x_centreline", "y_centreline"]])
        self._start_position = np.array(
            [self._centreline.coords[0][0], self._centreline.coords[0][1]]
        )
        self._closed_centreline = shapely.LinearRing(self._centreline)

        outer_boundary = shapely.offset_curve(self._centreline, abs(track_df["w_left"]))[0]
        inner_boundary = shapely.offset_curve(self._centreline, -abs(track_df["w_right"]))[0]
        self._track = shapely.Polygon(outer_boundary, holes=[inner_boundary])
        shapely.prepare(self._track)  # For faster collision detection

        self._curvatures = calculate_curvature(self._closed_centreline, clip_threshold=1.0)

        # Initialise car geometry (just a point for now)
        self._car_shape = shapely.Point(0.0, 0.0)

        # Determine scaling factor to fit track within screen
        self._screen_size = 800
        self._follow_car_zoom = 7.0

        track_bounds = self._track.bounds  # (minx, miny, maxx, maxy)
        track_width = track_bounds[2] - track_bounds[0]
        track_height = track_bounds[3] - track_bounds[1]

        scale_x = self._screen_size / track_width
        scale_y = self._screen_size / track_height

        self._scale = min(scale_x, scale_y) * (self._follow_car_zoom if follow_car else 1.0)

        # Validate and set render mode
        if (render_mode is not None) and (render_mode not in self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.render_mode = render_mode
        self._follow_car = follow_car

        # Set up pygame on first render call
        self._pygame_initialized = False

        # Declare the state variables
        self._state: CarState = CarState(
            position=np.array([0.0, 0.0]),  # Will be set to the start of the track in reset()
            velocity=np.array([0.0, 0.0]),
            acceleration=np.array([0.0, 0.0]),
            heading=0.0,  # Angle in radians measured anticlockwise from the positive x-axis
            angular_velocity=0.0,
        )

        # Store initial heading
        self._initial_heading = self._calculate_angle_to_track()

        # Define the observation and action spaces
        self.num_rangefinders = num_rangefinders
        self.num_future_curvatures = num_future_curvatures
        self._max_range = max_range
        self.observation_space = gym.spaces.Dict(
            {
                "kinematic_state": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "ranges": gym.spaces.Box(
                    low=0.0,
                    high=self._max_range,
                    shape=(num_rangefinders,),
                    dtype=np.float32,
                ),
                "future_curvatures": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(num_future_curvatures,),
                    dtype=np.float32,
                ),
            }
        )

        self.action_space = gym.spaces.Box(
            shape=(2,),
            low=-1.0,
            high=1.0,
            dtype=np.float32,
        )  # [throttle, steering] where negative throttle is braking and negative steering is left turn

        # Array of (start, end) tuples for rendering rangefinder rays
        self._rays: List[Tuple[NDArray, NDArray]] = []
        self._relative_ray_angles = np.linspace(-np.pi / 2, np.pi / 2, self.num_rangefinders)

    def _draw_shapely_shape(
        self,
        canvas: pygame.Surface,
        shape: shapely.geometry.base.BaseGeometry,
        color=Colors.WHITE,
        width=1,
    ):
        def coords_to_pixels(coords: CoordinateSequence) -> list[tuple[int, int]]:
            # Convert world coordinates to pixel coordinates, centering on the car's position
            if self._follow_car:
                return [
                    (
                        round((x - self._state.position[0]) * self._scale + self._screen_size / 2),
                        round((self._state.position[1] - y) * self._scale + self._screen_size / 2),
                    )
                    for x, y in list(coords)
                ]
            else:
                return [
                    (
                        round((x - self._track.bounds[0]) * self._scale),
                        round((self._track.bounds[3] - y) * self._scale),
                    )
                    for x, y in list(coords)
                ]

        # Determine the type of shape and draw accordingly
        match shape:
            case shapely.Polygon() as poly:
                pygame.draw.lines(
                    canvas, color, True, coords_to_pixels(poly.exterior.coords), width
                )
                for interior in poly.interiors:
                    pygame.draw.lines(canvas, color, True, coords_to_pixels(interior.coords), width)
            case shapely.Point() as point:
                pygame.draw.circle(canvas, color, coords_to_pixels(point.coords)[0], width)
            case shapely.LineString() as line:
                pygame.draw.lines(canvas, color, False, coords_to_pixels(line.coords), width)
            case shapely.MultiLineString() as mline:
                for line in mline.geoms:
                    pygame.draw.lines(canvas, color, False, coords_to_pixels(line.coords), width)
            case _:
                raise ValueError(f"Unsupported shape type: {type(shape)}")

    def _draw_action_bars(self, canvas: pygame.Surface, color=Colors.DARK_GRAY, bar_width=10):
        half_screen = self._screen_size / 2

        throttle, steering = self._latest_action
        throttle_length = int(half_screen * np.abs(throttle))
        steering_length = int(half_screen * np.abs(steering))

        # Draw throttle bar (vertical bar on the left)
        pygame.draw.rect(
            canvas,
            color,
            (0, half_screen - (0 if throttle < 0 else throttle_length), bar_width, throttle_length),
        )

        # Draw steering bar (horizontal bar on the top)
        pygame.draw.rect(
            canvas,
            color,
            (
                half_screen
                - (0 if steering < 0 else steering_length),  # Extends rightwards for a right turn
                0,
                steering_length,
                bar_width,
            ),
        )

    def render(self) -> NDArray | None:
        if self.render_mode is None:
            raise ValueError("Cannot render the environment when render_mode is None.")

        # Set up pygame if not already initialised
        if not self._pygame_initialized:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((self._screen_size, self._screen_size))
            elif self.render_mode == "rgb_array":
                self._screen = pygame.Surface((self._screen_size, self._screen_size))
            self._clock = pygame.time.Clock()
            self._pygame_initialized = True

        # Create a canvas to draw on
        canvas = pygame.Surface((self._screen_size, self._screen_size))
        canvas.fill(Colors.BLACK)

        # Draw rangefinder rays
        for ray_start, ray_end in self._rays:
            self._draw_shapely_shape(
                canvas,
                shapely.LineString([ray_start, ray_end]),
                color=Colors.DARK_YELLOW,
                width=1,
            )

        # Draw the track
        self._draw_shapely_shape(canvas, self._track, color=Colors.RED)

        # Draw velocity vector
        velocity_end = self._state.position + self._state.velocity
        self._draw_shapely_shape(
            canvas,
            shapely.LineString([self._state.position, velocity_end]),
            color=Colors.DARK_CYAN,
            width=2,
        )

        # Draw the car
        self._draw_shapely_shape(
            canvas, shapely.Point(self._state.position), color=Colors.GREEN, width=5
        )

        # Draw action bars
        self._draw_action_bars(canvas)

        if self.render_mode == "human":
            # Copy canvas to the visible window
            self._screen.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        pygame.quit()

    def _global_to_local(self, vec: Vec2) -> Vec2:
        # Rotate a vector from world coordinates to the car's local frame
        heading = self._state.heading
        rotation_matrix = np.array(
            [[np.cos(heading), np.sin(heading)], [-np.sin(heading), np.cos(heading)]]
        )
        return rotation_matrix @ vec

    def _update_progress(self):
        car_point = shapely.Point(self._state.position)
        self._progress = self._closed_centreline.project(car_point, normalized=True)

    def _calculate_angle_to_track(self) -> float:
        # Calculate the angle between the car's heading and the tangent to the centreline at the closest point
        track_angle = self._lerp_track_angle(self._progress)
        angle_to_track = track_angle - self._state.heading

        # Normalize angle to [-pi, pi]
        # Negative => car is pointing to the left of the track
        angle_to_track = (angle_to_track + np.pi) % (2 * np.pi) - np.pi
        return angle_to_track

    def _calculate_ranges(self) -> NDArray:
        if self.render_mode == "human":
            self._rays.clear()  # Clear previous rays for rendering

        heading = self._state.heading

        ranges = np.full(
            self.num_rangefinders, self._max_range
        )  # Initialize all ranges to max_range
        for i, angle in enumerate(heading + self._relative_ray_angles):
            ray_direction: Vec2 = np.array([np.cos(angle), np.sin(angle)])
            ray_end = self._state.position + ray_direction * self._max_range
            ray_line = shapely.LineString([self._state.position, ray_end])

            intersections = ray_line.intersection(self._track.boundary)
            if not intersections.is_empty:
                intersection_coords = np.atleast_2d(shapely.get_coordinates(intersections))
                distances = np.linalg.norm(intersection_coords - self._state.position, axis=1)
                min_distance_idx = np.argmin(distances)
                ranges[i] = distances[min_distance_idx]

                # For rendering
                if self.render_mode == "human":
                    self._rays.append((self._state.position, intersection_coords[min_distance_idx]))

        return ranges

    def _get_future_curvatures(self) -> NDArray:
        start_idx = int(self._progress * len(self._curvatures))
        future_curvatures = np.take(
            self._curvatures,
            np.arange(start_idx, start_idx + self.num_future_curvatures),
            mode="wrap",
        )
        return future_curvatures

    def _get_obs(self) -> _LtsObservation:
        # Longitudinal and lateral velocity and acceleration
        local_velocity = self._global_to_local(self._state.velocity)
        local_acceleration = self._global_to_local(self._state.acceleration)
        # Angle between car's heading and tangent to the centreline at the closest point
        angle_to_track = self._calculate_angle_to_track()
        angular_velocity = self._state.angular_velocity
        ranges = self._calculate_ranges()
        future_curvatures = self._get_future_curvatures()

        return _LtsObservation(
            kinematic_state=np.array(
                [
                    local_velocity[0],
                    local_velocity[1],
                    local_acceleration[0],
                    local_acceleration[1],
                    np.cos(angle_to_track),
                    np.sin(angle_to_track),
                    angular_velocity,
                ],
                dtype=np.float32,
            ),
            ranges=ranges.astype(np.float32),
            future_curvatures=future_curvatures.astype(np.float32),
        )

    def _get_info(self) -> Dict:
        return {
            "speed": vec_norm(self._state.velocity),
            "progress": self._progress,
        }

    def reset(
        self, seed: int | None = None, options: ResetOptions | None = None
    ) -> Tuple[_LtsObservation, Dict]:
        super().reset(seed=seed)

        self._progress = 0.0
        self._latest_action = np.array(
            [0.0, 0.0]
        )  # [throttle, steering] required for rendering action bars

        self._state: CarState = CarState(
            position=self._start_position.copy(),  # Don't overwrite the original start position
            velocity=np.array([0.0, 0.0]),
            acceleration=np.array([0.0, 0.0]),
            heading=self._initial_heading,  # Start aligned with the track
            angular_velocity=0.0,
        )

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _calculate_reward(self, action: NDArray, obs: _LtsObservation, is_colliding: bool) -> float:
        reward = 0.0

        # Reward increase in progress along the centreline
        reward += self._rcs.progress * (self._progress - self._previous_progress)
        self._previous_progress = self._progress

        # Reward velocity component in the direction of the track
        speed = vec_norm(self._state.velocity)
        parallel_velocity = speed * obs["kinematic_state"][4]  # cos(angle_to_track)
        reward += self._rcs.parallel_velocity * parallel_velocity

        # Reward survival
        if self._state.velocity[0] < 0.1:
            reward += self._rcs.survival

        # Penalise collisions
        if is_colliding:
            reward += self._rcs.collision

        return reward

    def step(self, action: NDArray):
        # Update the car's state based on the action using the car model
        self._state = self._car_model.step(
            self._state, throttle=action[0], steering=action[1], dt=self._dt
        )

        # Make a new observation after stepping the car
        obs = self._get_obs()

        # Update the car shape for collision detection and rendering
        self._car_shape = shapely.Point(self._state.position)
        self._update_progress()
        self._latest_action = action  # For rendering the action bars

        # Check for collisions
        is_colliding = not self._track.contains(self._car_shape)

        # Render the environment
        if self.render_mode == "human":
            self.render()

        # Calculate reward
        reward = self._calculate_reward(action, obs, is_colliding)

        # Terminate if the car has completed a lap or gone off-track
        terminated = (self._progress >= 0.99) or is_colliding

        # No explicit truncation here, but a wrapper could be used externally if desired
        truncated = False

        return obs, reward, terminated, truncated, self._get_info()
