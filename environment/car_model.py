from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from scipy.interpolate import CubicSpline

from environment.car_state import CarState
from utils.geom import vec_norm


@dataclass
class CarModel(ABC):
    """A stateless model for the car's dynamics."""

    air_density: ClassVar[float] = 1.225
    g: ClassVar[float] = 9.81

    # Geometric and kinematic parameters
    wheelbase: float = 3.6
    com_to_rear: float = 1.65
    max_steering_angle: float = np.radians(30)
    max_steering_rate: float = np.radians(90)
    frontal_area: float = 1.5

    # Dynamic parameters
    drag_coefficient: float = 0.8
    downforce_coefficient: float = 4.0
    rolling_friction_coefficient: float = 0.020
    mass: float = 850.0
    max_thrust: float = 11_000.0
    max_brake_force: float = -38_000.0
    tyre_friction_coefficient: float = 1.5

    def __post_init__(self) -> None:
        # Ensure braking force is always negative
        self.max_brake_force = -abs(self.max_brake_force)

    def _compute_drag(self, speed: float) -> float:
        return 0.5 * self.air_density * self.frontal_area * self.drag_coefficient * speed**2

    def _compute_downforce(self, speed: float) -> float:
        return 0.5 * self.air_density * self.frontal_area * self.downforce_coefficient * speed**2

    def _compute_normal_force(self, speed: float) -> float:
        return self.mass * self.g + self._compute_downforce(speed)

    def _compute_rolling_friction(self, speed: float) -> float:
        normal_force = self._compute_normal_force(speed)
        return self.rolling_friction_coefficient * normal_force

    @abstractmethod
    def step(self, state: CarState, throttle: float, steering: float, dt: float) -> CarState:
        """Compute the next state of the car given the current state and the time step.

        Args:
            state (CarState): The current state of the car. ALl parameters are in SI units.
            throttle (float): The throttle input, in the range [-1, 1], where -1 is full reverse and 1 is full forward.
            steering (float): The steering input, in the range [-1, 1], where -1 is full right and 1 is full left.
            dt (float): The time step in seconds over which to integrate the car's motion.
        """


@dataclass
class KinematicBicycleModel(CarModel):
    def __post_init__(self) -> None:
        super().__post_init__()

        throttle_waypoints = np.array([-1, 0, 1])
        force_waypoints = np.array([self.max_brake_force, 0, self.max_thrust])

        self._throttle_to_force = CubicSpline(
            throttle_waypoints,
            force_waypoints,
            bc_type=("natural", "clamped"),  # type: ignore
        )

    def step(self, state: CarState, throttle: float, steering: float, dt: float) -> CarState:
        # 1. Convert inputs to physical values
        driving_force = self._throttle_to_force(throttle)
        delta = steering * self.max_steering_angle

        # 2. Update Scalar Speed (Inertia)
        # We treat speed as a float. Friction/Drag can be added here.
        current_speed = vec_norm(state.velocity)

        # Add drag and rolling friction
        drag = self._compute_drag(current_speed)
        rolling_friction = self._compute_rolling_friction(current_speed)
        f_net = driving_force - np.sign(current_speed) * (drag + rolling_friction)

        # Clamp net force based on tyre grip limits
        f_max = self.tyre_friction_coefficient * self._compute_normal_force(current_speed)
        f_net = np.clip(f_net, -f_max, f_max)

        # Calculate acceleration and update speed
        a_net = f_net / self.mass
        speed_next = current_speed + (a_net * dt)

        speed_next = max(0, speed_next)  # Prevent magic reversing unless intended

        # 3. Calculate Angular Velocity (How fast we are turning)
        # Based on the new speed
        omega_next = (speed_next / self.wheelbase) * np.tan(delta)

        # Calculate required lateral acceleration to make this turn
        required_lat_accel = abs(speed_next**2 * np.tan(delta) / self.wheelbase)

        # Calculate max available lateral force and corresponding acceleration based on tyre friction limits
        max_lat_force = (
            self.mass * self.g + self._compute_downforce(speed_next)
        ) * self.tyre_friction_coefficient
        max_lat_accel = max_lat_force / self.mass

        if required_lat_accel > max_lat_accel:
            # Scale omega down because the car is understeering/sliding
            omega_next *= max_lat_accel / required_lat_accel

        # 4. Update Heading
        theta_next = state.heading + (omega_next * dt)

        # 5. Update Position (Using the average heading for smoother integration)
        avg_theta = (state.heading + theta_next) / 2
        heading_vec = np.array([np.cos(avg_theta), np.sin(avg_theta)])

        # Movement is simply speed * direction
        pos_next = state.position + (speed_next * heading_vec * dt)

        # 6. Reconstruct Velocity Vector for the next state
        vel_next = speed_next * heading_vec
        accel_next = f_net * heading_vec / self.mass

        return CarState(
            position=pos_next,
            velocity=vel_next,
            acceleration=accel_next,
            heading=theta_next,
            angular_velocity=omega_next,
        )
