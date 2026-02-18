from dataclasses import dataclass
from utils.geom import Vec2

@dataclass
class CarState:
    position: Vec2
    velocity: Vec2
    acceleration: Vec2
    heading: float  # Angle in radians measured anticlockwise from the positive x-axis
    angular_velocity: float