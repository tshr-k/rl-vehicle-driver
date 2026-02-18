import numpy as np
import shapely
from numpy.typing import NDArray

Vec2 = NDArray


def vec_norm(v: Vec2) -> float:
    """Compute the Euclidean norm of a 2D vector.
    
    Args:
        v (Vec2): A 2D vector.
    """
    return np.sqrt(v[0]**2 + v[1]**2)


def calculate_curvature(curve: shapely.LineString, clip_threshold: float | None = None) -> NDArray:
    """Calculate the Menger curvature of a shapely LineString or LinearRing.

    Args:
        curve (shapely.LineString | shapely.LinearRing): The curve for which to compute the curvature.
        clip_threshold (float): Maximum magnitude of curvature to allow. Default is None (no clipping).

    Returns:
        curvature (NDArray): An array of the same length as `curve` containing the curvature at each point. If curve is a LineString, this array is padded with a zero on either end.
    """

    points = shapely.get_coordinates(curve)
    x = points[:, 0]
    y = points[:, 1]

    # Pre-calculate differences
    if isinstance(curve, shapely.LinearRing):
        # Use periodic differences for closed curve
        xa = np.roll(x, 1)
        ya = np.roll(y, 1)
        xb = np.roll(x, -1)
        yb = np.roll(y, -1)

        dx1 = x - xa
        dy1 = y - ya
        dx2 = xb - x
        dy2 = yb - y
        dx3 = xb - xa
        dy3 = yb - ya
    else:
        dx1 = x[1:-1] - x[:-2]
        dy1 = y[1:-1] - y[:-2]
        dx2 = x[2:] - x[1:-1]
        dy2 = y[2:] - y[1:-1]
        dx3 = x[2:] - x[:-2]
        dy3 = y[2:] - y[:-2]

    # Side lengths squared
    a2 = dx1**2 + dy1**2
    b2 = dx2**2 + dy2**2
    c2 = dx3**2 + dy3**2

    # Area using 2D cross product: |x1*y2 - y1*x2|
    # This is 2 * Area
    area_2 = np.abs(dx1 * dy2 - dy1 * dx2)

    # Curvature: 4 * Area / (a * b * c)
    # Equivalent to: 2 * (2 * Area) / sqrt(a2 * b2 * c2)
    curvature = (2 * area_2) / (np.sqrt(a2 * b2 * c2) + 1e-12)

    # Clip curvature to avoid extreme values
    if clip_threshold is not None:
        curvature = np.clip(curvature, -clip_threshold, clip_threshold)

    if isinstance(curve, shapely.LinearRing):
        return curvature
    else:
        return np.pad(curvature, 1, mode="constant", constant_values=0)
