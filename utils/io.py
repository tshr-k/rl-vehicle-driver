import pandas as pd
from typing import NamedTuple


class Colors(NamedTuple):
    BLACK = (0, 0, 0)
    DARK_GRAY = (60, 60, 60)
    LIGHT_GRAY = (200, 200, 200)
    WHITE = (255, 255, 255)

    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLUE = (0, 0, 255)

    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    YELLOW = (255, 255, 0)
    
    DARK_CYAN = (0, 85, 85)
    DARK_YELLOW = (85, 85, 0)
    DARK_MAGENTA = (85, 0, 85)

    ORANGE = (255, 165, 0)
    PURPLE = (128, 0, 128)


def load_track(track_csv_path: str) -> pd.DataFrame:
    """
    Read a csv containing track data into a pandas DataFrame

    Args:
        track_csv_path (str): The path of the csv file.

    Returns:
        pd.DataFrame: The track DataFrame with the following columns:
            - x_centreline: The x coordinates of the track centreline points.
            - y_centreline: The y coordinates of the track centreline points.
            - w_right: The width of the track to the right of the centreline at each point.
            - w_left: The width of the track to the left of the centreline at each point.
    """

    return pd.read_csv(
        track_csv_path, header=0, names=["x_centreline", "y_centreline", "w_right", "w_left"]
    )
