from enum import Enum
import numpy as np
import random

class GameConstants:
    """Stores only static game constants"""

    # Map
    MAP_WIDTH = 24
    MAP_HEIGHT = 24
    MAP_SIZE = MAP_WIDTH * MAP_HEIGHT
    NUM_TEAMS = 2
    MATCH_COUNT_PER_EPISODE = 5
    MAX_STEPS_IN_MATCH = 100

    # Units
    MAX_UNITS = 16
    INIT_UNIT_ENERGY = 100
    MIN_UNIT_ENERGY = 0
    MAX_UNIT_ENERGY = 400

    # Energy and Resource Configurations
    MAX_ENERGY_NODES = 6
    MAX_ENERGY_PER_TILE = 20
    MIN_ENERGY_PER_TILE = -20

    # Relic Nodes
    MAX_RELIC_NODES = 6
    RELIC_CONFIG_SIZE = 5


class ActionType(Enum):
    """Represents valid unit actions in the Lux AI Season 3 game."""

    STAY = 0  # Do nothing
    MOVE_UP = 1  # Move north (y-1)
    MOVE_RIGHT = 2  # Move east (x+1)
    MOVE_DOWN = 3  # Move south (y+1)
    MOVE_LEFT = 4  # Move west (x-1)
    SAP = 5  # Sap a target tile (requires x, y offsets)

    def to_direction(self):
        return {
            ActionType.MOVE_UP: (0, -1),
            ActionType.MOVE_RIGHT: (1, 0),
            ActionType.MOVE_DOWN: (0, 1),
            ActionType.MOVE_LEFT: (-1, 0),
            ActionType.STAY: (0, 0),
            ActionType.SAP: (0, 0),
        }[self]