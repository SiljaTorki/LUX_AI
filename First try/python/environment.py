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
    FOG_OF_WAR = True

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

    DEFAULT_UNIT_POS = -1 * np.ones((NUM_TEAMS, MAX_UNITS, 2), dtype=np.float32)
    DEFAULT_UNIT_ENERGY = -1 * np.ones((NUM_TEAMS, MAX_UNITS, 1), dtype=np.float32)
    DEFAULT_UNITS_MASK = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.float32)
    DEFAULT_SENSOR_MASK = np.zeros((MAP_WIDTH, MAP_HEIGHT), dtype=np.float32)
    DEFAULT_TILE_TYPE = -1 * np.ones((MAP_WIDTH, MAP_HEIGHT), dtype=np.float32)
    DEFAULT_MAP_FEATURES_ENERGY = -1 * np.ones(
        (MAP_WIDTH, MAP_HEIGHT), dtype=np.float32
    )
    DEFAULT_RELIC_NODES = -1 * np.ones((MAX_RELIC_NODES, 2), dtype=np.float32)
    DEFAULT_RELIC_MASK = np.zeros((MAX_RELIC_NODES,), dtype=np.float32)
    DEFAULT_TEAM_POINTS = np.zeros((NUM_TEAMS,), dtype=np.float32)
    DEFAULT_TEAM_WINS = np.zeros((NUM_TEAMS,), dtype=np.float32)
    DEFAULT_STEP_COUNT = np.array([0], dtype=np.float32)


class NodeType(Enum):
    """Represents the type of tiles on the map"""

    EMPTY = 0  # Default tile
    NEBULA = 1  # Vision and energy-reducing tile
    ASTEROID = 2  # Impassable terrain
    ENERGY_NODE = 3  # Provides energy (confirmed from game description)
    RELIC_NODE = 4  # Generates points when occupied


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


class Space:
    def __init__(self, width, height, map_data):
        self.width = width
        self.height = height
        self.grid = map_data
        self.obstacles = self.detect_obstacles()
        self.nebula_tiles = self.detect_nebula_tiles()

    def is_obstacle(self, x, y):
        """Check if a tile is an obstacle."""
        return self.grid[x, y] == NodeType.ASTEROID.value

    def is_within_bounds(self, x, y):
        """Ensure a position is inside the grid."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_neighbors(self, x, y):
        """Get valid movement options (avoid obstacles)."""
        neighbors = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # U, D, L, R
            nx, ny = x + dx, y + dy
            if self.is_within_bounds(nx, ny) and not self.is_obstacle(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def detect_obstacles(self):
        """Detect all obstacles in the map."""
        obstacles = []
        for x in range(self.width):
            for y in range(self.height):
                if self.is_obstacle(x, y):
                    obstacles.append((x, y))
        return obstacles

    def detect_nebula_tiles(self):
        """Detect all nebula tiles in the map."""
        nebula_tiles = []
        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x, y] == NodeType.NEBULA.value:
                    nebula_tiles.append((x, y))
        return nebula_tiles