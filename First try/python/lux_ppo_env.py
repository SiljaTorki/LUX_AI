import gym
from gym import spaces
import numpy as np
from a_star import improved_a_star_search
from agent import Ship
from environment import GameConstants, NodeType, ActionType
import random

NUM_TEAMS = 2
MAX_UNITS = GameConstants.MAX_UNITS
MAX_RELIC_NODES = GameConstants.MAX_RELIC_NODES
ACTION_COUNT = len(list(ActionType))


class PPOGameEnv(gym.Env):
    def __init__(self):
        super(PPOGameEnv, self).__init__()
        # self.action_space = spaces.Dict({
        #     "goal": spaces.Box(
        #         low=0, high=np.array([GameConstants.MAP_WIDTH-1, GameConstants.MAP_HEIGHT-1]),
        #         shape=(2,), dtype=np.int32
        #     ),
        #     "unit_roles": spaces.MultiDiscrete([4] * GameConstants.MAX_UNITS),  # 4 roles
        #     "strategy": spaces.Discrete(4),  # 4 strategy types
        # })

        # Create a flattened action space:
        # - First 2 values: goal coordinates (0-23 each)
        # - Next 1 value: strategy (0-3)
        # - Remaining 16 values: unit roles (0-3 each)
        # self.action_space = spaces.MultiDiscrete(
        #     [GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, 4] + [4] * GameConstants.MAX_UNITS
        # )
        # self.action_space = self.action_space = spaces.Box(
        #     low=0,
        #     # Subtract 1 to ensure we stay within bounds (0 to MAP_WIDTH-1)
        #     high=np.array([GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1]),
        #     shape=(2,),
        #     dtype=np.int32,
        # )
        self.action_space = spaces.MultiDiscrete(
            [
                GameConstants.MAP_WIDTH,  # x coordinate
                GameConstants.MAP_HEIGHT,  # y coordinate
            ]
        )
        self.observation_space = spaces.Dict(
            {
                "unit_position": spaces.Box(
                    low=0, high=1, shape=(NUM_TEAMS, MAX_UNITS, 2), dtype=np.float32
                ),
                "unit_energy": spaces.Box(
                    low=0, high=1, shape=(NUM_TEAMS, MAX_UNITS, 1), dtype=np.float32
                ),
                "units_mask": spaces.Box(
                    low=0, high=1, shape=(NUM_TEAMS, MAX_UNITS), dtype=np.float32
                ),
                "sensor_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.float32,
                ),
                "explored_tiles": spaces.Box(
                    low=0,
                    high=1,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.float32,
                ),
                "energy": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.float32,
                ),
                "map_features_tile_type": spaces.Box(
                    low=0,
                    high=2,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.float32,
                ),
                "map_features_energy": spaces.Box(
                    low=-1,
                    high=GameConstants.MAX_ENERGY_PER_TILE,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.float32,
                ),
                "map_symmetry_asteroid_symmetry": spaces.Box(
                    low=0, high=1, shape=(3,), dtype=np.float32
                ),
                "map_symmetry_nebula_symmetry": spaces.Box(
                    low=0, high=1, shape=(3,), dtype=np.float32
                ),
                "map_symmetry_energy_symmetry": spaces.Box(
                    low=0, high=1, shape=(3,), dtype=np.float32
                ),
                "relic_nodes_mask": spaces.Box(
                    low=0, high=1, shape=(MAX_RELIC_NODES,), dtype=np.float32
                ),
                "relic_nodes": spaces.Box(
                    low=-1,
                    high=GameConstants.MAP_HEIGHT - 1,
                    shape=(MAX_RELIC_NODES, 2),
                    dtype=np.float32,
                ),
                "team_points": spaces.Box(
                    low=0, high=10000, shape=(NUM_TEAMS,), dtype=np.float32
                ),
                "team_wins": spaces.Box(
                    low=0,
                    high=GameConstants.MATCH_COUNT_PER_EPISODE,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "match_steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "remainingOverageTime": spaces.Box(
                    low=0, high=1000, shape=(1,), dtype=np.float32
                ),
                "env_cfg_map_width": spaces.Box(
                    low=0, high=GameConstants.MAP_HEIGHT, shape=(1,), dtype=np.int32
                ),
                "env_cfg_map_height": spaces.Box(
                    low=0, high=GameConstants.MAP_HEIGHT, shape=(1,), dtype=np.int32
                ),
                "env_cfg_max_steps_in_match": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "env_cfg_unit_move_cost": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "env_cfg_unit_sap_cost": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "env_cfg_unit_sap_range": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "match_index": spaces.Box(
                    low=0, high=1, shape=(1, 1), dtype=np.float32
                ),
                "detected_parameters": spaces.Box(
                    low=0, high=1, shape=(1, 5), dtype=np.float32
                ),
                # Add memory channels
                "memory_map": spaces.Box(
                    low=-1,
                    high=2,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, 3),
                    dtype=np.float32,
                ),
            }
        )

        self.max_steps = GameConstants.MAX_STEPS_IN_MATCH
        self.current_step = 0
        self.score = 0

        self.tile_map = None
        self.relic_map = None
        self.energy_map = None
        self.visited = None

        self.team_units = []
        self.enemy_units = []
        self.team_spawn = (0, 0)
        self.enemy_spawn = (GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1)

        self.env_cfg = {
            "map_width": GameConstants.MAP_WIDTH,
            "map_height": GameConstants.MAP_HEIGHT,
            "max_steps_in_match": GameConstants.MAX_STEPS_IN_MATCH,
            "unit_move_cost": GameConstants.UNIT_MOVE_COST,
            "unit_sap_cost": (
                GameConstants.UNIT_SAP_COST
                if hasattr(GameConstants, "UNIT_SAP_COST")
                else 30
            ),
            "unit_sap_range": GameConstants.UNIT_SAP_RANGE,
        }

        self.relic_config = ([],)
        self.potential_visited = None
        self.team_points_space = None
        self._init_state()

    def set_match_sequence_mode(self, enabled=True):
        """
        Enable or disable 5-match sequence mode.
        When enabled, environment will run through 5 consecutive matches
        with the same map and parameters before resetting.
        """
        self.match_sequence_mode = enabled
        self.match_index = 0
        self.match_count = 5 if enabled else 1
        self.reset_between_matches = True

    def get_match_index(self):
        """Get the current match index (0-4) in the 5-match sequence"""
        if hasattr(self, "match_index"):
            return self.match_index
        return 0  # Default to first match

    def is_match_end(self):
        """Check if we're at the end of a match"""
        return self.current_step >= self.max_steps - 1

    def detect_randomized_parameters(self):
        """Detect and track randomized game parameters"""
        if not hasattr(self, "detected_parameters"):
            self.detected_parameters = {}

    def _init_state(self):
        num_tiles = GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
        self.tile_map = np.zeros((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT))

        num_nubla = int(num_tiles * 0.2)
        num_asteroid = int(num_tiles * 0.2)
        num_relics = max(1, int(GameConstants.RELIC_CONFIG_SIZE * 0.1))
        indices = np.random.choice(num_tiles, num_nubla + num_asteroid, replace=False)
        relic_indices = np.random.choice(num_tiles, num_relics, replace=False)
        # print(f"Attempting to place {num_relics} relics")

        flat_tiles = self.tile_map.flatten()
        flat_tiles[indices[:num_nubla]] = NodeType.NEBULA.value
        flat_tiles[indices[num_nubla:]] = NodeType.ASTEROID.value
        self.tile_map = flat_tiles.reshape(
            GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT
        )
        self.relic_map = np.zeros((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT))

        relic_indices = np.random.choice(
            num_tiles, GameConstants.RELIC_CONFIG_SIZE, replace=False
        )
        flat_relics = self.relic_map.flatten()
        flat_relics[relic_indices] = NodeType.RELIC_NODE.value
        self.relic_map = flat_relics.reshape(
            GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT
        )
        self.energy_map = np.zeros((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT))

        energy_indices = np.random.choice(num_tiles, 2, replace=False)
        flat_energy = self.energy_map.flatten()

        for i in energy_indices:
            flat_energy[i] = GameConstants.MAX_ENERGY_PER_TILE
            self.energy_map = flat_energy.reshape(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT)
            )

        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})

        self.visited = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        union_mask = self.get_global_sensor_mask()
        self.visited = union_mask.copy()

        self.score = 0

        self.relic_config = []
        relic_coords = np.argwhere(self.relic_map == NodeType.RELIC_NODE.value)

        for y, x in relic_coords:
            mask = np.zeros((5, 5), dtype=bool)
            indices = np.random.choice(25, 5, replace=False)
            mask_flat = mask.flatten()
            mask_flat[indices] = True
            mask = mask_flat.reshape((5, 5))
            self.relic_config.append((x, y, mask))

        self.potential_visited = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        self.team_points_space = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        self.current_step = 0
        
        if not hasattr(self, "memory_map"):
            self.memory_map = np.full((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, 3), -1, dtype=np.float32)
            
        if not hasattr(self, "discovered_relics"):
            self.discovered_relics = []
            
        if not hasattr(self, "scoring_tiles"):
            self.scoring_tiles = []
            
        if not hasattr(self, "energy_locations"):
            self.energy_locations = []
            
    
    def update_memory(self):
        """Update memory with current observations"""
        # Get current vision
        sensor_mask = self.get_global_sensor_mask()
        
        # Update memory map with observed tile types
        self.memory_map[:, :, 0] = np.where(
            sensor_mask, 
            self.tile_map, 
            self.memory_map[:, :, 0]
        )
        
        # Update memory map with observed energy
        for y in range(GameConstants.MAP_HEIGHT):
            for x in range(GameConstants.MAP_WIDTH):
                if sensor_mask[y, x] and self.energy_map[y, x] > 0:
                    self.memory_map[y, x, 1] = self.energy_map[y, x]
                    if (x, y) not in self.energy_locations:
                        self.energy_locations.append((x, y))
        
        # Detect and store relic nodes
        for y, x in np.argwhere(self.relic_map == NodeType.RELIC_NODE.value):
            if sensor_mask[y, x]:
                pos = (x, y)
                if pos not in self.discovered_relics:
                    self.discovered_relics.append(pos)
                    # Also store the mask for scoring tiles
                    for rx, ry, mask in self.relic_config:
                        if rx == x and ry == y:
                            for my in range(5):
                                for mx in range(5):
                                    if mask[my, mx]:
                                        score_x = rx - 2 + mx
                                        score_y = ry - 2 + my
                                        score_pos = (score_x, score_y)
                                        if score_pos not in self.scoring_tiles:
                                            self.scoring_tiles.append(score_pos)

    def compute_unit_vision(self, unit):
        sensor_range = 2
        nebula_reduction = 2
        vision = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        x, y = unit["x"], unit["y"]
        for dx in range(-sensor_range, sensor_range + 1):
            for dy in range(-sensor_range, sensor_range + 1):

                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= GameConstants.MAP_WIDTH:
                    continue

                if ny < 0 or ny >= GameConstants.MAP_HEIGHT:
                    continue

                if (
                    abs(dx) + abs(dy) <= sensor_range
                    and self.tile_map[nx, ny] != NodeType.ASTEROID.value
                ):
                    vision[nx, ny] = sensor_range + 1 - (abs(dx) + abs(dy))
                    if self.tile_map[nx, ny] == NodeType.NEBULA.value:
                        vision[nx, ny] -= nebula_reduction

        return vision > 0

    def get_global_sensor_mask(self):
        mask = np.zeros((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool)
        for unit in self.team_units:
            mask |= self.compute_unit_vision(unit)

        return mask

    def get_unit_observation(self, unit):
        sensor_mask = self.compute_unit_vision(unit)
        map_tile_type = np.where(sensor_mask, self.tile_map, NodeType.EMPTY.value)
        map_energy = np.where(sensor_mask, self.energy_map, -1)
        map_features = {"energy": map_energy, "tile_type": map_tile_type}
        sensor_mask_init = sensor_mask.astype(np.float32)

        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)
        for i, u in enumerate(self.team_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = u["energy"]

        units_mask[0, i] = 1
        for i, u in enumerate(self.enemy_units):
            ux, uy = u["x"], u["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = u["energy"]
                units_mask[1, i] = 1
        units = {"position": units_position, "energy": units_energy}

        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros(MAX_RELIC_NODES, dtype=np.int8)
        idx = 0
        for ry, rx in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1

        team_points = np.array([self.score, 0], dtype=np.int32)
        team_wins = np.array([0, 0], dtype=np.int32)
        steps = self.current_step
        match_steps = self.current_step

        obs = {
            "units": units,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_init,
            "map_features": map_features,
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "match_steps": match_steps,
        }

        observation = {
            "obs": obs,
            "remainingOverageTime": 0,
            "player": "player_0",
            "info": {"env_cfg": self.env_cfg},
        }

        return observation

    def get_obs(self):
        sensor_mask = self.get_global_sensor_mask()
        sensor_mask_int = sensor_mask.astype(np.int8)

        map_features_tile_type = np.where(sensor_mask, self.tile_map, -1)
        map_features_energy = np.where(sensor_mask, self.energy_map, -1)
        symmetry_features = detect_symmetry(self.tile_map)

        energy = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=np.float32
        )

        units_position = np.full((NUM_TEAMS, MAX_UNITS, 2), -1, dtype=np.int32)
        units_energy = np.full((NUM_TEAMS, MAX_UNITS, 1), -1, dtype=np.int32)
        units_mask = np.zeros((NUM_TEAMS, MAX_UNITS), dtype=np.int8)

        for i, unit in enumerate(self.team_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = unit["energy"]
                units_mask[0, i] = 1
        for i, unit in enumerate(self.enemy_units):
            ux, uy = unit["x"], unit["y"]
            if sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = unit["energy"]
                units_mask[1, i] = 1

        relic_coords = np.argwhere(self.relic_map == 1)
        relic_nodes = np.full((MAX_RELIC_NODES, 2), -1, dtype=np.int32)
        relic_nodes_mask = np.zeros((MAX_RELIC_NODES,), dtype=np.int8)
        idx = 0
        for ry, rx in relic_coords:
            if idx >= MAX_RELIC_NODES:
                break
            if sensor_mask[ry, rx]:
                relic_nodes[idx] = np.array([rx, ry])
                relic_nodes_mask[idx] = 1
            else:
                relic_nodes[idx] = np.array([-1, -1])
                relic_nodes_mask[idx] = 0
            idx += 1

        team_points = np.array([self.score], dtype=np.int32)
        team_wins = np.array([0], dtype=np.int32)
        steps = np.array([self.current_step], dtype=np.int32)
        match_steps = np.array([self.current_step], dtype=np.int32)
        remainingOverageTime = np.array([60], dtype=np.int32)

        # Get match information
        match_index = self.get_match_index() if hasattr(self, "get_match_index") else 0
        match_index_array = np.array([[match_index]], dtype=np.float32)

        if hasattr(self, "memory_map"):
            memory_map = self.memory_map.copy()
        else:
            memory_map = np.full(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, 3),
                -1,
                dtype=np.float32,
            )
            self.memory_map = memory_map.copy()

        memory_map[:, :, 0] = np.where(sensor_mask, self.tile_map, memory_map[:, :, 0])

        # Add discovered relic information to memory
        if hasattr(self, "discovered_relics"):
            for rx, ry in self.discovered_relics:
                if (
                    0 <= rx < GameConstants.MAP_WIDTH
                    and 0 <= ry < GameConstants.MAP_HEIGHT
                ):
                    memory_map[ry, rx, 1] = 1  # Mark as relic position

        # Track visited scoring tiles if available
        if hasattr(self, "scoring_tiles"):
            for tx, ty in self.scoring_tiles:
                if (
                    0 <= tx < GameConstants.MAP_WIDTH
                    and 0 <= ty < GameConstants.MAP_HEIGHT
                ):
                    memory_map[ty, tx, 2] = 1  # Mark as scoring tile

        # Track detected parameters
        if hasattr(self, "detected_parameters"):
            detected_params = np.array(
                [
                    self.detected_parameters.get("move_cost", 0)
                    / 5.0,  # Normalize to 0-1
                    self.detected_parameters.get("vision_range", 0) / 4.0,
                    self.detected_parameters.get("nebula_reduction", 0) / 5.0,
                    self.detected_parameters.get("sap_cost", 0) / 50.0,
                    self.detected_parameters.get("sap_range", 0) / 7.0,
                ],
                dtype=np.float32,
            ).reshape(1, 5)
        else:
            detected_params = np.zeros((1, 5), dtype=np.float32)

        env_cfg_map_width = np.array([self.env_cfg["map_width"]], dtype=np.int32)
        env_cfg_map_height = np.array([self.env_cfg["map_height"]], dtype=np.int32)
        env_cfg_max_steps_in_match = np.array(
            [self.env_cfg["max_steps_in_match"]], dtype=np.int32
        )
        env_cfg_unit_move_cost = np.array(
            [self.env_cfg["unit_move_cost"]], dtype=np.int32
        )
        env_cfg_unit_sap_cost = np.array(
            [self.env_cfg["unit_sap_cost"]], dtype=np.int32
        )
        env_cfg_unit_sap_range = np.array(
            [self.env_cfg["unit_sap_range"]], dtype=np.int32
        )
        explored_array = np.array(self.visited, dtype=np.float32)

        flat_obs = {
            "unit_position": units_position,
            "unit_energy": units_energy,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask_int,
            "map_features_tile_type": map_features_tile_type,
            "map_features_energy": map_features_energy,
            "map_symmetry_asteroid_symmetry": symmetry_features["asteroid_symmetry"],
            "map_symmetry_nebula_symmetry": symmetry_features["nebula_symmetry"],
            "map_symmetry_energy_symmetry": symmetry_features["energy_symmetry"],
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "explored_tiles": explored_array,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": steps,
            "energy": energy,
            "match_steps": match_steps,
            "remainingOverageTime": remainingOverageTime,
            "env_cfg_map_width": env_cfg_map_width,
            "env_cfg_map_height": env_cfg_map_height,
            "env_cfg_max_steps_in_match": env_cfg_max_steps_in_match,
            "env_cfg_unit_move_cost": env_cfg_unit_move_cost,
            "env_cfg_unit_sap_cost": env_cfg_unit_sap_cost,
            "env_cfg_unit_sap_range": env_cfg_unit_sap_range,
            "match_index": match_index_array,
            "memory_map": memory_map.reshape(
                1, GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, 3
            ),
            "detected_parameters": detected_params,
        }

        for key in list(flat_obs.keys()):
            if isinstance(flat_obs[key], np.ndarray):
                if flat_obs[key].ndim == 1:
                    flat_obs[key] = np.expand_dims(
                        flat_obs[key], axis=0
                    )  # Ensure at least 2D
                elif flat_obs[key].ndim == 2 and flat_obs[key].shape[0] != 1:
                    flat_obs[key] = np.expand_dims(flat_obs[key], axis=0)

        for key in list(flat_obs.keys()):
            if isinstance(flat_obs[key], dict):
                print(f"Warning: Nested dictionary found for key {key}")

        flat_obs = self.clean_observations(flat_obs)

        return flat_obs

    # def reset(self):
    #     self._init_state()
    #     return self.get_obs()
    def reset(self):
        if not hasattr(self, "match_memory"):
            # Initialize cross-match memory
            self.match_memory = {
                "energy_locations": set(),
                "relic_positions": set(),
                "scoring_tiles": set(),
                "dangerous_areas": set()
            }
        
        # Only do a full reset at the start of a 5-match sequence
        if self.match_index == 0:
            self._init_state()
        else:
            # Keep map and knowledge, but reset unit positions and energy
            self._reset_match_state()
            
            # Transfer knowledge from previous matches
            for energy_loc in self.match_memory["energy_locations"]:
                # Mark known energy locations in memory map
                x, y = energy_loc
                if 0 <= x < self.energy_map.shape[1] and 0 <= y < self.energy_map.shape[0]:
                    self.memory_map[y, x, 1] = 1
        
        # Initialize the new match
        self.current_step = 0
        return self.get_obs()

    
    def _reset_match_state(self):
        """Reset match state while preserving map knowledge"""
        # Reset units to starting positions
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
        
        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})
        
        # Reset score and step counter
        self.score = 0
        self.current_step = 0
        
        # Keep map and relic information (knowledge transfer)
        # But reset scoring tiles
        self.team_points_space = np.zeros((GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool)


    def seed(self, seed=None):
        """Seed the environment's random number generator"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        random.seed(seed)
        return [seed]

    def step(self, action):

        # if len(action) > 2:
        #     goal_x = int(action[0])
        #     goal_y = int(action[1])
        #     strategy = int(action[2])
        #     unit_roles = action[3:3+GameConstants.MAX_UNITS]
        #     goal_position = (goal_x, goal_y)
        # else:
        #     # Simple Box action space with just goal coordinates
        #     goal_x = int(action[0])
        #     goal_y = int(action[1])
        #     goal_position = (goal_x, goal_y)

        #     # Default values for strategy and roles
        #     strategy = 0  # Balanced strategy
        #     unit_roles = np.zeros(GameConstants.MAX_UNITS, dtype=np.int32)

        # goal_x = int(action[0])
        # goal_y = int(action[1])
        # goal_position = (goal_x, goal_y)
        # strategy = int(action[2])
        # unit_roles = action[3 : 3 + GameConstants.MAX_UNITS]
        # strategy = self.determine_strategy(goal_position)

        # # Generate unit roles based on position and game state
        # unit_roles = self.determine_unit_roles(goal_position)

        # Use defaults for other action components if using multi-indexed actions
        # if len(action) > 2:
        #     strategy = int(action[2])
        #     unit_roles = action[3:3+GameConstants.MAX_UNITS]
        # else:
        #     strategy = 0  # Balanced strategy
        #     unit_roles = np.zeros(GameConstants.MAX_UNITS, dtype=np.int32)
        goal_x = min(max(int(action[0]), 0), GameConstants.MAP_WIDTH - 1)
        goal_y = min(max(int(action[1]), 0), GameConstants.MAP_HEIGHT - 1)
        goal_position = (goal_x, goal_y)
        
        # Determine strategy and roles
        strategy = self.determine_strategy(goal_position)
        unit_roles = self.determine_unit_roles(goal_position)
        
        # Print for debugging
        print(f"Step {self.current_step}: Goal ({goal_x},{goal_y}), Strategy: {strategy}")
        # print(f"Unit roles: {unit_roles}")

        # print(
        #     f"Goal position: {goal_position}, Strategy: {strategy}, Roles: {unit_roles}"
        # )

        """Process agent actions and advance the environment by one step"""
        prev_score = self.score

        # Increment step counter
        self.current_step += 1
        match_progress = self.current_step / self.max_steps

        # Track state before actions for reward calculation
        prev_visited = (
            self.visited.copy()
            if hasattr(self, "visited")
            else None
        )

        # Process actions
        discrete_actions = self.generate_discrete_actions(goal_position)

        # Process each unit's action
        unit_rewards = self.process_unit_actions(discrete_actions, goal_position, unit_roles)
        movement_reward = sum(unit_rewards)

        # Process enemy movements
        self.process_enemy_moves()
        self.update_memory()

        # Calculate complex reward components
        exploration_reward = self.calculate_exploration_reward(prev_visited)
        relic_reward = self.calculate_relic_reward()
        energy_reward = self.calculate_energy_reward()
        score_reward = self.calculate_score_reward(prev_score)
        # coordination_reward = self.calculate_coordination_reward()

        # goal_quality_reward = self.calculate_goal_quality_reward(goal_position)
        # path_efficiency_reward = self.calculate_path_efficiency_reward(goal_position)
        # time_appropriate_reward = self.calculate_time_appropriate_reward(
        #     goal_position, match_progress
        # )

        # Calculate final reward with dynamic weights
        final_reward = self.combine_rewards(
            exploration_reward=exploration_reward,
            relic_reward=relic_reward,
            energy_reward=energy_reward,
            score_reward=score_reward,
            movement_reward=movement_reward,
            match_progress=match_progress,
            match_index=self.match_index,
        )

        # Check if episode is done
        done = self.current_step >= self.max_steps
        

        # Return observation, reward, done flag, and info
        info = {
            "score": self.score,
            "step": self.current_step,
            "match_index": self.match_index if hasattr(self, "match_index") else 0,
        }
        if self.current_step >= self.max_steps - 1:
            done = True
            
            # If we're in match sequence mode, prepare for next match
            if hasattr(self, "match_sequence_mode") and self.match_sequence_mode:
                self.current_match += 1
                self.matches_completed += 1
                
                # If we've completed all 5 matches, reset to match 0
                if self.current_match >= 5:
                    self.current_match = 0
                    
                # Add match transition info
                info["match_complete"] = True
                info["match_index"] = self.current_match
        return self.get_obs(), final_reward, done, info

    def determine_strategy(self, goal_position):
        """Determine appropriate strategy based on goal position and game state"""
        match_progress = self.current_step / self.max_steps
        match_index = self.get_match_index() if hasattr(self, "get_match_index") else 0
        if match_index == 0:
            if match_progress < 0.7:
                return 1  # Exploration
            else:
                return 0  # Balanced
        
        # Second match: Balance exploration/energy
        elif match_index == 1:
            if match_progress < 0.4:
                return 1  # Still some exploration
            else:
                return 2  # Energy focus
        
        # Later matches: Exploit knowledge
        else:
            if match_progress < 0.3:
                return 2  # Energy focus early
            else:
                return 3 
        # Get goal coordinates
        # goal_x, goal_y = goal_position
        
        # # Check goal position itself
        # if 0 <= goal_y < self.energy_map.shape[0] and 0 <= goal_x < self.energy_map.shape[1]:
        #     if self.energy_map[goal_y, goal_x] > 0:
        #         print(f"Goal ({goal_x},{goal_y}) is ON energy tile!")
        #         return 2  # Energy focus
        
        # # Check nearby tiles (within 2 distance)
        # for dx in range(-2, 3):
        #     for dy in range(-2, 3):
        #         check_x, check_y = goal_x + dx, goal_y + dy
        #         if (0 <= check_y < self.energy_map.shape[0] and 
        #             0 <= check_x < self.energy_map.shape[1] and
        #             self.energy_map[check_y, check_x] > 0):
        #             print(f"Energy tile found at ({check_x},{check_y}) near goal ({goal_x},{goal_y})")
        #             return 2  # Energy focus
        
        # # Check if goal is near a relic
        # relic_positions = [(rx, ry) for rx, ry, _ in self.relic_config]
        # # print(f"Relic positions: {relic_positions}")
        
        # for rx, ry, _ in self.relic_config:
        #     dist = abs(goal_x - rx) + abs(goal_y - ry)
        #     if dist <= 3:
        #         # print(f"Goal ({goal_x},{goal_y}) is near relic at ({rx},{ry}), distance: {dist}")
        #         return 3  # Relic focus
        
        # # Early match or early in sequence - exploration focus
        # if match_index < 2 or match_progress < 0.3:
        #     # print(f"Exploration focus - early game (progress: {match_progress:.2f}, match: {match_index})")
        #     return 1  # Exploration focus
        
        # # Default to balanced
        # # print("Strategy: Balanced (0)")
        # return 0

    def determine_unit_roles(self, goal_position):
        """Assign roles to units based on goal position and game state"""
        # Always initialize with zeros (not random values)
        unit_roles = np.zeros(GameConstants.MAX_UNITS, dtype=np.int32)
        
        # Print for debugging
        # print(f"Assigning roles for {len(self.team_units)} units with goal {goal_position}")
        strategy = self.determine_strategy(goal_position)
        # Assign roles based on units' positions and energy
        for i, unit in enumerate(self.team_units):
            if i >= GameConstants.MAX_UNITS:
                break
            
            # Debug info
            x, y = unit["x"], unit["y"]
            energy = unit["energy"]
            dist_to_goal = abs(x - goal_position[0]) + abs(y - goal_position[1])
            # print(f"Unit {i}: position ({x},{y}), energy {energy}")
            
            energy_threshold = 80  # Default
            if strategy == 2:  # Energy focus strategy
                energy_threshold = 150  # Higher threshold for energy strategy
            elif strategy == 3:  # Relic focus strategy
                energy_threshold = 60
            # Explorers: Units far from the goal
            if strategy == 1:  # Exploration focus
                # Prioritize exploration unless energy is very low
                if energy < 50:
                    unit_roles[i] = 1  # Energy collector
                    # print(f"  Role: Energy collector (1) - Very low energy")
                else:
                    unit_roles[i] = 0  # Explorer
                    # print(f"  Role: Explorer (0) - Exploration strategy")
                    
            elif strategy == 2:  # Energy focus
                # Prioritize energy collection
                if energy < energy_threshold:
                    unit_roles[i] = 1  # Energy collector
                    # print(f"  Role: Energy collector (1) - Energy strategy")
                else:
                    # If energy is good, explore
                    unit_roles[i] = 0  # Explorer
                    # print(f"  Role: Explorer (0) - Energy sufficient")
                    
            elif strategy == 3:  # Relic focus
                # Check if near relic
                near_relic = False
                for rx, ry, _ in self.relic_config:
                    if abs(x - rx) + abs(y - ry) < 8:
                        near_relic = True
                        break
                
                if near_relic and energy > energy_threshold:
                    unit_roles[i] = 2  # Relic seeker
                    # print(f"  Role: Relic seeker (2) - Near relic")
                elif energy < energy_threshold:
                    unit_roles[i] = 1  # Energy collector
                    # print(f"  Role: Energy collector (1) - Need energy for relic")
                else:
                    unit_roles[i] = 0  # Explorer
                    # print(f"  Role: Explorer (0) - Looking for relics")
                    
            else:  # Balanced strategy (0)
                # General purpose role assignment
                if energy < energy_threshold:
                    unit_roles[i] = 1  # Energy collector
                    # print(f"  Role: Energy collector (1) - Low energy")
                elif any(abs(x - rx) + abs(y - ry) < 8 for rx, ry, _ in self.relic_config):
                    unit_roles[i] = 2  # Relic seeker
                    # print(f"  Role: Relic seeker (2) - Near relic")
                elif any(abs(x - enemy["x"]) + abs(y - enemy["y"]) < 5 for enemy in self.enemy_units):
                    unit_roles[i] = 3  # Defender
                    # print(f"  Role: Defender (3) - Near enemy")
                else:
                    unit_roles[i] = 0  # Explorer by default
                    # print(f"  Role: Explorer (0) - Default")
        
        return unit_roles

    def clean_observations(self, obs_dict):
        """Remove infinities and NaNs from observations"""
        for key, value in obs_dict.items():
            if isinstance(value, np.ndarray):
                # Replace infinities with large values
                value = np.nan_to_num(value, nan=0.0, posinf=10.0, neginf=-10.0)
                obs_dict[key] = value
        return obs_dict

    def generate_discrete_actions(self, goal_position):
        """Generate discrete actions for each unit based on goal position"""
        discrete_actions = []

        for unit in self.team_units:
            x, y = unit["x"], unit["y"]
            goal_x, goal_y = goal_position

            # Determine direction to move
            if x < goal_x:
                action = ActionType.MOVE_RIGHT.value
            elif x > goal_x:
                action = ActionType.MOVE_LEFT.value
            elif y < goal_y:
                action = ActionType.MOVE_DOWN.value
            elif y > goal_y:
                action = ActionType.MOVE_UP.value
            else:
                action = ActionType.STAY.value  # Already at goal

            discrete_actions.append(action)

        return np.array(discrete_actions)

    # def process_unit_actions(self, discrete_actions, goal_position, unit_roles=None):
    #     """Process actions for each unit and return individual rewards"""
    #     unit_rewards = []
    #     successful_movements = 0
    #     successful_saps = 0
        
    #     # print(f"Processing actions for {len(self.team_units)} units with goal {goal_position}")
        
    #     for idx, unit in enumerate(self.team_units):
    #         if idx >= len(discrete_actions):
    #             unit_rewards.append(0)
    #             continue
                
    #         action_enum = ActionType(discrete_actions[idx])
    #         unit_reward = 0.0
            
    #         # Process role-based behavior first
    #         role = unit_roles[idx] if unit_roles is not None and idx < len(unit_roles) else 0
    #         # print(f"Unit {idx} has role {role}")

             
    #         # Calculate movement direction toward goal
    #         x, y = unit["x"], unit["y"]
            
    #         if role == 0:  # Explorer
    #         # Explorers prioritize moving toward goal or unexplored areas
    #             next_x, next_y = self.explorer_movement(unit, goal_position)
    #             # print(f"Explorer movement to ({next_x},{next_y})")
                
    #         elif role == 1:  # Energy collector
    #             # Energy collectors prioritize energy tiles
    #             next_x, next_y = self.energy_collector_movement(unit, goal_position)
    #             # print(f"Energy collector movement to ({next_x},{next_y})")
                
    #         elif role == 2:  # Relic seeker
    #             # Relic seekers prioritize relics or scoring tiles
    #             next_x, next_y = self.relic_seeker_movement(unit, goal_position)
    #             # print(f"Relic seeker movement to ({next_x},{next_y})")
                
    #         elif role == 3:  # Defender/Attacker
    #             # Defenders prioritize SAP actions or moving toward enemies
    #             if action_enum == ActionType.SAP:
    #                 sap_reward, success = self.process_sap_action(unit)
    #                 unit_reward += sap_reward
    #                 if success:
    #                     successful_saps += 1
    #                 next_x, next_y = x, y  # Stay in place if using SAP
    #                 # print(f"Defender using SAP, reward: {sap_reward}")
    #             else:
    #                 next_x, next_y = self.defender_movement(unit, goal_position)
    #                 # print(f"Defender movement to ({next_x},{next_y})")
    #         else:
    #             # Default movement toward goal
    #             next_x, next_y = self.default_movement(unit, goal_position)
    #             # print(f"Default movement to ({next_x},{next_y})")
            
    #         if "unit_move_cost" in self.env_cfg.keys():
    #             move_cost = self.env_cfg["unit_move_cost"]
    #         else:
    #             move_cost = GameConstants.UNIT_MOVE_COST
    #         if (next_x, next_y) != (x, y):
    #             if (0 <= next_y < self.tile_map.shape[0] and 
    #                 0 <= next_x < self.tile_map.shape[1] and
    #                 self.tile_map[next_y, next_x] != NodeType.ASTEROID.value):
                    
    #                 # Check energy
    #                 move_cost = self.env_cfg.get("unit_move_cost", 5)
    #                 if unit["energy"] >= move_cost:
    #                     # Execute movement
    #                     # print(f"  Moving unit {idx} from ({x},{y}) to ({next_x},{next_y})")
    #                     unit["x"], unit["y"] = next_x, next_y
    #                     unit["energy"] -= move_cost
    #                     successful_movements += 1
                        
    #                     # Role-specific movement rewards
    #                     if role == 0:  # Explorer
    #                         # Higher reward for exploring new tiles
    #                         if not self.visited[next_y, next_x]:
    #                             unit_reward += 0.4
    #                             self.visited[next_y, next_x] = True
    #                         else:
    #                             unit_reward += 0.1
    #                     elif role == 1:  # Energy collector
    #                         # Higher reward for moving to energy tiles
    #                         if self.energy_map[next_y, next_x] > 0:
    #                             unit_reward += 0.3
    #                         else:
    #                             unit_reward += 0.1
    #                     elif role == 2:  # Relic seeker
    #                         # Higher reward for approaching relics
    #                         near_relic = False
    #                         for rx, ry, _ in self.relic_config:
    #                             if abs(next_x - rx) + abs(next_y - ry) <= 5:
    #                                 near_relic = True
    #                                 break
    #                         unit_reward += 0.3 if near_relic else 0.1
    #                     elif role == 3:  # Defender
    #                         # Higher reward for approaching enemies
    #                         near_enemy = False
    #                         for enemy in self.enemy_units:
    #                             if abs(next_x - enemy["x"]) + abs(next_y - enemy["y"]) <= 3:
    #                                 near_enemy = True
    #                                 break
    #                         unit_reward += 0.3 if near_enemy else 0.1
    #                     else:
    #                         unit_reward += 0.1  # Default reward
    #                 else:
    #                     print(f"  Unit {idx} not enough energy: {unit['energy']} < {move_cost}")
    #             else:
    #                 print(f"  Unit {idx} cannot move to ({next_x},{next_y}) - invalid or asteroid")
            
    #         # Process energy collection at current position
    #         x, y = unit["x"], unit["y"]
    #         if self.energy_map[y, x] > 0:
    #             energy_gain = min(self.energy_map[y, x], 100)
    #             unit["energy"] += energy_gain
    #             self.energy_map[y, x] -= energy_gain
                
    #             # Higher reward for energy collectors
    #             if role == 1:
    #                 unit_reward += 0.8
    #             else:
    #                 unit_reward += 0.5
                    
    #             # print(f"  Unit {idx} collected {energy_gain} energy at ({x},{y})")
                
    #         # Process relic scoring (higher reward for relic seekers)
    #         relic_reward = self.check_relic_scoring(unit)
    #         if role == 2:  # Relic seeker
    #             relic_reward *= 1.5  # Bonus for relic seekers
    #         unit_reward += relic_reward
            
    #         unit_rewards.append(unit_reward)
        
    #     # Bonus for team coordination
    #     if successful_movements > 1:
    #         unit_rewards.append(0.1 * successful_movements)
    #     if successful_saps > 0:
    #         unit_rewards.append(0.2 * successful_saps)
        
    #     # print(f"Total rewards: {sum(unit_rewards)}")
    #     return unit_rewards
    def process_unit_actions(self, discrete_actions, goal_position, unit_roles=None):
        """Simplified unit action processing"""
        unit_rewards = []
        
        for idx, unit in enumerate(self.team_units):
            x, y = unit["x"], unit["y"]
            unit_reward = 0.0
            
            # Get best direction toward goal
            goal_x, goal_y = goal_position
            
            # Try all four directions
            directions = [(0,1), (1,0), (0,-1), (-1,0)]
            valid_moves = []
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < GameConstants.MAP_WIDTH and 
                    0 <= ny < GameConstants.MAP_HEIGHT and 
                    self.tile_map[ny, nx] != NodeType.ASTEROID.value):
                    
                    valid_moves.append((nx, ny, abs(nx - goal_x) + abs(ny - goal_y)))
            
            # Sort by distance to goal
            valid_moves.sort(key=lambda move: move[2])
            
            # Move if we have a valid move
            if valid_moves:
                next_x, next_y, _ = valid_moves[0]
                move_cost = 2  # Fixed move cost
                
                if unit["energy"] >= move_cost:
                    unit["x"], unit["y"] = next_x, next_y
                    unit["energy"] -= move_cost
                    
                    # Reward for movement (always positive)
                    unit_reward += 0.5
                    
                    # Extra reward if closer to goal
                    old_dist = abs(x - goal_x) + abs(y - goal_y)
                    new_dist = abs(next_x - goal_x) + abs(next_y - goal_y)
                    if new_dist < old_dist:
                        unit_reward += 0.3
                    
                    # Mark tile as visited
                    if not self.visited[next_y, next_x]:
                        self.visited[next_y, next_x] = True
                        unit_reward += 0.7
                    
                    print(f"Unit {idx} moved from ({x},{y}) to ({next_x},{next_y})")
                else:
                    print(f"Unit {idx} needs energy: {unit['energy']}/{move_cost}")
            else:
                print(f"Unit {idx} has no valid moves from ({x},{y})")
                unit_reward -= 0.2
            
            # Check for energy collection at new position
            x, y = unit["x"], unit["y"]
            if self.energy_map[y, x] > 0:
                energy_gain = min(self.energy_map[y, x], 50)
                unit["energy"] += energy_gain
                self.energy_map[y, x] -= energy_gain
                unit_reward += 2.0
                print(f"Unit {idx} collected {energy_gain} energy at ({x},{y})")
            
            self.visited[unit["y"], unit["x"]] = True
            # Check for relic scoring
            relic_reward = self.check_relic_scoring(unit)
            unit_reward += relic_reward * 5.0  # Strongly incentivize scoring
            
            unit_rewards.append(unit_reward)
        
        return unit_rewards
    def explorer_movement(self, unit, goal_position):
        """Improved explorer movement with obstacle avoidance"""
        x, y = unit["x"], unit["y"]
        
        # Get possible directions (no diagonals)
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        random.shuffle(directions)
        # Check all directions for unexplored tiles
        valid_moves = []
        unexplored_moves = []
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GameConstants.MAP_WIDTH and 
                0 <= ny < GameConstants.MAP_HEIGHT and
                self.tile_map[ny, nx] != NodeType.ASTEROID.value and
            not self.visited[ny, nx]):
                
                print(f"Found unexplored tile at ({nx},{ny})")
                return nx, ny
                # This is a valid move
                valid_moves.append((nx, ny))
                
                # Check if unexplored
                if hasattr(self, "visited") and not self.visited[ny, nx]:
                    unexplored_moves.append((nx, ny))
        
        # Prioritize unexplored tiles
        # if unexplored_moves:
        #     return unexplored_moves[0]
        
        # # If no unexplored tiles, move toward goal using valid moves
        # if valid_moves:
        #     # Find move that gets closest to goal
        #     goal_x, goal_y = goal_position
        #     best_move = min(
        #         valid_moves,
        #         key=lambda pos: abs(pos[0] - goal_x) + abs(pos[1] - goal_y)
        #     )
        #     return best_move
        goal_x, goal_y = goal_position
        best_direction = None
        best_distance = float('inf')
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GameConstants.MAP_WIDTH and 
                0 <= ny < GameConstants.MAP_HEIGHT and
                self.tile_map[ny, nx] != NodeType.ASTEROID.value):
                
                distance = abs(nx - goal_x) + abs(ny - goal_y)
                if distance < best_distance:
                    best_distance = distance
                    best_direction = (nx, ny)
        
        if best_direction:
            return best_direction
        
        # If no valid moves (trapped), stay in place
        return x, y

    def energy_collector_movement(self, unit, goal_position):
        """Improved energy collector movement with obstacle avoidance"""
        x, y = unit["x"], unit["y"]
        
        # Get possible directions
        directions = [(0,1), (1,0), (0,-1), (-1,0)]
        
        # Check all directions for energy tiles
        valid_moves = []
        energy_moves = []
        
        energy_scan_range = 5  # Look further for energy
        for scan_y in range(max(0, y-energy_scan_range), min(GameConstants.MAP_HEIGHT, y+energy_scan_range+1)):
            for scan_x in range(max(0, x-energy_scan_range), min(GameConstants.MAP_WIDTH, x+energy_scan_range+1)):
                if self.energy_map[scan_y, scan_x] > 20:  # Higher threshold for significant energy
                    # Path toward this energy source
                    if scan_x > x: return x+1, y
                    if scan_x < x: return x-1, y
                    if scan_y > y: return x, y+1
                    if scan_y < y: return x, y-1
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GameConstants.MAP_WIDTH and 
                0 <= ny < GameConstants.MAP_HEIGHT and
                self.tile_map[ny, nx] != NodeType.ASTEROID.value):
                
                # This is a valid move
                valid_moves.append((nx, ny))
                
                # Check if energy tile
                if self.energy_map[ny, nx] > 0:
                    energy_moves.append((nx, ny))
        
        # Prioritize energy tiles
        if energy_moves:
            return energy_moves[0]
            
        # If known energy locations, move toward closest
        if hasattr(self, "energy_locations") and self.energy_locations:
            # Filter to only include accessible energy locations
            closest_energy = min(
                self.energy_locations,
                key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y)
            )
            
            # Find move that gets closest to the energy
            if valid_moves:
                best_move = min(
                    valid_moves,
                    key=lambda pos: abs(pos[0] - closest_energy[0]) + abs(pos[1] - closest_energy[1])
                )
                return best_move
        
        # If no energy found or no valid moves toward energy, move toward goal using valid moves
        if valid_moves:
            goal_x, goal_y = goal_position
            best_move = min(
                valid_moves,
                key=lambda pos: abs(pos[0] - goal_x) + abs(pos[1] - goal_y)
            )
            return best_move
        
        # If no valid moves (trapped), stay in place
        return x, y

    def relic_seeker_movement(self, unit, goal_position):
        """Movement logic for relic seekers - prioritize relics and scoring tiles"""
        x, y = unit["x"], unit["y"]
        
        # Check if we're near a relic node
        for rx, ry, mask in self.relic_config:
            # If near a relic, check for scoring tiles
            if abs(x - rx) <= 5 and abs(y - ry) <= 5:
                # Look for scoring tiles in the mask
                for mx in range(5):
                    for my in range(5):
                        if mask[my, mx]:
                            tx = rx - 2 + mx
                            ty = ry - 2 + my
                            # If this is a valid scoring tile
                            if (0 <= tx < GameConstants.MAP_WIDTH and 
                                0 <= ty < GameConstants.MAP_HEIGHT and
                                not self.team_points_space[ty, tx]):
                                # Move toward scoring tile
                                if x < tx:
                                    return x + 1, y
                                elif x > tx:
                                    return x - 1, y
                                elif y < ty:
                                    return x, y + 1
                                elif y > ty:
                                    return x, y - 1
                                return x, y  # Already on the tile
        
        # If not near a relic but we know relic locations
        if hasattr(self, "discovered_relics") and self.discovered_relics:
            closest_relic = min(
                self.discovered_relics,
                key=lambda pos: abs(pos[0] - x) + abs(pos[1] - y)
            )
            # Move toward closest relic
            rx, ry = closest_relic
            if x < rx:
                return x + 1, y
            elif x > rx:
                return x - 1, y
            elif y < ry:
                return x, y + 1
            elif y > ry:
                return x, y - 1
        
        # If no relics found, move toward goal
        return self.default_movement(unit, goal_position)

    def defender_movement(self, unit, goal_position):
        """Movement logic for defenders - prioritize moving toward enemies"""
        x, y = unit["x"], unit["y"]
        
        # Look for nearby enemies
        if self.enemy_units:
            closest_enemy = min(
                self.enemy_units,
                key=lambda e: abs(e["x"] - x) + abs(e["y"] - y)
            )
            
            # Move toward closest enemy
            ex, ey = closest_enemy["x"], closest_enemy["y"]
            if abs(ex - x) + abs(ey - y) <= 5:  # If enemy is nearby
                if x < ex:
                    return x + 1, y
                elif x > ex:
                    return x - 1, y
                elif y < ey:
                    return x, y + 1
                elif y > ey:
                    return x, y - 1
        
        # If no enemies nearby, move toward goal
        return self.default_movement(unit, goal_position)

    def default_movement(self, unit, goal_position):
        """Default movement logic toward goal"""
        x, y = unit["x"], unit["y"]
        goal_x, goal_y = goal_position
        
        # Determine direction toward goal
        if x < goal_x:
            return x + 1, y
        elif x > goal_x:
            return x - 1, y
        elif y < goal_y:
            return x, y + 1
        elif y > goal_y:
            return x, y - 1
        else:
            return x, y  # Already at goal

    # def process_unit_actions(self, discrete_actions, goal_position, unit_roles=None):
    #     """Process actions for each unit and return individual rewards"""
    #     unit_rewards = []
    #     successful_movements = 0
    #     successful_saps = 0

    #     for idx, unit in enumerate(self.team_units):
    #         if idx >= len(discrete_actions):
    #             unit_rewards.append(0)
    #             continue

    #         action_enum = ActionType(discrete_actions[idx])
    #         unit_reward = 0.0

    #         role = (
    #             unit_roles[idx]
    #             if unit_roles is not None and idx < len(unit_roles)
    #             else None
    #         )

    #         # Process relic scoring
    #         unit_reward += self.check_relic_scoring(unit)

    #         if role is not None:
    #             # Role 0: Explorer
    #             if role == 0:
    #                 # Explorers get extra reward for finding new tiles
    #                 if not hasattr(self, "unit_previously_visited"):
    #                     self.unit_previously_visited = {}

    #                 unit_id = idx
    #                 if unit_id not in self.unit_previously_visited:
    #                     self.unit_previously_visited[unit_id] = set()

    #                 x, y = unit["x"], unit["y"]
    #                 pos = (x, y)

    #                 if pos not in self.unit_previously_visited[unit_id]:
    #                     unit_reward += 0.3  # Bonus for exploring new tile
    #                     self.unit_previously_visited[unit_id].add(pos)

    #             # Role 1: Energy Collector
    #             elif role == 1:
    #                 # Energy collectors get extra reward for being on energy tiles
    #                 x, y = unit["x"], unit["y"]
    #                 if self.energy_map[y, x] > 0:
    #                     unit_reward += 0.5  # Bonus for collecting energy

    #             # Role 2: Relic Seeker
    #             elif role == 2:
    #                 # Relic seekers get extra reward for being near relics
    #                 x, y = unit["x"], unit["y"]
    #                 for rx, ry, _ in self.relic_config:
    #                     dist = abs(x - rx) + abs(y - ry)
    #                     if dist <= 3:  # Within 3 tiles of a relic
    #                         unit_reward += 0.4 * (
    #                             1.0 - dist / 4
    #                         )  # More reward for closer proximity

    #             # Role 3: Defender/Attacker
    #             elif role == 3:
    #                 # Attackers get extra reward for successful SAP actions
    #                 if action_enum == ActionType.SAP:
    #                     unit_reward += 0.3  # Bonus for offensive action

    #                 # Process SAP action
    #                 if action_enum == ActionType.SAP:
    #                     sap_reward, was_successful = self.process_sap_action(unit)
    #                     unit_reward += sap_reward
    #                     if was_successful:
    #                         successful_saps += 1
    #                 # Process movement actions
    #                 else:
    #                     move_reward, was_successful = self.process_movement_action(
    #                         idx, unit, goal_position
    #                     )
    #                     unit_reward += move_reward
    #                     if was_successful:
    #                         successful_movements += 1

    #                 unit_rewards.append(unit_reward)

    #             # Bonus for team coordination
    #             if successful_movements > 1:
    #                 unit_rewards.append(0.1 * successful_movements)
    #             if successful_saps > 0:
    #                 unit_rewards.append(0.2 * successful_saps)

    #             return unit_rewards
    #         # Process SAP action
    #         if action_enum == ActionType.SAP:
    #             sap_reward, was_successful = self.process_sap_action(unit)
    #             unit_reward += sap_reward
    #             if was_successful:
    #                 successful_saps += 1
    #         # Process movement actions
    #         else:
    #             move_reward, was_successful = self.process_movement_action(
    #                 idx, unit, goal_position
    #             )
    #             unit_reward += move_reward
    #             if was_successful:
    #                 successful_movements += 1

    #         unit_rewards.append(unit_reward)
    #     if successful_movements > 1:
    #         unit_rewards.append(0.1 * successful_movements)
    #     if successful_saps > 0:
    #         unit_rewards.append(0.2 * successful_saps)

    #     return unit_rewards

    def calculate_path_efficiency_reward(self, goal_position):
        """Reward for choosing a goal that can be efficiently reached"""
        reward = 0

        # Calculate average path length for units to reach the goal
        total_path_length = 0
        valid_paths = 0

        for unit in self.team_units:
            start_pos = (unit["x"], unit["y"])
            if start_pos == goal_position:
                # Already at goal
                total_path_length += 0
                valid_paths += 1
                continue

            # Try to find a path
            try:
                # Create a Ship object for pathfinding
                ship = Ship(0, start_pos, unit["energy"])
                enemy_positions = [(e["x"], e["y"]) for e in self.enemy_units]

                path = improved_a_star_search(
                    ship=ship,
                    goal=goal_position,
                    map_features=self.tile_map,
                    team_vision=self.get_global_sensor_mask(),
                    enemy_positions=enemy_positions,
                    step_count=self.current_step,
                )

                if path and len(path) > 1:
                    total_path_length += len(path) - 1
                    valid_paths += 1
            except:
                # If pathfinding fails, ignore this unit
                pass

        if valid_paths > 0:
            avg_path_length = total_path_length / valid_paths

            # Shorter paths are better
            max_possible_length = GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT
            normalized_length = min(avg_path_length / max_possible_length, 1.0)

            # Reward is higher for shorter paths
            reward = 1.0 - normalized_length
        else:
            # No valid paths found
            reward = -0.5

        return reward

    def calculate_time_appropriate_reward(self, goal_position, match_progress):
        """Reward for choosing a goal appropriate for the current match phase"""
        reward = 0

        # Early match (first 30%)
        if match_progress < 0.3:
            # In early match, reward exploration and energy collection

            # Reward unexplored tiles
            if (
                hasattr(self, "visited")
                and not self.visited[goal_position[1], goal_position[0]]
            ):
                reward += 1.0

            # Reward energy tiles
            if self.energy_map[goal_position[1], goal_position[0]] > 0:
                reward += 0.8

        # Mid match (30% to 70%)
        elif match_progress < 0.7:
            # In mid match, balance between energy and relics

            # Check if goal is related to relics
            relic_related = False
            for rx, ry, _ in self.relic_config:
                if abs(goal_position[0] - rx) <= 3 and abs(goal_position[1] - ry) <= 3:
                    relic_related = True
                    reward += 0.7
                    break

            # If not relic related, reward energy
            if (
                not relic_related
                and self.energy_map[goal_position[1], goal_position[0]] > 0
            ):
                reward += 0.5

        # Late match (last 30%)
        else:
            # In late match, prioritize relics and scoring

            # High reward for relic scoring tiles
            for rx, ry, mask in self.relic_config:
                # If goal is in scoring area
                if abs(goal_position[0] - rx) <= 2 and abs(goal_position[1] - ry) <= 2:
                    mask_x = goal_position[0] - rx + 2
                    mask_y = goal_position[1] - ry + 2
                    if 0 <= mask_x < 5 and 0 <= mask_y < 5 and mask[mask_y, mask_x]:
                        reward += 2.0
                        break

            # Low reward for energy tiles at this point
            if self.energy_map[goal_position[1], goal_position[0]] > 0:
                reward += 0.2

        return reward

    def calculate_goal_quality_reward(self, goal_position):
        """Calculate reward based on the quality of the selected goal"""
        reward = 0

        # Check if goal is valid
        if not (
            0 <= goal_position[0] < GameConstants.MAP_WIDTH
            and 0 <= goal_position[1] < GameConstants.MAP_HEIGHT
        ):
            return -1.0  # Invalid goal position

        # Penalty for selecting an asteroid tile as goal
        if self.tile_map[goal_position[1], goal_position[0]] == NodeType.ASTEROID.value:
            return -0.5

        # Reward for selecting energy tiles
        if self.energy_map[goal_position[1], goal_position[0]] > 0:
            # Higher reward if team has low energy
            avg_energy = np.mean([unit["energy"] for unit in self.team_units])
            if avg_energy < 100:
                reward += 1.0
            else:
                reward += 0.3

        # Reward for selecting relic nodes or scoring tiles
        for rx, ry, mask in self.relic_config:
            # If goal is the relic node itself
            if goal_position[0] == rx and goal_position[1] == ry:
                reward += 1.0
                break

            # If goal is in scoring area
            if abs(goal_position[0] - rx) <= 2 and abs(goal_position[1] - ry) <= 2:
                # Check if it's a scoring tile
                mask_x = goal_position[0] - rx + 2
                mask_y = goal_position[1] - ry + 2
                if 0 <= mask_x < 5 and 0 <= mask_y < 5 and mask[mask_y, mask_x]:
                    reward += 2.0
                    break

        # Reward for unexplored tiles if it's early in the match
        if self.current_step < self.max_steps / 2:
            if (
                hasattr(self, "visited")
                and not self.visited[goal_position[1], goal_position[0]]
            ):
                reward += 0.5

        return reward

    def check_relic_scoring(self, unit):
        """Check if unit is on relic scoring tile and update score"""
        x, y = unit["x"], unit["y"]
        reward = 0

        for rx, ry, mask in self.relic_config:
            # If unit is within the 5x5 area around a relic
            if abs(x - rx) <= 2 and abs(y - ry) <= 2:
                # Calculate position in mask
                mx, my = x - rx + 2, y - ry + 2

                # Check if this is a scoring tile
                if 0 <= mx < 5 and 0 <= my < 5 and mask[my, mx]:
                    # If not already scored this tile
                    if not self.team_points_space[y, x]:
                        self.score += 1
                        reward += 3.0
                        self.team_points_space[y, x] = True
                    else:
                        # Reward for continuing to occupy scoring tile
                        reward += 1.0
                # If in potential area but not scoring tile
                elif not self.potential_visited[y, x]:
                    reward += 0.5
                    self.potential_visited[y, x] = True

        return reward

    def apply_strategy(self, strategy, rewards):
        """Modify rewards based on selected strategy"""
        # Strategy 0: Balanced
        if strategy == 0:
            # No modifications to rewards
            return rewards

        # Strategy 1: Exploration Focus
        elif strategy == 1:
            rewards["exploration_reward"] *= 1.5
            rewards["relic_reward"] *= 0.75

        # Strategy 2: Energy Focus
        elif strategy == 2:
            rewards["energy_reward"] *= 2.0
            rewards["exploration_reward"] *= 0.5

        # Strategy 3: Relic Focus
        elif strategy == 3:
            rewards["relic_reward"] *= 2.0
            rewards["coordination_reward"] *= 1.5
            rewards["exploration_reward"] *= 0.5

        return rewards

    def process_sap_action(self, unit):
        """Process SAP action for a unit"""
        x, y = unit["x"], unit["y"]
        unit_obs = self.get_unit_observation(unit)

        # Check if on relic node
        on_relic = np.any(unit_obs["obs"]["relic_nodes_mask"] == 1)

        # Count nearby enemies
        nearby_enemies = sum(
            1
            for enemy in self.enemy_units
            if abs(enemy["x"] - x) + abs(enemy["y"] - y) <= 1
        )

        # Calculate reward based on situation
        if on_relic:
            if nearby_enemies > 0:
                # Defending relic is valuable
                return 2.0 * nearby_enemies, True
            else:
                # Penalty for incorrect SAP when on relic
                return -0.5, False
        else:
            if nearby_enemies > 0:
                # Reward for attacking enemies
                return 1.0 * nearby_enemies, True
            else:
                # Penalty for wasting energy
                return -1.0, False
            
    def process_movement_action(self, idx, unit, goal_position):
        """Process movement action for a unit"""
        x, y = unit["x"], unit["y"]
        unit_ship = Ship(idx, (x, y), unit["energy"])

        # Get path to goal
        enemy_positions = [(enemy["x"], enemy["y"]) for enemy in self.enemy_units]
        path = improved_a_star_search(
            unit_ship,
            goal_position,
            self.tile_map,
            self.get_global_sensor_mask(),
            enemy_positions,
            step_count=self.current_step,
        )

        # Check if we have a valid path
        if path is None or len(path) <= 1:
            print(f"No valid path from ({x},{y}) to goal {goal_position}")
            return -0.1, False
        
        # Execute movement
        next_x, next_y = path[1]
        prev_x, prev_y = unit["x"], unit["y"]
        
        # Debug movement
        # print(f"Moving unit from ({prev_x},{prev_y}) to ({next_x},{next_y})")
        
        # Check if movement is valid (not trying to move into an asteroid)
        if self.tile_map[next_y, next_x] == NodeType.ASTEROID.value:
            # print(f"Cannot move to ({next_x},{next_y}) - asteroid")
            return -0.1, False
        
        # Check energy cost
        move_cost = self.env_cfg.get("unit_move_cost", 1)
        if unit["energy"] < move_cost:
            # print(f"Not enough energy to move: {unit['energy']} < {move_cost}")
            return -0.1, False
            
        # Update position and reduce energy
        unit["x"], unit["y"] = next_x, next_y
        unit["energy"] -= move_cost
        
        # Calculate movement reward
        reward = 0.1  # Base reward
        return reward, True

    # def process_movement_action(self, idx, unit, goal_position):
    #     """Process movement action for a unit"""
    #     x, y = unit["x"], unit["y"]
    #     unit_ship = Ship(idx, (x, y), unit["energy"])

    #     # Get goal position
    #     goal = self.determine_unit_goals(goal_position=goal_position, ship=unit_ship)

    #     # Calculate path
    #     enemy_positions = [(enemy["x"], enemy["y"]) for enemy in self.enemy_units]
    #     path = improved_a_star_search(
    #         unit_ship,
    #         goal,
    #         self.tile_map,
    #         self.get_global_sensor_mask(),
    #         enemy_positions,
    #         step_count=self.current_step,
    #     )

    #     # Execute movement if path exists
    #     if len(path) > 1:
    #         next_x, next_y = path[1]
    #         prev_x, prev_y = unit["x"], unit["y"]
    #         unit["x"], unit["y"] = next_x, next_y

    #         # Calculate movement reward
    #         reward = 0.1  # Base reward

    #         # Reward for moving toward goal
    #         dist_to_goal = abs(next_x - goal[0]) + abs(next_y - goal[1])
    #         prev_dist = abs(prev_x - goal[0]) + abs(prev_y - goal[1])
    #         if dist_to_goal < prev_dist:
    #             reward += 0.1

    #         # Reward for moving to energy when low
    #         if unit["energy"] < 100 and self.energy_map[next_y, next_x] > 0:
    #             reward += 0.3

    #         return reward, True
    #     else:
    #         # Penalty based on whether already at goal
    #         if (x, y) == goal:
    #             return -0.05, False  # Smaller penalty
    #         else:
    #             return -0.1, False  # Larger penalty

    def calculate_exploration_reward(self, prev_visited):
        """Enhanced exploration reward with match-specific scaling"""
        match_index = self.get_match_index() if hasattr(self, "get_match_index") else 0
        
        # Scaling factor decreases with match index (prioritize exploration in early matches)
        exploration_scale = max(1.5 - match_index * 0.3, 0.3)
        
        # Count newly explored tiles
        current_vision = self.get_global_sensor_mask()
        newly_explored = np.logical_and(current_vision, np.logical_not(self.visited))
        exploration_count = np.sum(newly_explored)
        
        # Higher reward for early match exploration
        if match_index < 2:
            reward = exploration_scale * exploration_count * 2.0
        else:
            reward = exploration_scale * exploration_count * 0.5
        
        # Update visited mask
        self.visited = np.logical_or(self.visited, current_vision)
        
        return reward
    # def calculate_exploration_reward(self, prev_visited):
    #     """Calculate reward for exploring new areas"""
    #     # Initialize previously_visited if needed
    #     if not hasattr(self, "previously_visited"):
    #         self.previously_visited = np.zeros(
    #             (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
    #         )
    #         self.exploration_progress = 0.0

    #     # Get current vision and count newly explored tiles
    #     current_vision = self.get_global_sensor_mask()
    #     newly_explored = np.logical_and(
    #         current_vision, np.logical_not(self.previously_visited)
    #     )
    #     exploration_count = np.sum(newly_explored)

    #     # Update visited tiles
    #     self.previously_visited = np.logical_or(self.previously_visited, current_vision)

    #     total_tiles = GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
    #     current_coverage = np.sum(self.previously_visited) / total_tiles

    #     if current_coverage > self.exploration_progress:
    #         coverage_increase = current_coverage - self.exploration_progress
    #         self.exploration_progress = current_coverage
    #         coverage_reward = (
    #             10.0 * coverage_increase * total_tiles
    #         )  # Scale by map size
    #     else:
    #         coverage_reward = 0.0

    #     # Return reward proportional to new tiles
    #     return 1.0 * exploration_count + coverage_reward

    def calculate_relic_reward(self):
        """Progressive relic rewards based on proximity and occupation"""
        units_on_relics = 0
        proximity_reward = 0

        # Track closest approach to relics
        if not hasattr(self, "closest_relic_approach"):
            self.closest_relic_approach = (
                {}
            )  # Maps unit_id to closest distance achieved

        for i, unit in enumerate(self.team_units):
            unit_pos = (unit["x"], unit["y"])

            # Check if on relic
            if self.is_on_relic(unit_pos):
                units_on_relics += 1
                continue

            # Calculate closest distance to any relic
            min_dist = float("inf")
            for y, x in np.argwhere(self.relic_map == NodeType.RELIC_NODE.value):
                # relic_pos = (x, y)
                # dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
                dist = abs(unit_pos[0] - x) + abs(unit_pos[1] - y)
                min_dist = min(min_dist, dist)

            prev_closest = self.closest_relic_approach.get(i, float("inf"))
            # If this is closer than before, reward the progress
            if min_dist < prev_closest:
                proximity_reward += 0.3 * (
                    prev_closest - min_dist
                )  # Reward for getting closer
                self.closest_relic_approach[i] = min_dist
        # reward = 10.0 * units_on_relics  # Your original calculation

        # # Clip to prevent extreme values
        # reward = np.clip(reward, -10.0, 10.0)
        reward = 5.0 * units_on_relics + proximity_reward
        # # Ensure valid value
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        # Combine direct occupation and proximity rewards
        return reward

    def calculate_energy_reward(self):
        """Calculate reward for energy management"""
        total_reward = 0
        game_progress = self.current_step / self.max_steps

        # Determine energy importance based on game phase
        # Early game: Energy is critical for exploration
        # Mid game: Energy should be maintained for operations
        # Late game: Energy is only relevant for reaching relics and tiebreakers
        energy_importance = max(1.0 - game_progress * 1.5, 0.1)

        for unit in self.team_units:
            energy = unit["energy"]

            # Energy efficiency curve - rewards maintaining energy between 100-300
            # Penalizes both very low and very high energy (wasting collection opportunities)
            if energy < 50:
                # Critical energy - significant penalty
                energy_reward = -0.05 * (50 - energy)
            elif energy < 100:
                # Low energy - mild penalty
                energy_reward = -0.05 * (100 - energy)
            elif energy <= 300:
                # Optimal energy range - positive reward
                energy_reward = 0.1 * energy / 300
            else:
                # Excessive energy - diminishing returns
                energy_reward = 0.3 - 0.1 * (energy - 300) / 300

            total_reward += energy_reward

        # Reward energy collection actions explicitly
        for unit in self.team_units:
            unit_pos = (unit["x"], unit["y"])
            if self.energy_map[unit_pos[1], unit_pos[0]] > 0:
                total_reward += 0.5  # Direct reward for being on energy tiles

        # Reward energy collection actions - important in early/mid game, less in late game
        for unit in self.team_units:
            unit_pos = (unit["x"], unit["y"])
            if self.energy_map[unit_pos[1], unit_pos[0]] > 0:
                # Only reward energy collection if unit has room for more energy
                if unit["energy"] < 300:
                    total_reward += 0.3 * energy_importance

        return np.clip(total_reward, -5, 5)

    def calculate_score_reward(self, prev_score):
        """Calculate reward for score increases"""
        score_increase = self.score - prev_score
        return 5.0 * score_increase

    def calculate_memory_reward(self):
        """Reward the agent for building and using memory across matches"""
        reward = 0

        # Track newly discovered information
        if not hasattr(self, "previously_discovered_tiles"):
            self.previously_discovered_tiles = np.zeros(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
            )
            self.previously_discovered_relics = set()
            self.previously_discovered_energy = set()

        # Get current vision and count newly discovered features
        visible_tiles = self.get_global_sensor_mask()

        # Count newly discovered tiles
        newly_discovered = np.logical_and(
            visible_tiles, ~self.previously_discovered_tiles
        )
        new_tile_count = np.sum(newly_discovered)

        # Reward based on match index
        match_index = self.get_match_index()

        if match_index < 2:  # Early matches - reward exploration
            reward += 0.2 * new_tile_count
        else:  # Later matches - penalize unnecessary exploration
            reward -= 0.05 * new_tile_count

        # Update memory
        self.previously_discovered_tiles = np.logical_or(
            self.previously_discovered_tiles, visible_tiles
        )

        return reward

    def calculate_exploitation_reward(self):
        """Reward the agent for increasingly effective exploitation in later matches"""
        reward = 0
        match_index = self.get_match_index()

        # Calculate score per step (efficiency metric)
        if not hasattr(self, "previous_score"):
            self.previous_score = 0

        score_increase = self.score - self.previous_score
        self.previous_score = self.score

        # Scale reward by match index - exploitation should improve in later matches
        exploitation_factor = 1.0 + (match_index / 2.0)  # Ranges from 1.0 to 3.0

        reward += score_increase * exploitation_factor

        # Add penalty for wasted steps in later matches
        if match_index >= 3 and score_increase == 0:
            reward -= 0.1 * match_index

        return reward

    def calculate_knowledge_transfer_reward(self):
        """Reward the agent for effectively transferring knowledge between matches"""
        reward = 0
        match_index = self.get_match_index()

        if match_index > 0:  # Not relevant for first match
            # Calculate how quickly agent reaches known objectives

            # Check time to reach first known relic node
            if hasattr(self, "known_relic_nodes") and self.known_relic_nodes:
                if not hasattr(self, "relic_discovery_step"):
                    self.relic_discovery_step = {}

                for relic_pos in self.known_relic_nodes:
                    # If we've reached a previously known relic
                    unit_positions = [
                        (unit["x"], unit["y"]) for unit in self.team_units
                    ]
                    if (
                        any(pos == relic_pos for pos in unit_positions)
                        and relic_pos not in self.relic_discovery_step
                    ):
                        self.relic_discovery_step[relic_pos] = self.current_step
                        # Reward for quick discovery (earlier is better)
                        time_factor = max(
                            0, 1.0 - (self.current_step / 50)
                        )  # 0 to 1 scale
                        reward += 5.0 * time_factor * match_index

        return reward

    def calculate_parameter_learning_reward(self):
        """Reward the agent for adapting to randomized game parameters"""
        reward = 0

        # Only calculate after we've had chance to detect parameters
        if hasattr(self, "detected_parameters"):
            match_index = self.get_match_index()

            # Track energy utilization relative to move cost
            if "move_cost" in self.detected_parameters:
                move_cost = self.detected_parameters["move_cost"]

                # Efficient energy use given the move cost
                avg_energy = sum(unit["energy"] for unit in self.team_units) / len(
                    self.team_units
                )

                # For high move costs, reward maintaining higher energy levels
                if move_cost > 3:  # High move cost
                    energy_efficiency = min(avg_energy / 200, 1.0)  # Scale from 0-1
                    reward += 0.5 * energy_efficiency * match_index
                else:  # Low move cost
                    # Reward aggressive action when move cost is low
                    energy_efficiency = min(
                        max(avg_energy / 100, 0.2), 0.8
                    )  # Keep some energy, but not too much
                    reward += 0.3 * energy_efficiency * match_index

        return reward

    def calculate_temporal_evolution_reward(self):
        """Reward increasing score efficiency over the match sequence"""
        reward = 0
        match_index = self.get_match_index()

        # Skip if not tracking previous matches
        if not hasattr(self, "match_scores"):
            self.match_scores = []
            self.match_step_ratios = []
            return 0

        # Current efficiency
        step_ratio = self.current_step / self.max_steps
        current_score = self.score

        # If we're at match end, compare to previous matches
        if self.is_match_end():
            self.match_scores.append(current_score)
            self.match_step_ratios.append(step_ratio)

            if len(self.match_scores) > 1:
                prev_score = self.match_scores[-2]
                prev_step_ratio = self.match_step_ratios[-2]

                # Calculate efficiency improvement
                current_rate = current_score / step_ratio if step_ratio > 0 else 0
                previous_rate = (
                    prev_score / prev_step_ratio if prev_step_ratio > 0 else 0
                )

                improvement = current_rate - previous_rate

                # Reward improvement, with higher expectations in later matches
                expected_improvement = 2.0 * (
                    match_index - 1
                )  # Higher expectations in later matches
                reward += max(0, improvement - expected_improvement) * match_index

        return reward

    def calculate_coordination_reward(self):
        """Reward team coordination and strategic positioning"""
        reward = 0

        # Get unit positions
        unit_positions = [(unit["x"], unit["y"]) for unit in self.team_units]

        # Reward for good map coverage (units spread out)
        if len(unit_positions) >= 2:
            # Calculate average distance between units
            total_dist = 0
            count = 0
            for i in range(len(unit_positions)):
                for j in range(i + 1, len(unit_positions)):
                    dist = abs(unit_positions[i][0] - unit_positions[j][0]) + abs(
                        unit_positions[i][1] - unit_positions[j][1]
                    )
                    total_dist += dist
                    count += 1

            if count > 0:
                avg_dist = total_dist / count
                optimal_dist = (
                    min(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT) / 3
                )

                # Reward distances close to optimal (not too close, not too far)
                dist_reward = 0.5 * (1.0 - abs(avg_dist - optimal_dist) / optimal_dist)
                reward += dist_reward

        # Reward for complementary roles
        energy_seekers = sum(1 for unit in self.team_units if unit["energy"] < 100)
        relic_seekers = sum(
            1
            for i, unit in enumerate(self.team_units)
            if i in self.closest_relic_approach and self.closest_relic_approach[i] < 5
        )

        if energy_seekers > 0 and relic_seekers > 0:
            reward += 1.0  # Reward complementary roles

        for unit in self.team_units:
            unit_pos = (unit["x"], unit["y"])
            if self.relic_map[unit_pos[1], unit_pos[0]] > 0 or self.energy_map[unit_pos[1], unit_pos[0]] > 0:
                reward += 0.5 
                
        return reward

    def combine_rewards(
        self,
        exploration_reward,
        relic_reward,
        energy_reward,
        score_reward,
        movement_reward,
        match_progress,
        match_index=0,
    ):
        """
        Simplified reward combination function focused on winning conditions.

        Parameters:
            exploration_reward: Reward for exploring new tiles
            relic_reward: Reward for finding and scoring relics
            energy_reward: Reward for managing energy
            score_reward: Reward for increasing team score
            movement_reward: Reward for efficient movement
            match_progress: Current progress through the match (0.0-1.0)
            match_index: Current match in the sequence (0-4)
        """
        if exploration_reward < 0:
            exploration_reward = 0.1
            
        exploration_scaled = min(exploration_reward * 0.5, 5.0)  # Cap at 5.0
        energy_scaled = min(energy_reward * 0.8, 5.0)  # Cap at 5.0
        relic_scaled = min(relic_reward * 2.0, 20.0)  # Cap at 20.0
        score_scaled = min(score_reward * 10.0, 50.0)  # Cap at 50.0
        score_reward = np.clip(score_reward / 50.0, -1, 1)
        
        final_reward = (
            1.0 * exploration_reward +  # Encourage exploration
            1.5 * relic_reward +  # Higher weight on relics
            1.0 * energy_reward +  # Reward energy efficiency
            0.5 * score_reward +  # Reduce dominance
            1.0 * movement_reward +  # Reward efficient movement
            0.8 * match_progress  # Reward steady progress
        )
        return np.clip(final_reward, -10, 10)
        
        # Base reward (slightly positive to encourage any action)
        # base_reward = 0.05
        
        # # Combine rewards
        # total_reward = base_reward + exploration_scaled + energy_scaled + relic_scaled + score_scaled
        
        # # Small penalty for doing nothing (-0.1)
        # if total_reward <= base_reward:
        #     total_reward = -0.1
        # Dynamic weight adjustment based on match phase
        # early_phase = match_progress < 0.3
        # mid_phase = 0.3 <= match_progress < 0.7
        # late_phase = match_progress >= 0.7

        # # Prioritize differently based on match phase
        # if early_phase:
        #     # Early game: Exploration and energy are critical
        #     exploration_weight = 1.0
        #     energy_weight = 0.8
        #     relic_weight = 0.2
        #     score_weight = 1.0
        # elif mid_phase:
        #     # Mid game: Balance between exploration, energy and relics
        #     exploration_weight = 0.5
        #     energy_weight = 0.6
        #     relic_weight = 0.8
        #     score_weight = 1.5
        # else:
        #     # Late game: Focus heavily on scoring and relics
        #     exploration_weight = 0.1
        #     energy_weight = 0.4
        #     relic_weight = 1.0
        #     score_weight = 2.0

        # Movement efficiency is always important but less so than strategic objectives
        # movement_weight = 0.3

        # # Cross-match learning (apply higher weights in later matches)
        # if match_index > 0:
        #     # Knowledge from previous matches should help with relic finding
        #     relic_weight *= 1.0 + 0.2 * match_index
        #     # In later matches, prioritize scoring over exploration
        #     if match_index >= 3:
        #         exploration_weight *= 0.5
        #         score_weight *= 1.5

        # # Calculate final reward (using the weighted components)
        # total_reward = (
        #     exploration_reward * exploration_weight
        #     + relic_reward * relic_weight
        #     + energy_reward * energy_weight
        #     + score_reward * score_weight
        #     + movement_reward * movement_weight
        # )

        # # Clip to prevent extreme values
        # total_reward = np.clip(total_reward, -10.0, 10.0)

        # # Add a sparse bonus for winning the match (if applicable)
        # if (
        #     self.is_match_end()
        #     and hasattr(self, "match_winner")
        #     and self.match_winner == 0
        # ):
        #     total_reward += 10.0

        # return total_reward

    # def combine_rewards(
    #     self,
    #     exploration_reward,
    #     relic_reward,
    #     energy_reward,
    #     score_reward,
    #     movement_reward,
    #     coordination_reward,
    #     goal_quality_reward,
    #     path_efficiency_reward,
    #     time_appropriate_reward,
    #     match_progress,
    #     strategy=0,
    # ):
    #     """Combine all reward components with dynamic weights"""

    #     # Get match index (0-4) for the 5-match sequence
    #     match_index = self.get_match_index() if hasattr(self, "get_match_index") else 0
    #     match_weight = match_index / 4.0

    #     # Calculate cross-match rewards
    #     memory_reward = self.calculate_memory_reward()
    #     exploitation_reward = self.calculate_exploitation_reward()
    #     knowledge_transfer_reward = self.calculate_knowledge_transfer_reward()
    #     parameter_learning_reward = self.calculate_parameter_learning_reward()
    #     temporal_evolution_reward = self.calculate_temporal_evolution_reward()

    #     # Dynamic weight adjustment based on match index and game progress
    #     exploration_weight = max(0.8 - match_weight - match_progress * 0.5, 0.1)
    #     energy_weight = max(0.3 - match_weight * 0.2 - match_progress * 0.2, 0.05)
    #     relic_weight = min(0.2 + match_weight * 0.4 + match_progress * 0.3, 0.8)

    #     # Cross-match learning weights increase with match index
    #     memory_weight = 0.1 + match_weight * 0.2
    #     exploitation_weight = 0.1 + match_weight * 0.3
    #     knowledge_transfer_weight = 0.1 + match_weight * 0.4
    #     parameter_learning_weight = 0.1 + match_weight * 0.2
    #     temporal_evolution_weight = 0.1 + match_weight * 0.2

    #     rewards = {
    #         "exploration_reward": exploration_reward * exploration_weight,
    #         "energy_reward": energy_reward * energy_weight,
    #         "relic_reward": relic_reward * relic_weight,
    #         "score_reward": score_reward * 1.0,
    #         "movement_reward": movement_reward * 0.1,
    #         "coordination_reward": coordination_reward * 0.2,
    #         "goal_quality_reward": goal_quality_reward * 0.2,
    #         "path_efficiency_reward": path_efficiency_reward * 0.2,
    #         "time_appropriate_reward": time_appropriate_reward * 0.2,
    #         "memory_reward": memory_reward * memory_weight,
    #         "exploitation_reward": exploitation_reward * exploitation_weight,
    #         "knowledge_transfer_reward": knowledge_transfer_reward * knowledge_transfer_weight,
    #         "parameter_learning_reward": parameter_learning_reward * parameter_learning_weight,
    #         "temporal_evolution_reward": temporal_evolution_reward * temporal_evolution_weight
    #     }

    #     # Apply strategy modification
    #     rewards = self.apply_strategy(strategy, rewards)

    #     # Calculate final reward
    #     total_reward = sum(rewards.values())

    #     return total_reward

    def process_enemy_moves(self):
        """Process movements for enemy units"""
        for enemy in self.enemy_units:
            # Find nearest player unit
            nearest_unit = min(
                self.team_units,
                key=lambda u: abs(u["x"] - enemy["x"]) + abs(u["y"] - enemy["y"]),
            )

            # Move toward player with 70% probability, otherwise move randomly
            if np.random.random() < 0.7:
                dx = np.clip(nearest_unit["x"] - enemy["x"], -1, 1)
                dy = np.clip(nearest_unit["y"] - enemy["y"], -1, 1)
            else:
                dx, dy = np.random.choice([-1, 0, 1]), np.random.choice([-1, 0, 1])

            new_x = np.clip(enemy["x"] + dx, 0, GameConstants.MAP_WIDTH - 1)
            new_y = np.clip(enemy["y"] + dy, 0, GameConstants.MAP_HEIGHT - 1)

            # Only move if not blocked by asteroid
            if self.tile_map[new_y, new_x] != NodeType.ASTEROID.value:
                enemy["x"], enemy["y"] = new_x, new_y

    # def determine_unit_goals(self, goal_position, ship):
    #     """Determine goals for a specific unit based on game state"""
    #     game_step = self.current_step
    #     position = ship.get_position()
    #     energy = ship.get_energy()

    #     # Priority 1: Find scoring tiles around relics if we have enough energy
    #     if energy >= 100:
    #         # Get all relic configurations
    #         for rx, ry, mask in self.relic_config:

    #             if abs(goal_position[0] - rx) <= 2 and abs(goal_position[1] - ry) <= 2:
    #                 mask_x = goal_position[0] - rx + 2
    #                 mask_y = goal_position[1] - ry + 2
    #                 if 0 <= mask_x < 5 and 0 <= mask_y < 5:
    #                     target_x = goal_position[0]
    #                     target_y = goal_position[1]

    #                     # Check bounds again
    #                     if (0 <= target_x < GameConstants.MAP_WIDTH and
    #                         0 <= target_y < GameConstants.MAP_HEIGHT):

    #                         # If it's a scoring tile that hasn't been scored yet
    #                         if mask[mask_y, mask_x] and not self.team_points_space[target_y, target_x]:
    #                             return goal_position
    #                 # Make sure the coordinates are valid before checking
    #                 # target_x = rx - 2 + mask_x
    #                 # target_y = ry - 2 + mask_y

    #                 # if (0 <= target_x < GameConstants.MAP_WIDTH and
    #                 # 0 <= target_y < GameConstants.MAP_HEIGHT):
    #                 #     if not self.team_points_space[target_y, target_x]:
    #                 #         print("Scoring tile found, moving towards it: ", target_x, target_y)
    #                 #         return (target_x, target_y)
    #             # Find scoring tiles in this relic's configuration
    #             for my in range(5):
    #                 for mx in range(5):
    #                     if mask[my, mx]:
    #                         # Calculate real map coordinates
    #                         target_x = rx - 2 + mx
    #                         target_y = ry - 2 + my

    #                         if not self.team_points_space[target_y, target_x]:
    #                             return (target_x, target_y)

    #                         # Check if valid coordinates
    #                         if (
    #                             0 <= target_x < GameConstants.MAP_WIDTH
    #                             and 0 <= target_y < GameConstants.MAP_HEIGHT
    #                         ):
    #                             # If not already visited or we're prioritizing points over exploration
    #                             if (
    #                                 not self.team_points_space[target_y, target_x]
    #                                 # or self.current_step
    #                                 # > GameConstants.MAX_STEPS_IN_MATCH * 0.5
    #                             ):
    #                                 # Calculate distance
    #                                 dist = abs(target_x - position[0]) + abs(
    #                                     target_y - position[1]
    #                                 )
    #                                 # If reasonably close, go for it
    #                                 if dist < 15:
    #                                     return (target_x, target_y)
    #     if energy < 100:
    #         for y in range(GameConstants.MAP_HEIGHT):
    #             for x in range(GameConstants.MAP_WIDTH):
    #                 if self.energy_map[y, x] > 0:
    #                     return (x, y)

    #     # Priority 3: Explore unexplored areas
    #     unexplored_positions = np.argwhere(~self.visited)
    #     if len(unexplored_positions) > 0:
    #         # Find the closest unexplored position
    #         closest_pos = min(
    #             unexplored_positions,
    #             key=lambda pos: abs(pos[1] - position[0]) + abs(pos[0] - position[1]),
    #         )
    #         return (closest_pos[1], closest_pos[0])

    #     # Fallback: Move toward a relic
    #     relic_coords = np.argwhere(self.relic_map == NodeType.RELIC_NODE.value)
    #     if len(relic_coords) > 0:
    #         closest_relic = min(
    #             relic_coords,
    #             key=lambda coord: abs(coord[1] - position[0])
    #             + abs(coord[0] - position[1]),
    #         )
    #         return (closest_relic[1], closest_relic[0])

    #     # Last resort: center of map
    #     return (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)

    # def determine_unit_goals(self, goal_position, ship):
        """Determine goals for a specific unit based on game state"""
        # game_step = self.current_step
        # position = ship.get_position()
        # energy = ship.get_energy()

        # # Priority 1: Find scoring tiles around relics if we have enough energy
        # if energy >= 100:
        #     # Get all relic configurations
        #     for rx, ry, mask in self.relic_config:
        #         # Check if the goal position is near this relic
        #         if abs(goal_position[0] - rx) <= 2 and abs(goal_position[1] - ry) <= 2:
        #             # Calculate position in the mask
        #             mask_x = goal_position[0] - rx + 2
        #             mask_y = goal_position[1] - ry + 2

        #             # Ensure mask coordinates are valid
        #             if 0 <= mask_x < 5 and 0 <= mask_y < 5:
        #                 # Calculate target coordinates
        #                 target_x = rx - 2 + mask_x
        #                 target_y = ry - 2 + mask_y

        #                 # Ensure target coordinates are within map bounds
        #                 if (
        #                     0 <= target_x < GameConstants.MAP_WIDTH
        #                     and 0 <= target_y < GameConstants.MAP_HEIGHT
        #                 ):

        #                     # Check if this is a scoring tile that hasn't been scored
        #                     if (
        #                         mask[mask_y, mask_x]
        #                         and not self.team_points_space[target_y, target_x]
        #                     ):
        #                         return goal_position

        #         # Also check all scoring tiles in this relic config
        #         for my in range(5):
        #             for mx in range(5):
        #                 if mask[my, mx]:
        #                     # Calculate real map coordinates
        #                     target_x = rx - 2 + mx
        #                     target_y = ry - 2 + my

        #                     # Ensure coordinates are within map bounds
        #                     if (
        #                         0 <= target_x < GameConstants.MAP_WIDTH
        #                         and 0 <= target_y < GameConstants.MAP_HEIGHT
        #                     ):

        #                         # Check if this tile hasn't been scored yet
        #                         if not self.team_points_space[target_y, target_x]:
        #                             # Calculate distance
        #                             dist = abs(target_x - position[0]) + abs(
        #                                 target_y - position[1]
        #                             )
        #                             # If reasonably close, go for it
        #                             if dist < 15:
        #                                 return (target_x, target_y)

        # # Priority 2: Find energy when low
        # if energy < 100:
        #     for y in range(GameConstants.MAP_HEIGHT):
        #         for x in range(GameConstants.MAP_WIDTH):
        #             if self.energy_map[y, x] > 0:
        #                 return (x, y)

        # # Priority 3: Explore unexplored areas
        # unexplored_positions = np.argwhere(~self.visited)
        # if len(unexplored_positions) > 0:
        #     # Find the closest unexplored position
        #     closest_pos = min(
        #         unexplored_positions,
        #         key=lambda pos: abs(pos[1] - position[0]) + abs(pos[0] - position[1]),
        #     )
        #     return (closest_pos[1], closest_pos[0])

        # # Fallback: Move toward a relic
        # relic_coords = np.argwhere(self.relic_map == NodeType.RELIC_NODE.value)
        # if len(relic_coords) > 0:
        #     closest_relic = min(
        #         relic_coords,
        #         key=lambda coord: abs(coord[1] - position[0])
        #         + abs(coord[0] - position[1]),
        #     )
        #     return (closest_relic[1], closest_relic[0])

        # Last resort: center of map
        # return (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)
    def determine_unit_goals(self, goal_position, ship):
        match_index = self.get_match_index() if hasattr(self, "get_match_index") else 0
        exploration_weight = max(0.9 - match_index * 0.2, 0.3)
        position = ship.get_position()
        energy = ship.get_energy()
        match_progress = self.current_step / self.max_steps
        
        if random.random() < exploration_weight:
            unexplored_positions = np.argwhere(~self.visited)
            if len(unexplored_positions) > 0:
                # Select a random unexplored position
                random_idx = np.random.randint(0, len(unexplored_positions))
                return (unexplored_positions[random_idx][1], unexplored_positions[random_idx][0])
        
        # If in later matches, prioritize known scoring tiles from memory
        if match_index > 0 and hasattr(self, "scoring_tiles") and self.scoring_tiles:
            if energy >= 100 and match_progress > 0.3:
                # Find closest scoring tile from memory
                closest_scoring = min(
                    self.scoring_tiles,
                    key=lambda pos: abs(pos[0] - position[0]) + abs(pos[1] - position[1])
                )
                return closest_scoring
        
        # If energy is low, go to remembered energy locations
        if energy < 100 and hasattr(self, "energy_locations") and self.energy_locations:
            closest_energy = min(
                self.energy_locations,
                key=lambda pos: abs(pos[0] - position[0]) + abs(pos[1] - position[1])
            )
            return closest_energy
        
        # If we have remembered relics, prefer them in later game
        if match_progress > 0.5 and hasattr(self, "discovered_relics") and self.discovered_relics:
            closest_relic = min(
                self.discovered_relics,
                key=lambda pos: abs(pos[0] - position[0]) + abs(pos[1] - position[1])
            )
            return closest_relic
        return (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)

    def render(self, mode="human"):
        display = self.tile_map.astype(str).copy()
        for unit in self.team_units:
            display[unit["y"], unit["x"]] = "A"
        print("Step:", self.current_step)
        print(display)

    def is_on_relic(self, position):
        """Check if a position is on a relic node"""
        x, y = position
        # Ensure position is within bounds
        if 0 <= x < GameConstants.MAP_WIDTH and 0 <= y < GameConstants.MAP_HEIGHT:
            # Check if this position has a relic node
            return self.relic_map[y, x] == NodeType.RELIC_NODE.value
        return False


def detect_symmetry(tile_map):
    """Detects symmetry in asteroid, nebula, and energy tiles separately."""

    def is_symmetric(matrix1, matrix2, tolerance=0.05):
        """Helper function to check if two matrices are symmetric within a small tolerance."""
        return np.allclose(matrix1, matrix2, atol=tolerance)

    # Extract tile types
    asteroid_map = (tile_map == NodeType.ASTEROID.value).astype(np.float32)
    nebula_map = (tile_map == NodeType.NEBULA.value).astype(np.float32)
    energy_map = (tile_map == NodeType.ENERGY_NODE.value).astype(np.float32)

    # Check symmetry for each type
    asteroid_symmetry = [
        is_symmetric(asteroid_map, np.flip(asteroid_map, axis=1)),  # Horizontal
        is_symmetric(asteroid_map, np.flip(asteroid_map, axis=0)),  # Vertical
        is_symmetric(asteroid_map, np.rot90(asteroid_map, 2)),  # Rotational
    ]

    nebula_symmetry = [
        is_symmetric(nebula_map, np.flip(nebula_map, axis=1)),
        is_symmetric(nebula_map, np.flip(nebula_map, axis=0)),
        is_symmetric(nebula_map, np.rot90(nebula_map, 2)),
    ]

    energy_symmetry = [
        is_symmetric(energy_map, np.flip(energy_map, axis=1)),
        is_symmetric(energy_map, np.flip(energy_map, axis=0)),
        is_symmetric(energy_map, np.rot90(energy_map, 2)),
    ]

    # Combine results
    return {
        "asteroid_symmetry": np.array(asteroid_symmetry, dtype=np.float32),
        "nebula_symmetry": np.array(nebula_symmetry, dtype=np.float32),
        "energy_symmetry": np.array(energy_symmetry, dtype=np.float32),
    }
