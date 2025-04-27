import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces

from luxai_s3.wrappers import LuxAIS3GymEnv
from stable_baselines3 import PPO
from common.environment import GameConstants, ActionType


class SB3LuxEnvBase(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with Stable Baselines 3.
    This wrapper focuses on training a single player while allowing the opponent to be controlled by a different strategy.

    Key features:
    - Changes opponent difficulty as training progresses
    - Converts observations to work with Stable Baselines 3
    - Creates helpful rewards for exploration, gathering resources, and controlling objectives
    - Translates simple actions to the format needed by the Lux environment
    - Tracks the map to help units explore efficiently

    """

    def __init__(
        self,
        env=None,
        player_id="player_0",
        opponent_strategy="random",
        max_units=GameConstants.MAX_UNITS,
    ):
        if env is None:
            env = LuxAIS3GymEnv()
        super().__init__(env)

        # Store the player ID
        self.player_id = player_id
        self.opponent_id = "player_1" if player_id == "player_0" else "player_0"
        self.max_units = max_units

        # Set opponent strategy
        self.opponent_strategy = opponent_strategy

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                # Unit information (position, energy, and existence mask)
                "units_position": spaces.Box(
                    low=-1,
                    high=GameConstants.MAP_SIZE - 1,
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2),
                    dtype=np.int32,
                ),
                "units_energy": spaces.Box(
                    low=0,
                    high=400,  # Max energy a unit can have
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 1),
                    dtype=np.int32,
                ),
                "units_mask": spaces.Box(
                    low=0,
                    high=1,  # Binary mask: 1 if unit exists, 0 otherwise
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS),
                    dtype=np.int8,
                ),
                # Map information
                "sensor_mask": spaces.Box(
                    low=0,
                    high=1,  # Binary mask: 1 if tile is visible, 0 otherwise
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                "map_features_tile_type": spaces.Box(
                    low=-1,
                    high=2,  # -1: unknown, 0: empty, 1: nebula, 2: asteroid
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                "map_features_energy": spaces.Box(
                    low=-1,
                    high=GameConstants.MAX_ENERGY_PER_TILE,  # Energy available on tile (-1 if unknown)
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                # Relic node information
                "relic_nodes_mask": spaces.Box(
                    low=0,
                    high=1,  # Binary mask: 1 if relic node is visible, 0 otherwise
                    shape=(GameConstants.MAX_RELIC_NODES,),
                    dtype=np.int8,
                ),
                "relic_nodes": spaces.Box(
                    low=-1,
                    high=GameConstants.MAP_WIDTH
                    - 1,  # Position of relic nodes (-1 if unknown/invalid)
                    shape=(GameConstants.MAX_RELIC_NODES, 2),
                    dtype=np.int32,
                ),
                # Score and game state information
                "team_points": spaces.Box(
                    low=0,
                    high=1000,  # Points for each team
                    shape=(GameConstants.NUM_TEAMS,),
                    dtype=np.int32,
                ),
                "team_wins": spaces.Box(
                    low=0,
                    high=1000,  # Win count for each team
                    shape=(GameConstants.NUM_TEAMS,),
                    dtype=np.int32,
                ),
                "steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,  # Current step in the match
                    shape=(1,),
                    dtype=np.int32,
                ),
                "match_steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,  # Total match steps
                    shape=(1,),
                    dtype=np.int32,
                ),
                # Environment configuration parameters
                "remainingOverageTime": spaces.Box(
                    low=0,
                    high=1000,  # Remaining computation time
                    shape=(1,),
                    dtype=np.int32,
                ),
                "env_cfg_map_width": spaces.Box(
                    low=0, high=GameConstants.MAP_WIDTH, shape=(1,), dtype=np.int32
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
                    low=0,
                    high=100,  # Energy cost for movement
                    shape=(1,),
                    dtype=np.int32,
                ),
                "env_cfg_unit_sap_cost": spaces.Box(
                    low=0,
                    high=100,  # Energy cost for sap action
                    shape=(1,),
                    dtype=np.int32,
                ),
                "env_cfg_unit_sap_range": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32  # Range of sap action
                ),
            }
        )

        # State tracking for the opponent
        self.opponent_obs = None  # Stores opponent's observations
        self.opponent_model = None  # Holds loaded model for self-play opponent
        self.opponent_model_path = None  # Path to the model used for self-play

        # State tracking for the current agent
        self.last_obs = None  # Previous step's observation
        self.last_info = None  # Previous step's info dictionary
        self.last_energy = None  # Total energy from previous step
        self.last_move_direction = (
            None  # Last movement direction (used for sap targeting)
        )
        self.energy_map = None  # Tracked energy distribution on map

        # Define action space (0=stay, 1-4=movement directions, 5=sap)
        self.action_space = spaces.MultiDiscrete([6] * GameConstants.MAX_UNITS)

        # Exploration tracking
        self.cumulative_sensor_mask = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )  # All tiles ever seen
        self.visited_tiles = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )  # All tiles visited by units
        self.tiles_visited_this_step = (
            set()
        )  # Tiles visited in current step (for penalties)

        # Game state tracking
        self.relic_points_tiles = set()  # Set of positions that generate points
        self.consecutive_relic_control = (
            0  # Counter for consecutive turns controlling relics
        )
        self.last_total_points = 0  # Points from previous step
        self.team_units = []  # List of active units on our team
        self.enemy_units = []  # List of known enemy units
        self.unit_position_history = {}  # History of positions for each unit

        # Map and game configuration
        self.team_spawn = (0, 0)  # Starting position for our team
        self.enemy_spawn = (
            GameConstants.MAP_WIDTH - 1,
            GameConstants.MAP_HEIGHT - 1,
        )  # Enemy start
        self.current_step = 0  # Current step in episode
        self.last_spawn_step = 0  # Last step when a unit was spawned
        self.spawn_interval = 10  # Steps between spawning new units
        self.base_model_path = None  # Path to model used for self-play

        # Environment configuration parameters
        self.env_config = {
            "unit_move_cost": 1,  # Energy cost for movement
            "unit_sensor_range": 1,  # Base visibility range
            "nebula_tile_vision_reduction": 8,  # Vision penalty in nebula
            "nebula_tile_energy_reduction": 25,  # Energy penalty in nebula
            "unit_sap_cost": 51,  # Energy cost for sap action
            "unit_sap_range": 3,  # Range of sap ability
        }

    def reset(self, **kwargs):
        """
        Reset the environment and initialize state variables.

        Args:
            **kwargs: Additional arguments for the base environment reset.
        Returns:
            obs: The initial observation after reset.
            info: Additional information about the environment state.
        """

        obs, info = self.env.reset(**kwargs)
        # Check if this is a new game or just a new match
        is_new_game = False
        if (
            hasattr(obs[self.player_id], "match_steps")
            and obs[self.player_id].match_steps == 0
        ):
            # This is likely a new game (first match starting)
            is_new_game = True

        if is_new_game:
            # FULL RESET - only for new games (every 5 matches)
            # Reset exploration tracking
            self.cumulative_sensor_mask = np.zeros(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
            )
            self.visited_tiles = np.zeros(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
            )
            # Reset game tracking
            self.relic_points_tiles = set()
            self.unit_position_history = {i: [] for i in range(GameConstants.MAX_UNITS)}

        # Reset all state variables
        self.opponent_obs = None
        self.opponent_model = None
        self.opponent_model_path = None
        self.last_obs = None
        self.last_info = None
        self.last_energy = None
        self.last_move_direction = None
        self.energy_map = None

        # Reset game tracking
        self.consecutive_relic_control = 0
        self.last_total_points = 0

        # Reset step counter
        self.current_step = 0
        self.last_spawn_step = 0
        self.spawn_interval = 25

        # Reset team units and enemy units
        self.team_spawn = (
            (0, 0)
            if self.player_id == "player_0"
            else (GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1)
        )
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

        self.enemy_spawn = (
            (GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1)
            if self.player_id == "player_0"
            else (0, 0)
        )
        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})

        # Store initial observation
        self.last_obs = obs
        self.last_info = info

        processed_obs = self.process_observation(obs, info)

        return processed_obs, info

    def get_opponent_action(self, obs=None, info=None):
        """
        Generate actions for the opponent based on the random selection of strategies.
        Uses a mixture of strategies to create better training.

        Args:
            obs: The current observation from the environment.
            info: Additional information about the environment state.

        Returns:
            env_action: The action to be taken by the opponent.
        """

        strategy_choice = np.random.random()

        # As training progresses, gradually shift from random to self-play opponents
        if self.base_model_path is None or (
            self.base_model_path is not None
            and not os.path.exists(self.base_model_path)
        ):  # Early training
            # 80% random, 20% rule-based
            if strategy_choice < 0.8:
                return self.random_strategy()
            else:
                return self.rule_based_strategy(obs)
        elif self.current_step < 250:  # Mid training
            # 40% rule-based, 60% previous model
            if strategy_choice < 0.4:
                return self.rule_based_strategy(obs)
            else:
                return self.model_based_strategy(obs, info)
        else:  # Later training
            # 20% rule-based, 80% previous models
            if strategy_choice < 0.2:
                return self.rule_based_strategy(obs)
            else:
                return self.model_based_strategy(obs, info)

    def random_strategy(self):
        """
        Simple random action strategy

        Args:
            None

        Returns:
            env_action: Random action for the opponent
        """

        random_actions = np.random.randint(0, 6, size=(GameConstants.MAX_UNITS,))

        env_action = {
            self.opponent_id: np.zeros((GameConstants.MAX_UNITS, 3), dtype=np.int16),
        }

        for idx, action in enumerate(random_actions):
            env_action[self.opponent_id][idx, 0] = action
            env_action[self.opponent_id][idx, 1] = 0
            env_action[self.opponent_id][idx, 2] = 0

            if action == 5:
                # Randomly choose a target for the sap action
                sap_range = self.env_config["unit_sap_range"] if self.env_config else 3
                dx = np.random.randint(-sap_range, sap_range + 1)
                dy = np.random.randint(-sap_range, sap_range + 1)
                env_action[self.opponent_id][idx, 1] = dx
                env_action[self.opponent_id][idx, 2] = dy

        return env_action

    def rule_based_strategy(self, obs):
        """
        A simple rule-based strategy that's better than random

        Args:
            obs: The current observation from the environment.

        Returns:
            action: The action to be taken by the opponent.
        """
        if obs is None:
            return self.random_strategy()

        action = {self.opponent_id: np.zeros((16, 3), dtype=np.int16)}

        # Get opponent unit positions and masks
        unit_mask = np.array(obs[self.opponent_id].units_mask)
        available_units = np.where(unit_mask)[0]

        for unit_id in available_units:
            unit_pos = obs[self.opponent_id].units.position[1][unit_id]

            # Look for energy or move toward relic nodes
            if (
                np.random.random() < 0.7
                and len(obs[self.opponent_id].relic_nodes_mask) > 0
            ):
                # Find visible relic nodes
                visible_relics = np.where(obs[self.opponent_id].relic_nodes_mask)[0]
                if len(visible_relics) > 0:
                    # Move toward a random visible relic
                    relic_idx = np.random.choice(visible_relics)
                    relic_pos = obs[self.opponent_id].relic_nodes[relic_idx]

                    # Determine direction to move (simplistic approach)
                    direction = ActionType.STAY.value
                    if relic_pos[0] > unit_pos[0]:
                        direction = ActionType.MOVE_RIGHT.value
                    elif relic_pos[0] < unit_pos[0]:
                        direction = ActionType.MOVE_LEFT.value

                    if relic_pos[1] > unit_pos[1]:
                        direction = ActionType.MOVE_DOWN.value
                    elif relic_pos[1] < unit_pos[1]:
                        direction = ActionType.MOVE_UP.value

                    # Set action to move
                    action[self.opponent_id][unit_id, 0] = direction
                    action[self.opponent_id][unit_id, 1] = 0
                    action[self.opponent_id][unit_id, 2] = 0
                    continue

            # Otherwise move randomly
            random_action = np.random.randint(0, 5)

            if random_action == ActionType.STAY.value:
                action[self.opponent_id][unit_id, 0] = ActionType.STAY.value
            else:  # Move in a direction
                action[self.opponent_id][
                    unit_id, 0
                ] = random_action  # Random move action
                action[self.opponent_id][unit_id, 1] = 0
                action[self.opponent_id][unit_id, 2] = 0
                if random_action == ActionType.SAP.value:
                    sap_range = obs[self.opponent_id].env_cfg_unit_sap_cost
                    dx, dy = 0, 0

                    # Target based on last movement direction if available
                    if self.last_move_direction is not None:
                        if self.last_move_direction == ActionType.MOVE_UP.value:
                            dy = -sap_range
                        elif self.last_move_direction == ActionType.MOVE_RIGHT.value:
                            dx = sap_range
                        elif self.last_move_direction == ActionType.MOVE_DOWN.value:
                            dy = sap_range
                        elif self.last_move_direction == ActionType.MOVE_LEFT.value:
                            dx = -sap_range

                    action[self.opponent_id][unit_id, 1] = dx
                    action[self.opponent_id][unit_id, 2] = dy

        return action

    def model_based_strategy(self, obs, info):
        """
        Use a previous model checkpoint as the opponent

        Args:
            obs: The current observation from the environment.
            info: Additional information about the environment state.

        Returns:
            action: The action to be taken by the opponent.
        """
        if (
            self.base_model_path == None
            and not os.path.exists(self.base_model_path)
            or obs is None
            or info is None
        ):
            return self.random_strategy()

        try:
            # Only reload if it's a different model
            if (
                not hasattr(self, "opponent_model_path")
                or self.opponent_model_path != self.base_model_path
            ):
                self.opponent_model = PPO.load(self.base_model_path)
                self.opponent_model_path = self.base_model_path

            # Get action from model
            opponent_action, _ = self.opponent_model.predict(
                self.process_observation(obs, info), deterministic=False
            )

            return self.process_action(opponent_action, self.opponent_id)
        except Exception as e:
            print(f"Error loading or using model: {e}")
            return self.random_strategy()

    def find_resource_targets(self, obs):
        """
        Find resource targets (energy nodes and relic nodes) for units based on visibility.

        Args:
            obs: The current observation from the environment.

        Returns:
            targets: A dictionary of unit indices and their corresponding target positions.
        """
        targets = {}

        # Extract player's units and their positions
        unit_positions = []

        # Get positions of active units
        for i in range(len(obs.units_mask)):
            if obs.units_mask[i].any():
                pos = obs.units.position[i]
                unit_energy = obs.units.energy[i][0]  # Get unit's energy
                unit_positions.append((i, unit_energy))

            # If we have units
            if unit_positions:
                # Get visible map area
                visible_mask = np.array(obs.sensor_mask)

                # For each unit, find the most valuable target
                for unit_idx, unit_energy in unit_positions:
                    # Start with no target
                    best_target = None
                    best_target_type = None  # 'energy' or 'relic'
                    best_value = -float("inf")
                    unit_pos = pos[unit_idx]
                    energy_nodes = np.array(obs.map_features.energy > 0)

                    # Check energy nodes
                    for node_pos in energy_nodes:
                        # Calculate Manhattan distance
                        distance = abs(node_pos[0] - unit_pos[0]) + abs(
                            node_pos[1] - unit_pos[1]
                        )

                        # Skip if node is not visible
                        if not visible_mask[node_pos[1], node_pos[0]].any():
                            continue

                        # Calculate value of this energy node
                        energy_value = 10.0 / (distance + 1)

                        # If unit is low on energy, energy nodes become more valuable
                        if unit_energy < 30:
                            energy_value *= 2

                        if energy_value > best_value:
                            best_value = energy_value
                            best_target = tuple(node_pos)
                            best_target_type = "energy"

                    # Check relic nodes
                    if len(obs.relic_nodes) > 0:
                        for node_pos in obs.relic_nodes:
                            # Skip invalid nodes (value -1 means not visible)
                            if node_pos[0] == -1 or node_pos[1] == -1:
                                continue

                            # Calculate Manhattan distance
                            distance = abs(node_pos[0] - unit_pos[0]) + abs(
                                node_pos[1] - unit_pos[1]
                            )

                            # Skip if node is not visible
                            if not visible_mask[node_pos[1], node_pos[0]].any():
                                continue

                            # Calculate value of this relic node
                            relic_value = 20.0 / (distance + 1)

                            # If unit has enough energy to mine the relic
                            if unit_energy >= 60:
                                relic_value *= 1.5

                            if relic_value > best_value:
                                best_value = relic_value
                                best_target = tuple(node_pos)
                                best_target_type = "relic"

                    # If we found a target, set it
                    if best_target is not None:
                        targets[unit_idx] = (best_target, best_target_type)
                    else:
                        # No visible resource nodes, explore in a direction with high energy on the map
                        energy_map = np.array(obs.map_features.energy)
                        visible_energy = energy_map * visible_mask

                        # Find highest energy visible tile
                        if np.max(visible_energy) > 0:
                            max_pos = np.unravel_index(
                                np.argmax(visible_energy), visible_energy.shape
                            )
                            targets[unit_idx] = (
                                (max_pos[1], max_pos[0]),
                                "exploration",
                            )  # Convert (y,x) to (x,y)

        return targets

    def process_observation(self, obs, info, remining_overage_time=60):
        """
        Process observation into a dictionary format compatible with Dict observation space.

        Args:
            obs: The current observation from the environment.
            info: Additional information about the environment state.
            remining_overage_time: Remaining computation time.

        Returns:
            processed_obs: A dictionary containing the processed observation.
        """
        self.opponent_obs = obs[self.opponent_id]
        obs = obs[self.player_id]

        # Initialize Dict observation
        processed_obs = {}

        # Define constants if not already defined elsewhere in your class
        SPACE_SIZE = GameConstants.MAP_SIZE
        MAX_RELIC_NODES = GameConstants.MAX_RELIC_NODES
        num_tiles = SPACE_SIZE * SPACE_SIZE
        defalt_pos = 0 if self.player_id == "player_0" else GameConstants.MAP_WIDTH - 1

        # Basic unit information
        units_position = np.full(
            (GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2),
            defalt_pos,
            dtype=np.int32,
        )
        units_energy = np.full(
            (GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 1), 0, dtype=np.int32
        )
        units_mask = np.zeros(
            (GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS), dtype=np.int8
        )
        for i, u in enumerate(self.team_units):
            ux, uy = u["x"], u["y"]
            if obs.sensor_mask[uy, ux]:
                units_position[0, i] = np.array([ux, uy])
                units_energy[0, i] = obs.units.energy[0][i]
                units_mask[0, i] = 1
        for i, u in enumerate(self.enemy_units):
            ux, uy = u["x"], u["y"]
            if self.opponent_obs.sensor_mask[uy, ux]:
                units_position[1, i] = np.array([ux, uy])
                units_energy[1, i] = self.opponent_obs.units.energy[1][i]
                units_mask[1, i] = 1
        processed_obs["units_position"] = units_position
        processed_obs["units_energy"] = units_energy
        processed_obs["units_mask"] = units_mask

        # Map information
        processed_obs["sensor_mask"] = np.array(obs.sensor_mask, dtype=np.int8)
        processed_obs["map_features_tile_type"] = np.array(
            obs.map_features.tile_type, dtype=np.int8
        )

        self.energy_map = np.zeros((SPACE_SIZE, SPACE_SIZE), dtype=np.int8)
        num_energy_nodes = 2
        indices_energy = np.random.choice(num_tiles, num_energy_nodes, replace=False)
        flat_energy = self.energy_map.flatten()
        for idx in indices_energy:
            flat_energy[idx] = GameConstants.MAX_ENERGY_PER_TILE
        self.energy_map = flat_energy.reshape((SPACE_SIZE, SPACE_SIZE))
        map_features_energy = np.array(obs.map_features.energy, dtype=np.int8)

        processed_obs["map_features_energy"] = np.where(
            map_features_energy > 0, map_features_energy, -1
        )

        # Resource nodes
        processed_obs["relic_nodes"] = np.array(obs.relic_nodes, dtype=np.int32)

        # Create relic nodes mask (1 for valid nodes, 0 for invalid)
        relic_mask = np.zeros(MAX_RELIC_NODES, dtype=np.int8)
        for i in range(min(MAX_RELIC_NODES, len(obs.relic_nodes))):
            if (
                obs.relic_nodes[i][0] >= 0 and obs.relic_nodes[i][1] >= 0
            ):  # Valid coordinates
                relic_mask[i] = 1

        processed_obs["relic_nodes_mask"] = relic_mask

        processed_obs["team_points"] = np.array(obs.team_points, dtype=np.int32)
        processed_obs["team_wins"] = np.array(obs.team_wins, dtype=np.int32)
        processed_obs["steps"] = np.array([obs.steps], dtype=np.int32)
        processed_obs["match_steps"] = np.array([obs.match_steps], dtype=np.int32)

        # If this information is in info, otherwise use defaults
        if "full_params" in info:
            max_steps = info["full_params"]["max_steps_in_match"]
            unit_move_cost = info["full_params"]["unit_move_cost"]
            unit_sap_cost = info["full_params"]["unit_sap_cost"]
            unit_sap_range = info["full_params"]["unit_sap_range"]
            unit_sensor_range = info["full_params"]["unit_sensor_range"]
            map_width = info["full_params"]["map_width"]
            map_height = info["full_params"]["map_height"]
            nebula_tile_vision_reduction = info["full_params"][
                "nebula_tile_vision_reduction"
            ]
            nebula_tile_energy_reduction = info["full_params"][
                "nebula_tile_energy_reduction"
            ]
        else:
            # Defaults
            max_steps = GameConstants.MAX_STEPS_IN_MATCH
            unit_move_cost = GameConstants.DEFUALT_UNIT_MOVE_COST
            unit_sap_cost = GameConstants.DEFAULT_UNIT_SAP_COST
            unit_sap_range = GameConstants.DEFAULT_UNIT_SAP_RANGE
            unit_sensor_range = GameConstants.DEFAULT_UNIT_SENSOR_RANGE
            nebula_tile_vision_reduction = (
                GameConstants.DEFAULT_NEBULA_TILE_VISION_REDUCTION
            )
            nebula_tile_energy_reduction = (
                GameConstants.DEFAULT_NEBULA_TILE_ENERGY_REDUCTION
            )
            map_width = GameConstants.MAP_WIDTH
            map_height = GameConstants.MAP_HEIGHT

        processed_obs["env_cfg_max_steps_in_match"] = np.array(
            [max_steps], dtype=np.int32
        )
        processed_obs["env_cfg_unit_move_cost"] = np.array(
            [unit_move_cost], dtype=np.int32
        )
        processed_obs["env_cfg_unit_sap_cost"] = np.array(
            [unit_sap_cost], dtype=np.int32
        )
        processed_obs["env_cfg_unit_sap_range"] = np.array(
            [unit_sap_range], dtype=np.int32
        )
        processed_obs["remainingOverageTime"] = np.array(
            [remining_overage_time], dtype=np.int32
        )
        processed_obs["env_cfg_map_width"] = np.array([map_width], dtype=np.int32)
        processed_obs["env_cfg_map_height"] = np.array([map_height], dtype=np.int32)

        self.env_config = {
            "unit_move_cost": unit_move_cost,
            "unit_sensor_range": unit_sensor_range,
            "nebula_tile_vision_reduction": nebula_tile_vision_reduction,
            "nebula_tile_energy_reduction": nebula_tile_energy_reduction,
            "unit_sap_cost": unit_sap_cost,
            "unit_sap_range": unit_sap_range,
        }

        return processed_obs

    def process_action(self, actions, player_id="player_0", obs=None):
        """
        Convert a simple Discrete action to the Dict action expected by the environment.

        Args:
            actions: The action to be taken by the player.
            player_id: The ID of the player (default is "player_0").
            obs: The current observation from the environment.

        Returns:
            env_action: The action to be taken by the player in the environment.
        """

        # Initialize actions for all units
        env_action = {
            player_id: np.zeros((GameConstants.MAX_UNITS, 3), dtype=np.int16),
        }

        for idx, action in enumerate(actions):

            if ActionType.MOVE_UP.value <= action <= ActionType.MOVE_LEFT.value:
                env_action[player_id][idx, 0] = action
                env_action[player_id][idx, 1] = 0
                env_action[player_id][idx, 2] = 0

                self.last_move_direction = action

            elif action == ActionType.SAP.value:
                env_action[player_id][idx, 0] = ActionType.SAP.value

                # Get the sap range from parameters
                sap_range = self.env_config["unit_sap_range"] if self.env_config else 3
                dx, dy = 0, 0
                player_idx = 0 if player_id == "player_0" else 1
                enemy_idx = 1 if player_id == "player_0" else 0

                if obs is not None and idx < len(obs[player_id].units_mask):
                    # Check if this unit exists and can take actions
                    if obs[player_id].units_mask[idx]:
                        # Get the unit's current position
                        unit_pos = obs[player_id].units.position[player_idx][idx]

                        # Find visible enemy units
                        enemy_positions = []
                        for idx in range(len(obs[player_id].units_mask)):
                            if obs[player_id].units_mask[idx]:
                                enemy_pos = obs[player_id].units.position[enemy_idx][
                                    idx
                                ]

                                # Check if enemy is visible (within sensor range)
                                ex, ey = enemy_pos[0], enemy_pos[1]
                                if (
                                    0 <= ex < obs[player_id].sensor_mask.shape[1]
                                    and 0 <= ey < obs[player_id].sensor_mask.shape[0]
                                ):
                                    if obs[player_id].sensor_mask[ey][ex]:
                                        enemy_positions.append(
                                            (enemy_pos[0], enemy_pos[1])
                                        )

                        # Count enemy units at each position to find stacks
                        enemy_counts = {}
                        for pos in enemy_positions:
                            if pos in enemy_counts:
                                enemy_counts[pos] += 1
                            else:
                                enemy_counts[pos] = 1

                        best_target = None
                        best_score = -1

                        # Evaluate each enemy position
                        for pos, count in enemy_counts.items():
                            ex, ey = pos
                            # Calculate distance from our unit
                            dx_target = ex - unit_pos[0]
                            dy_target = ey - unit_pos[1]

                            # Check if within sap range
                            if (
                                abs(dx_target) <= sap_range
                                and abs(dy_target) <= sap_range
                            ):
                                # Calculate a score based on number of enemies and distance
                                score = count * 10 - (abs(dx_target) + abs(dy_target))

                                # Check for additional enemies in the 8 adjacent tiles (AOE effect)
                                for adj_dx in [-1, 0, 1]:
                                    for adj_dy in [-1, 0, 1]:
                                        if adj_dx == 0 and adj_dy == 0:
                                            continue

                                        adj_pos = (ex + adj_dx, ey + adj_dy)
                                        if adj_pos in enemy_counts:
                                            # Add to score with the dropoff factor
                                            score += enemy_counts[adj_pos] * 5

                                if score > best_score:
                                    best_score = score
                                    best_target = (dx_target, dy_target)

                        # If we found a good target, use it
                        if best_target is not None:
                            dx, dy = best_target
                        else:
                            # Fallback to using last movement direction
                            if self.last_move_direction is not None:
                                if self.last_move_direction == ActionType.MOVE_UP.value:
                                    dy = -sap_range
                                elif (
                                    self.last_move_direction
                                    == ActionType.MOVE_RIGHT.value
                                ):
                                    dx = sap_range
                                elif (
                                    self.last_move_direction
                                    == ActionType.MOVE_DOWN.value
                                ):
                                    dy = sap_range
                                elif (
                                    self.last_move_direction
                                    == ActionType.MOVE_LEFT.value
                                ):
                                    dx = -sap_range
                else:
                    # Fallback to using last movement direction
                    if self.last_move_direction is not None:
                        if self.last_move_direction == ActionType.MOVE_UP.value:
                            dy = -sap_range
                        elif self.last_move_direction == ActionType.MOVE_RIGHT.value:
                            dx = sap_range
                        elif self.last_move_direction == ActionType.MOVE_DOWN.value:
                            dy = sap_range
                        elif self.last_move_direction == ActionType.MOVE_LEFT.value:
                            dx = -sap_range

                env_action[player_id][idx, 1] = dx
                env_action[player_id][idx, 2] = dy

        return env_action

    def sap_reward(self, idx, obs, unit_pos, combined_action, unit_energy):
        """
        Calculate rewards for sap actions based on energy and enemy proximity.

        Args:
            idx: Index of the unit taking the action.
            obs: The current observation from the environment.
            unit_pos: The position of the unit.
            combined_action: The combined action taken by the player.
            unit_energy: The energy level of the unit.

        Returns:
            reward: The calculated reward for the sap action.
        """

        any_enemy_in_range = False
        sap_range = self.env_config["unit_sap_range"] if self.env_config else 3

        # Check if any enemy is within potential sap range
        for enemy in obs[self.player_id].units.position[1]:
            enemy_x, enemy_y = enemy[0].item(), enemy[1].item()
            # Check if enemy is in sensor range
            if obs[self.player_id].sensor_mask[enemy_y, enemy_x]:
                # Check if enemy is within reasonable sap range
                dist = abs(enemy_x - unit_pos[0]) + abs(enemy_y - unit_pos[1])
                if dist <= sap_range + 1:
                    any_enemy_in_range = True
                    break

        # Base energy check with smaller penalty
        energy_ratio = unit_energy / GameConstants.MAX_UNIT_ENERGY
        if energy_ratio < 0.3:  # Low energy so shouldn't sap
            return -1.0

        # Calculate target position
        target_x = unit_pos[0] + combined_action[self.player_id][idx, 1]
        target_y = unit_pos[1] + combined_action[self.player_id][idx, 2]

        # Check if hit enemy
        enemy_hit, enemy_count = self.check_sap_hit_enemy(obs, target_x, target_y)

        if enemy_hit:
            # Increased base reward for successful sap
            reward = 15.0

            # Bonus for multiple enemies
            if enemy_count > 1:
                reward += 6.0 * (enemy_count - 1)

            # Bonus for effective energy management
            reward += energy_ratio * 2.0

            return reward
        else:
            # If no enemy was hit, check if any were in range
            if any_enemy_in_range:
                return -1.0
            else:
                return -0.5  # Small penalty when no enemies were in range anyway

    def check_sap_hit_enemy(self, obs, target_x, target_y):
        """
        Helper function to check if sap hits enemy

        Args:
            obs: The current observation from the environment.
            target_x: The x-coordinate of the target position.
            target_y: The y-coordinate of the target position.

        Returns:
            enemy_hit: Boolean indicating if an enemy was hit.
            enemy_count: The number of enemies hit.
        """
        enemy_count = 0
        enemy_hit = False
        # Adjacent hit check
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                check_x = target_x + dx
                check_y = target_y + dy

                for enemy in obs[self.player_id].units.position[1]:
                    if enemy[0].item() == check_x and enemy[1].item() == check_y:
                        enemy_count += 1
                        enemy_hit = True

        return enemy_hit, enemy_count

    def calculate_movement_reward(
        self, act, unit_pos, unit_energy, pre_step_positions, obs, info
    ):
        """
        Calculate rewards for movement actions

        Args:
            act: The action taken by the player.
            unit_pos: The position of the unit.
            unit_energy: The energy level of the unit.
            pre_step_positions: Previous positions of the units.
            obs: The current observation from the environment.
            info: Additional information about the environment state.

        Returns:
            reward: The calculated reward for the movement action.
        """
        reward = 0.0

        if (
            ActionType.MOVE_UP.value <= act <= ActionType.MOVE_LEFT.value
        ):  # Movement actions
            # Get movement direction
            if act == ActionType.MOVE_UP.value:
                dx, dy = 0, -1
            elif act == ActionType.MOVE_RIGHT.value:
                dx, dy = 1, 0
            elif act == ActionType.MOVE_DOWN.value:
                dx, dy = 0, 1
            elif act == ActionType.MOVE_LEFT.value:
                dx, dy = -1, 0
            else:
                dx, dy = 0, 0

            # Check if we have record of previous positions
            if unit_pos in pre_step_positions:
                prev_pos = pre_step_positions[unit_pos]
                new_x, new_y = unit_pos

                # Check if the move was valid or if it hit a boundary/obstacle
                if (
                    new_x == prev_pos[0]
                    and new_y == prev_pos[1]
                    and (dx != 0 or dy != 0)
                ):
                    # Movement was blocked
                    reward -= 0.3  # Penalty for invalid moves

                # Enhanced relic proximity rewards
                for relic_idx in range(len(obs[self.player_id].relic_nodes)):
                    if obs[self.player_id].relic_nodes_mask[relic_idx] == 1:
                        relic_pos = tuple(obs[self.player_id].relic_nodes[relic_idx])

                        # Graduated rewards based on distance to relic
                        relic_dist = abs(relic_pos[0] - new_x) + abs(
                            relic_pos[1] - new_y
                        )
                        if (
                            relic_dist <= 2
                        ):  # Close to relic center (potential point zone)
                            reward += 2.0
                        elif relic_dist <= 4:  # Within reasonable exploration distance
                            reward += 0.5

                # Energy field rewards
                if (
                    0 <= new_y < GameConstants.MAP_SIZE
                    and 0 <= new_x < GameConstants.MAP_SIZE
                ):
                    # Using the energy map from observation
                    energy_val = obs[self.player_id].map_features.energy[new_y][new_x]

                    # Dynamic energy rewards based on unit's current energy level
                    if energy_val > 0:  # If there's energy at this position
                        # Higher reward for low-energy units on energy nodes
                        energy_factor = max(
                            0.1, 1.0 - (unit_energy / 400.0)
                        )  # 400 is max energy
                        reward += 0.3 * energy_factor * energy_val

                    # Enhanced nebula penalties based on parameters
                    if (
                        obs[self.player_id].map_features.tile_type[new_y][new_x] == 1
                    ):  # Nebula
                        nebula_penalty = 0.2  # Base penalty

                        # Increase penalty if we've discovered nebula costs are high
                        nebula_energy_reduction = info.get("full_params", {}).get(
                            "nebula_tile_energy_reduction", 0
                        )
                        if nebula_energy_reduction > 0:
                            nebula_penalty += 0.1 * nebula_energy_reduction

                        reward -= nebula_penalty

        return reward

    def calculate_distance_reward(self, unit_pos):
        """
        Reward units for moving away from spawn

        Args:
            unit_pos: The position of the unit.

        Returns:
            reward: Normalized distance from spawn.
        """
        dist_from_start = abs(unit_pos[0] - self.team_spawn[0]) + abs(
            unit_pos[1] - self.team_spawn[1]
        )
        normalized_dist = dist_from_start / (
            GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT
        )
        return 0.5 * normalized_dist

    def calculate_stage_reward(self, unit_pos, unit_energy):
        """
        Calculate stage-specific rewards based on game progress

        Args:
            unit_pos: The position of the unit.
            unit_energy: The energy level of the unit.

        Returns:
            reward: The calculated reward for the current stage.
        """
        reward = 0.0

        # Early game reward (first 30% of the game)
        if self.current_step < (GameConstants.MAX_STEPS_IN_MATCH * 0.3):
            # Calculate distance from enemy spawn
            dist_to_enemy_spawn = abs(unit_pos[0] - self.enemy_spawn[0]) + abs(
                unit_pos[1] - self.enemy_spawn[1]
            )
            # Normalize to 0-1
            normalized_dist = 1.0 - (
                dist_to_enemy_spawn
                / (GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT)
            )
            # Encourage movement toward enemy spawn early in game
            reward += 0.5 * (1.0 - normalized_dist)

        # Energy maintenance reward (all game stages)
        energy_ratio = unit_energy / GameConstants.MAX_UNIT_ENERGY
        reward += 0.1 * energy_ratio

        return reward

    def calculate_exploration_reward(self, unit_pos):
        """
        Reward for exploring new tiles on the map

        Args:
            unit_pos: The position of the unit.

        Returns:
            reward: The calculated reward for exploration.
        """

        if (
            0 <= unit_pos[1] < GameConstants.MAP_WIDTH
            and 0 <= unit_pos[0] < GameConstants.MAP_HEIGHT
            and not self.visited_tiles[unit_pos[1], unit_pos[0]]
        ):

            self.visited_tiles[unit_pos[1], unit_pos[0]] = True

            return 15.0

        return 0.0

    def calculate_relic_reward(self, unit_pos, points_delta):
        """
        Calulate rewards for generating points from relics

        Args:
            unit_pos: The position of the unit.
            points_delta: Change in points.

        Returns:
            reward: The calculated reward for generating points.
        """
        reward = 0.0

        # Large reward for point generation
        if points_delta > 0:
            reward += 8.0
            self.relic_points_tiles.add(unit_pos)

        # Continuing reward for staying on point tiles
        if unit_pos in self.relic_points_tiles:
            reward += 5.0

        return reward

    def calculate_relic_exploration_reward(self, unit_pos, act, obs):
        """
        Calculate reward for exploring different positions around relics

        Args:
            unit_pos: The position of the unit.
            act: The action taken by the player.
            obs: The current observation from the environment.

        Returns:
            reward: The calculated reward for exploring relics.
        """

        if act < ActionType.MOVE_UP.value or act > ActionType.MOVE_LEFT.value:
            return 0.0

        reward = 0.0

        # Find nearby relics
        for j in range(len(obs[self.player_id].relic_nodes_mask)):
            if obs[self.player_id].relic_nodes_mask[j] == 1:
                relic_pos = tuple(obs[self.player_id].relic_nodes[j])
                relic_dist = abs(relic_pos[0] - unit_pos[0]) + abs(
                    relic_pos[1] - unit_pos[1]
                )

                if relic_dist <= 3:  # Close to a relic
                    # Check if we've visited this specific position before
                    pos_key = (unit_pos[0], unit_pos[1])

                    if pos_key not in self.relic_points_tiles:
                        # New position near relic - good for exploration
                        self.relic_points_tiles.add(pos_key)
                        reward += 1.5

                        # Extra reward if very close to relic (might generate points)
                        if relic_dist <= 2:
                            reward += 1.0

        return reward

    def calculate_penalty_for_stying_in_same_area(self, unit_pos, idx):
        """
        Calculate penalty for staying in the same area for too long if not on a relic point

        Args:
            unit_pos: The position of the unit.
            idx: Index of the unit.

        Returns:
            reward: The calculated penalty for staying in the same area for too long if not on a relic point.
        """
        reward = 0.0
        if len(self.unit_position_history[idx]) > 10:  # Look at last 10 steps
            # Don't penalize staying on relic point tiles
            if (
                unit_pos not in self.relic_points_tiles
            ):  # Check if NOT on a relic point tile
                recent_positions = self.unit_position_history[idx][-10:]
                unique_positions = set(recent_positions)
                if (
                    len(unique_positions) < 3
                ):  # Less than 3 unique positions in last 10 steps
                    reward -= (
                        10.0  # Severe penalty for staying put when not on a relic point
                    )
            else:
                # Actually reward consistent control of a relic point tile
                reward += 2.0  # Positive reinforcement for maintaining position on point tiles
        return reward

    def calculate_global_exploration_reward(self, obs):
        """
        Calculate the reward for global team exploration progress

        Args:
            obs: The current observation from the environment.

        Returns:
            coverage_reward: The calculated reward for global exploration.
        """

        if self.last_obs is not None:
            previous_mask = np.array(self.last_obs[self.player_id].sensor_mask)
            current_mask = np.array(obs[self.player_id].sensor_mask)
            new_tiles_mask = current_mask & ~previous_mask
            newly_discovered = np.sum(new_tiles_mask)
            coverage_reward = 0.2 * newly_discovered
        else:
            coverage_reward = 0.0

        return coverage_reward

    def calculate_dispersion_reward(self, unit_positions):
        """
        Calculate reward for team unit dispersion (avoiding clustering)

        Args:
            unit_positions: List of positions of all units.

        Returns:
            reward: The calculated reward for unit dispersion.
        """
        if len(unit_positions) > 1:
            avg_x = sum(pos[0] for pos in unit_positions) / len(unit_positions)
            avg_y = sum(pos[1] for pos in unit_positions) / len(unit_positions)

            # Calculate average distance from centroid
            avg_dispersion = sum(
                abs(pos[0] - avg_x) + abs(pos[1] - avg_y) for pos in unit_positions
            ) / len(unit_positions)

            # Normalize by map size
            normalized_dispersion = avg_dispersion / (
                GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT
            )

            # Return dispersion reward
            return 0.5 * normalized_dispersion

        return 0.0

    def calculate_weighted_reward(self, rule_based_reward, points_reward):
        """
        Calculate the final weighted reward based on game progress

        Args:
            rule_based_reward: The reward based on game rules.
            points_reward: The reward based on points generated.

        Returns:
            final_reward: The final calculated reward.
        """
        # Get current map coverage percentage
        current_coverage = np.sum(self.visited_tiles) / (
            GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
        )

        # If map coverage is very low, make exploration the ONLY priority
        if current_coverage < 0.25:  # Less than 25% map coverage
            return rule_based_reward
        else:
            return (rule_based_reward * 0.5) + (points_reward * 0.5)

    def calculate_map_coverage_reward(self, unit_pos, act):
        """
        Reward for actions that contribute to increasing map coverage

        Args:
            unit_pos: The position of the unit.
            act: The action taken by the player.

        Returns:
            reward: The calculated reward for map coverage.
        """

        # Only reward for movement actions
        if act < ActionType.MOVE_UP.value or act > ActionType.MOVE_LEFT.value:
            return 0.0

        # Check if this position is new (unexplored)
        if (
            0 <= unit_pos[1] < GameConstants.MAP_WIDTH
            and 0 <= unit_pos[0] < GameConstants.MAP_HEIGHT
            and not self.visited_tiles[unit_pos[1], unit_pos[0]]
        ):

            # Calculate current map coverage percentage before marking this tile
            current_coverage = np.sum(self.visited_tiles) / (
                GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
            )

            if current_coverage >= 0.50 and self.last_coverage < 0.50:
                return 10.0  # Big bonus for reaching 50% exploration
            elif current_coverage >= 0.25 and self.last_coverage < 0.25:
                return 5.0  # Bonus for reaching 25% exploration
            else:
                return 1.0  # Base reward for any new tile

        return 0.0  # No reward if not exploring a new tile

    def get_exploration_direction(self, unit_pos):
        """
        Find direction to nearest unexplored frontier

        Args:
            unit_pos: The position of the unit.

        Returns:
            action: The action to be taken to explore.
        """

        frontiers = []

        # Simple frontier detection - any unexplored tile next to an explored one
        for y in range(GameConstants.MAP_HEIGHT):
            for x in range(GameConstants.MAP_WIDTH):
                if not self.visited_tiles[y][x]:
                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (
                            0 <= nx < GameConstants.MAP_WIDTH
                            and 0 <= ny < GameConstants.MAP_HEIGHT
                            and self.visited_tiles[ny][nx]
                        ):
                            frontiers.append((x, y))
                            break

        if frontiers:
            # Find closest frontier
            closest = min(
                frontiers,
                key=lambda f: abs(f[0] - unit_pos[0]) + abs(f[1] - unit_pos[1]),
            )

            # Return direction toward this frontier
            dx = closest[0] - unit_pos[0]
            dy = closest[1] - unit_pos[1]
            if abs(dx) > abs(dy):
                return (
                    ActionType.MOVE_RIGHT.value
                    if dx > 0
                    else ActionType.MOVE_LEFT.value
                )
            else:
                return (
                    ActionType.MOVE_DOWN.value if dy > 0 else ActionType.MOVE_UP.value
                )

        # If no frontiers, return random direction
        return np.random.choice(
            [
                ActionType.MOVE_UP.value,
                ActionType.MOVE_RIGHT.value,
                ActionType.MOVE_DOWN.value,
                ActionType.MOVE_LEFT.value,
            ]
        )

    def step(self, actions, models_dir=None):
        """
        Step the environment and process actions for all players.

        Args:
            actions: The actions to be taken by the players.
            models_dir: Directory containing model files (default is None).

        Returns:
            obs: The current observation from the environment.
            reward: The calculated reward for the player.
            terminated: Boolean indicating if the episode has terminated.
            truncated: Boolean indicating if the episode has been truncated.
            info: Additional information about the environment state.
        """

        self.current_step += 1
        self.tiles_visited_this_step = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )

        if self.current_step % 20 == 0 and self.energy_map is not None:
            self.energy_map = np.roll(self.energy_map, shift=1, axis=1)

        # Also shift enemy units slightly to prevent camping
        for enemy in self.enemy_units:
            new_x = enemy["x"] + 1
            if new_x >= GameConstants.MAP_WIDTH:
                new_x = GameConstants.MAP_WIDTH - 1
            enemy["x"] = new_x

        if not models_dir:
            models_dir = "../ppo_lux_model_base_all/"
            model_files = (
                [
                    os.path.join(models_dir, f)
                    for f in os.listdir(models_dir)
                    if f.endswith(".zip")
                ]
                if os.path.exists(models_dir)
                else []
            )
            if len(model_files) > 0:
                self.base_model_path = max(model_files, key=os.path.getmtime)
        else:
            self.base_model_path = models_dir

        # Process our agent's action
        player_action = self.process_action(actions, self.player_id)

        # Get opponent's action
        opponent_action = self.get_opponent_action(self.last_obs, self.last_info)

        # Combine actions for all players
        combined_action = {**player_action, **opponent_action}

        # Remember unit positions before step
        prev_obs = self.last_obs[self.player_id] if self.last_obs is not None else None
        prev_visible_tiles = (
            set(zip(*np.where(prev_obs.sensor_mask))) if prev_obs is not None else set()
        )
        pre_step_positions = {}
        pre_step_energy = {}

        if self.last_obs is not None:
            for i in range(len(self.last_obs[self.player_id].units_mask)):
                if self.last_obs[self.player_id].units_mask[i].any():
                    pre_step_positions[i] = tuple(
                        self.last_obs[self.player_id].units.position[i]
                    )
                    pre_step_energy[i] = self.last_obs[self.player_id].units.energy[i]

        # Step the environment
        obs, reward_dict, terminated_dict, truncated_dict, info = self.env.step(
            combined_action
        )

        if self.player_id in obs:
            self.cumulative_sensor_mask |= np.array(
                obs[self.player_id].sensor_mask, dtype=bool
            )

        unit_rewards = []
        sap_rewards = []
        unit_energies = []
        unit_positions = []
        current_visible_tiles = set(zip(*np.where(obs[self.player_id].sensor_mask)))
        new_tiles = current_visible_tiles - prev_visible_tiles
        exploration_reward = len(new_tiles) * 0.1

        # Extract reward for our player
        if isinstance(reward_dict, dict) and self.player_id in reward_dict:
            reward_value = reward_dict[self.player_id]
            if hasattr(reward_value, "item"):
                reward = float(reward_value.item())
            else:
                reward = float(reward_value)
        else:
            reward = 0.0

        current_total_points = (
            obs[self.player_id].team_points[0]
            if hasattr(obs[self.player_id], "team_points")
            else 0
        )
        points_delta = current_total_points - self.last_total_points
        self.last_total_points = current_total_points

        relic_control_this_turn = points_delta > 0
        if relic_control_this_turn:
            self.consecutive_relic_control += 1
        else:
            self.consecutive_relic_control = 0

        for idx, _ in enumerate(self.team_units):
            unit_reward = 0.0
            act = actions[idx]
            unit_pos = tuple(
                (
                    obs[self.player_id].units.position[0][idx][0].item(),
                    obs[self.player_id].units.position[0][idx][1].item(),
                )
            )
            unit_energy = obs[self.player_id].units.energy[0][idx]

            # Add bonus for moving toward frontiers
            if ActionType.MOVE_UP.value <= act <= ActionType.MOVE_LEFT.value:
                frontier_direction = self.get_exploration_direction(unit_pos)
                if act == frontier_direction:
                    unit_reward += 5.0

            if self.tiles_visited_this_step[unit_pos[1], unit_pos[0]]:
                unit_reward -= 5.0  # Strong penalty for revisiting during same match
            self.tiles_visited_this_step[unit_pos[1], unit_pos[0]] = True

            # Add distance from spawn reward
            unit_reward += self.calculate_distance_reward(unit_pos)

            # Add stage-based rewards (early/mid/late game specific)
            unit_reward += self.calculate_stage_reward(unit_pos, unit_energy)

            # Add exploration reward
            unit_reward += self.calculate_exploration_reward(unit_pos)

            # Add map coverage reward
            unit_reward += self.calculate_map_coverage_reward(unit_pos, act)

            # Add relic proximity and point generation reward
            unit_reward += self.calculate_relic_reward(unit_pos, points_delta)

            # Add relic exploration reward
            unit_reward += self.calculate_relic_exploration_reward(unit_pos, act, obs)

            # Add penalty for staying in the same area for too long if not at a relic point
            unit_reward += self.calculate_penalty_for_stying_in_same_area(unit_pos, idx)

            # Handle sap action
            if act == 5:  # Sap action
                sap_reward = self.sap_reward(
                    idx, obs, unit_pos, combined_action, unit_energy
                )
                sap_rewards.append(sap_reward)
            else:
                unit_reward += self.calculate_movement_reward(
                    act, unit_pos, unit_energy, pre_step_positions, obs, info
                )
                sap_rewards.append(0.0)

            unit_rewards.append(unit_reward)
            unit_energies.append(unit_energy)
            unit_positions.append(unit_pos)

        # Calculate global team rewards
        exploration_reward = self.calculate_global_exploration_reward(obs)
        dispersion_reward = self.calculate_dispersion_reward(unit_positions)

        # Sum up all components
        rule_based_reward = (
            sum(unit_rewards)
            + exploration_reward
            + dispersion_reward
            + sum(sap_rewards)
        )

        # Apply dynamic weighting based on game progress
        final_reward = self.calculate_weighted_reward(rule_based_reward, reward)

        # Extract terminated for our player
        if isinstance(terminated_dict, dict) and self.player_id in terminated_dict:
            terminated_value = terminated_dict[self.player_id]
            if hasattr(terminated_value, "item"):
                terminated = bool(terminated_value.item())
            else:
                terminated = bool(terminated_value)
        else:
            terminated = False

        # Extract truncated for our player
        if isinstance(truncated_dict, dict) and self.player_id in truncated_dict:
            truncated_value = truncated_dict[self.player_id]
            if hasattr(truncated_value, "item"):
                truncated = bool(truncated_value.item())
            else:
                truncated = bool(truncated_value)
        else:
            truncated = False

        terminated = (
            bool(terminated_dict[self.player_id].item())
            if isinstance(terminated_dict, dict)
            else False
        )
        truncated = (
            bool(truncated_dict[self.player_id].item())
            if isinstance(truncated_dict, dict)
            else False
        )

        # Log to tensorboard
        lux_metrics = {}
        lux_metrics["sap_actions_taken"] = sum(1 for a in actions if a == 5)
        lux_metrics["sap_reward"] = sum(sap_rewards)
        lux_metrics["rule_reward"] = rule_based_reward
        lux_metrics["point_reward"] = reward
        lux_metrics["final_reward"] = final_reward
        lux_metrics["relic_control_streak"] = self.consecutive_relic_control
        lux_metrics["visited_tiles_count"] = np.sum(self.visited_tiles)
        lux_metrics["relic_point_tiles_found"] = len(self.relic_points_tiles)

        # Track energy collection
        energy_before = self.last_energy if self.last_energy is not None else 0
        total_energy = sum(unit_energies)
        self.last_energy = total_energy
        energy_collected = max(0, total_energy - energy_before)
        lux_metrics["energy_collected"] = energy_collected
        lux_metrics["total_energy"] = total_energy

        self.cumulative_sensor_mask |= np.array(obs[self.player_id].sensor_mask)

        # Calculate total explored map percentage (ever seen)
        total_explored = np.sum(self.cumulative_sensor_mask)
        total_tiles = GameConstants.MAP_SIZE
        lux_metrics["map_coverage"] = (total_explored / total_tiles) * 100

        if self.last_obs is not None:
            previous_mask = np.array(self.last_obs[self.player_id].sensor_mask)
            new_tiles_mask = np.array(obs[self.player_id].sensor_mask) & ~previous_mask
            newly_discovered = np.sum(new_tiles_mask)
            lux_metrics["new_tiles_revealed"] = newly_discovered

        lux_metrics["points_earned"] = (
            obs[self.player_id].team_points[0]
            if hasattr(obs[self.player_id], "team_points")
            else 0
        )
        lux_metrics["win_rate"] = (
            obs[self.player_id].team_wins[0]
            / (obs[self.player_id].team_wins[0] + obs[self.opponent_id].team_wins[1])
            if (obs[self.player_id].team_wins[0] + obs[self.opponent_id].team_wins[1])
            > 0
            else 0
        )

        # Add metrics to info
        info["lux_metrics"] = lux_metrics
        processed_obs = self.process_observation(obs, info)

        # Remember the current observation for next step
        self.last_obs = obs
        self.last_info = info
        if self.current_step - self.last_spawn_step >= self.spawn_interval:
            if len(self.team_units) < GameConstants.MAX_UNITS:
                spawn_x, spawn_y = self.team_spawn
                self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})
                self.last_spawn_step = self.current_step

        return processed_obs, float(final_reward), terminated, truncated, info
