from luxai_s3.wrappers import LuxAIS3GymEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from stable_baselines3 import PPO
from environment import GameConstants, ActionType
from path_finding import StaticPathPlanner


def path_to_action(current_pos, next_pos):
    """
    Convert a path step to a corresponding action.

    Args:
        current_pos: Current position tuple (x, y)
        next_pos: Next position tuple (x, y)

    Returns:
        Action integer (0-4) corresponding to the move direction
    """
    # If positions are the same, do nothing
    if current_pos == next_pos:
        return ActionType.STAY.value

    dx = next_pos[0] - current_pos[0]
    dy = next_pos[1] - current_pos[1]

    # Convert direction to action
    if dx == 0 and dy == -1:
        return ActionType.MOVE_UP.value  # Move Up
    elif dx == 1 and dy == 0:
        return ActionType.MOVE_RIGHT.value  # Move Rightß
    elif dx == 0 and dy == 1:
        return ActionType.MOVE_DOWN.value  # Move Down
    elif dx == -1 and dy == 0:
        return ActionType.MOVE_LEFT.value  # Move Left
    else:
        # Default to no action if not a cardinal direction
        return ActionType.STAY.value


class SB3LuxEnvBase(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with SBX/Stable Baselines 3.
    This wrapper focuses on training a single player while allowing the opponent to be controlled
    by a different strategy. 
    """

    def __init__(
        self, env=None, player_id="player_0", opponent_strategy="random", max_units=GameConstants.MAX_UNITS
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
                "units_position": spaces.Box(
                    low=-1,
                    high=GameConstants.MAP_SIZE - 1,
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2),
                    dtype=np.int32,
                ),
                "units_energy": spaces.Box(
                    low=0,
                    high=400,
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 1),
                    dtype=np.int32,
                ),
                "units_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS),
                    dtype=np.int8,
                ),
                "sensor_mask": spaces.Box(
                    low=0,
                    high=1,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                "map_features_tile_type": spaces.Box(
                    low=-1,
                    high=2,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                "map_features_energy": spaces.Box(
                    low=-1,
                    high=GameConstants.MAX_ENERGY_PER_TILE,
                    shape=(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
                    dtype=np.int8,
                ),
                "relic_nodes_mask": spaces.Box(
                    low=0, high=1, shape=(GameConstants.MAX_RELIC_NODES,), dtype=np.int8
                ),
                "relic_nodes": spaces.Box(
                    low=-1,
                    high=GameConstants.MAP_WIDTH - 1,
                    shape=(GameConstants.MAX_RELIC_NODES, 2),
                    dtype=np.int32,
                ),
                "team_points": spaces.Box(
                    low=0, high=1000, shape=(GameConstants.NUM_TEAMS,), dtype=np.int32
                ),
                "team_wins": spaces.Box(
                    low=0, high=1000, shape=(GameConstants.NUM_TEAMS,), dtype=np.int32
                ),
                "steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "match_steps": spaces.Box(
                    low=0,
                    high=GameConstants.MAX_STEPS_IN_MATCH,
                    shape=(1,),
                    dtype=np.int32,
                ),
                "remainingOverageTime": spaces.Box(
                    low=0, high=1000, shape=(1,), dtype=np.int32
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
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "env_cfg_unit_sap_cost": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
                "env_cfg_unit_sap_range": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
            }
        )

        self.opponent_obs = None
        self.opponent_model = None
        self.opponent_model_path = None
        self.last_obs = None
        self.last_info = None
        self.last_energy = None
        self.last_move_direction = None
        self.last_action = None
        self.energy_map = None
        self.visible_map = None

        # Define action space
        self.action_space = spaces.MultiDiscrete([6] * GameConstants.MAX_UNITS)
        self.cumulative_sensor_mask = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )

        # To help with preprocessing
        self.first_reset = True

        self.visited_tiles = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        self.relic_points_tiles = set()
        self.consecutive_relic_control = 0
        self.last_total_energy = 0
        self.last_total_points = 0
        self.team_units = []
        self.enemy_units = []

        self.team_spawn = (0, 0)
        self.enemy_spawn = (GameConstants.MAP_WIDTH - 1, GameConstants.MAP_HEIGHT - 1)
        self.max_steps = GameConstants.MAX_STEPS_IN_MATCH
        self.current_step = 0

    def _get_opponent_action(self, obs=None, info=None):
        """
        Generate actions for the opponent based on the selected strategy.
        Uses a mixture of strategies to create better training.
        """

        # 1. Simple strategy mixing - choose an opponent type based on training progress
        strategy_choice = np.random.random()

        # As training progresses, gradually shift from random to self-play opponents
        if self.current_step < 10000:  # Early training
            # 80% random, 20% rule-based
            if strategy_choice < 0.8:
                return self._random_strategy()
            else:
                return self._rule_based_strategy(obs)
        elif self.current_step < 50000:  # Mid training
            # 40% random, 40% rule-based, 20% previous model
            if strategy_choice < 0.4:
                return self._random_strategy()
            elif strategy_choice < 0.8:
                return self._rule_based_strategy(obs)
            else:
                return self._model_based_strategy(obs, info, use_latest=False)
        else:  # Later training
            # 20% random, 30% rule-based, 50% previous models
            if strategy_choice < 0.2:
                return self._random_strategy()
            elif strategy_choice < 0.5:
                return self._rule_based_strategy(obs)
            else:
                return self._model_based_strategy(
                    obs, info, use_latest=(strategy_choice > 0.8)
                )

    def _random_strategy(self):
        """Simple random action strategy"""

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
                dx = np.random.randint(-3, 4)
                dy = np.random.randint(-3, 4)
                env_action[self.opponent_id][idx, 1] = dx
                env_action[self.opponent_id][idx, 2] = dy

        return env_action

    def _rule_based_strategy(self, obs):
        """A simple rule-based strategy that's better than random"""
        if obs is None:
            return self._random_strategy()

        action = {self.opponent_id: np.zeros((16, 3), dtype=np.int16)}

        # Get opponent unit positions and masks
        unit_mask = np.array(obs[self.opponent_id].units_mask)
        available_units = np.where(unit_mask)[0]

        for unit_id in available_units:
            unit_pos = obs[self.opponent_id].units.position[1][unit_id]

            # Simple behavior: look for energy or move toward relic nodes
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
                    direction = 0  # S tay
                    if relic_pos[0] > unit_pos[0]:
                        direction = 2  # Move right
                    elif relic_pos[0] < unit_pos[0]:
                        direction = 4  # Move left

                    if relic_pos[1] > unit_pos[1]:
                        direction = 3  # Move down
                    elif relic_pos[1] < unit_pos[1]:
                        direction = 1  # Move up

                    # Set action to move
                    action[self.opponent_id][unit_id, 0] = direction
                    action[self.opponent_id][unit_id, 1] = 0
                    action[self.opponent_id][unit_id, 2] = 0
                    continue

            # Otherwise move randomly
            random_action = np.random.randint(0, 5)  # 0-4 for basic actions

            if random_action == 0:  # Stay
                action[self.opponent_id][unit_id, 0] = 0
            else:  # Move in a direction
                action[self.opponent_id][unit_id, 0] = random_action  # Move action
                action[self.opponent_id][unit_id, 1] = 0
                action[self.opponent_id][unit_id, 2] = 0
                if random_action == 5:  # SAP
                    sap_range = obs[self.opponent_id].env_cfg_unit_sap_cost
                    dx, dy = 0, 0

                    # Target based on last movement direction if available
                    if self.last_move_direction is not None:
                        if self.last_move_direction == 1:  # Up
                            dy = -sap_range
                        elif self.last_move_direction == 2:  # Right
                            dx = sap_range
                        elif self.last_move_direction == 3:  # Down
                            dy = sap_range
                        elif self.last_move_direction == 4:  # Left
                            dx = -sap_range

                    action[self.opponent_id][unit_id, 1] = dx
                    action[self.opponent_id][unit_id, 2] = dy

        return action

    def _model_based_strategy(self, obs, info, use_latest=True):
        """Use a previous model checkpoint as the opponent"""
        models_dir = "./ppo_lux_model_base/"
        model_files = (
            [
                os.path.join(models_dir, f)
                for f in os.listdir(models_dir)
                if f.endswith(".zip")
            ]
            if os.path.exists(models_dir)
            else []
        )

        if not model_files or obs is None or info is None:
            return self._random_strategy()

        try:
            # Sort models by modification time
            sorted_models = sorted(model_files, key=os.path.getmtime)

            if use_latest:
                # Use the latest model
                model_path = sorted_models[-1]
            else:
                # Use a random older model (not the very latest)
                older_models = (
                    sorted_models[:-1] if len(sorted_models) > 1 else sorted_models
                )
                model_path = np.random.choice(older_models)

            # Only reload if it's a different model
            if (
                not hasattr(self, "opponent_model_path")
                or self.opponent_model_path != model_path
            ):
                self.opponent_model = PPO.load(model_path)
                self.opponent_model_path = model_path

            # Get action from model
            opponent_action, _ = self.opponent_model.predict(
                self._process_observation(obs, info), deterministic=False
            )

            return self._process_action(opponent_action, self.opponent_id)
        except Exception as e:
            print(f"Error loading or using model: {e}")
            return self._random_strategy()

    def _find_resource_targets(self, obs):
        """
        Find resource targets (energy nodes and relic nodes) for units based on visibility.
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
                        # Basic value formula: energy value / distance
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
                            # Relic nodes are generally more valuable than energy nodes
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

    def _process_observation(self, obs, info, remining_overage_time=60):
        """
        Process observation into a dictionary format compatible with Dict observation space.
        """
        self.opponent_obs = obs[self.opponent_id]
        obs = obs[self.player_id]
        # Store visible map for A* pathfinding (still useful for internal logic)
        self.visible_map = np.array(obs.sensor_mask)

        # Initialize Dict observation
        processed_obs = {}

        # Define constants if not already defined elsewhere in your class
        SPACE_SIZE = obs.map_features.energy.shape[0]  # Assuming square map
        MAX_RELIC_NODES = GameConstants.MAX_RELIC_NODES
        num_tiles = SPACE_SIZE * SPACE_SIZE
        # Basic unit information
        units_position = np.full(
            (GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2), -1, dtype=np.int32
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

        # If this information is in info, use it; otherwise use defaults
        if "full_params" in info:
            max_steps = info["full_params"]["max_steps_in_match"]
            unit_move_cost = info["full_params"]["unit_move_cost"]
            unit_sap_cost = info["full_params"]["unit_sap_cost"]
            unit_sap_range = info["full_params"]["unit_sap_range"]
            map_width = info["full_params"]["map_width"]
            map_height = info["full_params"]["map_height"]
        else:
            # Defaults
            max_steps = GameConstants.MAX_STEPS_IN_MATCH
            unit_move_cost = 1
            unit_sap_cost = 5
            unit_sap_range = 2
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

        return processed_obs

    def _process_action(self, actions, player_id="player_0"):
        """
        Convert a simple Discrete action to the Dict action expected by the environment.
        Action mapping:
        0: Do nothing
        1: Move up
        2: Move right
        3: Move down
        4: Move left
        5: Sap with fixed targeting (can be improved later)
        """
        # Initialize actions for all units
        env_action = {
            player_id: np.zeros((GameConstants.MAX_UNITS, 3), dtype=np.int16),
        }

        for idx, action in enumerate(actions):

            if 0 <= action <= 4:  # Do nothing or move actions
                # Direct mapping to action type
                env_action[player_id][idx, 0] = action
                env_action[player_id][idx, 1] = 0
                env_action[player_id][idx, 2] = 0

                # Remember movement direction for smarter sap targeting
                if 1 <= action <= 4:
                    self.last_move_direction = action

            elif action == 5:  # Sap action
                env_action[player_id][idx, 0] = 5  # Sap action type
                # Default to sapping in the direction of last movement
                sap_range = 3
                dx, dy = 0, 0

                # Target based on last movement direction if available
                if self.last_move_direction is not None:
                    if self.last_move_direction == 1:  # Up
                        dy = -sap_range
                    elif self.last_move_direction == 2:  # Right
                        dx = sap_range
                    elif self.last_move_direction == 3:  # Down
                        dy = sap_range
                    elif self.last_move_direction == 4:  # Left
                        dx = -sap_range

                env_action[player_id][idx, 1] = dx
                env_action[player_id][idx, 2] = dy

        return env_action

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        self.team_units = []
        spawn_x, spawn_y = self.team_spawn
        self.team_units.append({"x": spawn_x, "y": spawn_y, "energy": 100})

        self.enemy_units = []
        spawn_x_e, spawn_y_e = self.enemy_spawn
        self.enemy_units.append({"x": spawn_x_e, "y": spawn_y_e, "energy": 100})

        processed_obs = self._process_observation(obs, info)

        return processed_obs, info

    def step(self, actions):
        self.last_action = actions
        self.current_step += 1

        # Process our agent's action
        player_action = self._process_action(actions, self.player_id)

        # Get opponent's action
        opponent_action = self._get_opponent_action(self.last_obs, self.last_info)

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

            unit_reward = 0.0
            sap_reward = 0.0
            unit_pos = tuple(
                (
                    obs[self.player_id].units.position[0][idx][0].item(),
                    obs[self.player_id].units.position[0][idx][1].item(),
                )
            )
            unit_energy = obs[self.player_id].units.energy[0][idx]

            # Mark this tile as visited for exploration tracking
            if (
                0 <= unit_pos[1] < GameConstants.MAP_WIDTH
                and 0 <= unit_pos[0] < GameConstants.MAP_HEIGHT
            ):
                if not self.visited_tiles[unit_pos[1], unit_pos[0]] == True:
                    normalized_distance_to_center = (
                        abs(unit_pos[0] - GameConstants.MAP_WIDTH / 2)
                        + abs(unit_pos[1] - GameConstants.MAP_HEIGHT / 2)
                    ) / (GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT)
                    self.visited_tiles[unit_pos[1], unit_pos[0]] = True
                    unit_reward += 0.5 * (
                        1 - normalized_distance_to_center
                    )  # Reward for exploring new tiles

            # Check for points generation - if on a relic tile that generated points
            if points_delta > 0:
                # Check if this unit is near a relic node
                near_relic = False
                for j in range(len(obs[self.player_id].relic_nodes_mask)):
                    if obs[self.player_id].relic_nodes_mask[j] == 1:
                        relic_pos = tuple(obs[self.player_id].relic_nodes[j])
                        if (
                            abs(relic_pos[0] - unit_pos[0]) <= 2
                            and abs(relic_pos[1] - unit_pos[1]) <= 2
                        ):
                            near_relic = True
                            # Add significant reward for being on a point-generating tile
                            unit_reward += 5.0
                            # Remember this as a point-generating tile
                            self.relic_points_tiles.add(unit_pos)
                            break

                # If not near a relic but team got points, give small team reward
                if not near_relic:
                    unit_reward += 0.5
            # Bonus for being on a previously discovered point-generating tile
            elif unit_pos in self.relic_points_tiles:
                unit_reward += 2.0

            # Handle sap action
            if act == 5:  # Sap action
                # Check if relic is visible in local observation
                relic_visible = False
                for j in range(len(obs[self.player_id].relic_nodes_mask)):
                    if obs[self.player_id].relic_nodes_mask[j] == True:
                        relic_visible = True
                        break

                if relic_visible:
                    # Count enemy units in 8-neighborhood
                    enemy_count = 0
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dx == 0 and dy == 0:
                                continue
                            nx = unit_pos[0] + dx
                            ny = unit_pos[1] + dy

                            # Check bounds
                            if not (
                                0 <= nx < GameConstants.MAP_SIZE
                                and 0 <= ny < GameConstants.MAP_SIZE
                            ):
                                continue

                            # Check for enemy units at this position
                            for enemy in self.enemy_units:
                                if enemy["x"] == nx and enemy["y"] == ny:
                                    enemy_count += 1

                    # Enhanced reward based on enemy count and energy
                    if enemy_count >= 2:
                        sap_reward += 1.0 * enemy_count  # Base reward for multiple enemies

                        # Additional reward based on potential energy damage
                        sap_cost = (
                            info["full_params"]["unit_sap_cost"] or 51
                        )  # Default if unknown
                        sap_cost = max(1, sap_cost)  # Avoid division by zero
                        potential_damage = max(1, enemy_count * sap_cost)
                        sap_reward += 0.05 * potential_damage
                    else:
                        # Reduced penalty if at least one enemy is nearby
                        sap_reward -= 1.0 if enemy_count == 0 else 0.5
                else:
                    sap_reward -= 1.0  # Higher penalty for sapping with no relic visible
            else:
                # Handle movement actions
                if 1 <= act <= 4:  # Movement actions
                    # Get movement direction
                    if act == 1:  # Up
                        dx, dy = 0, -1
                    elif act == 2:  # Right
                        dx, dy = 1, 0
                    elif act == 3:  # Down
                        dx, dy = 0, 1
                    elif act == 4:  # Left
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
                            # Movement was blocked (position didn't change despite action)
                            unit_reward -= 0.3  # Increased penalty for invalid moves

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
                                    unit_reward += 2.0
                                elif relic_dist <= 4:  # Within reasonable exploration distance
                                    unit_reward += 0.5

                        # Energy field rewards - with energy level considerations
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
                                unit_reward += 0.3 * energy_factor * energy_val

                            # Enhanced nebula penalties based on parameters
                            if (
                                obs[self.player_id].map_features.tile_type[new_y][new_x] == 1
                            ):  # Nebula
                                nebula_penalty = 0.2  # Base penalty

                                # Increase penalty if we've discovered nebula costs are high
                                nebula_energy_reduction = info["full_params"][
                                    "nebula_tile_energy_reduction"
                                ]
                                if (
                                    nebula_energy_reduction is not None
                                    and nebula_energy_reduction > 0
                                ):
                                    nebula_penalty += 0.1 * nebula_energy_reduction

                                unit_reward -= nebula_penalty

                        # Energy void field tactical rewards
                        # Reward for positioning near weaker enemy units (to drain them with energy void)
                        enemy_affected_count = 0
                        for dy_void in [-1, 0, 1]:
                            for dx_void in [-1, 0, 1]:
                                if abs(dx_void) + abs(dy_void) != 1:  # Cardinals only
                                    continue

                                check_x = new_x + dx_void
                                check_y = new_y + dy_void

                                if not (
                                    0 <= check_x < GameConstants.MAP_SIZE
                                    and 0 <= check_y < GameConstants.MAP_SIZE
                                ):
                                    continue

                                for enemy in self.enemy_units:
                                    if enemy["x"] == nx and enemy["y"] == ny:
                                        enemy_count += 1
                                        enemy_pos = (enemy["x"], enemy["y"])
                                        enemy_energy = enemy["energy"]

                                        if enemy_pos[0] == check_x and enemy_pos[1] == check_y:
                                            if unit_energy > enemy_energy:
                                                enemy_affected_count += 1
                                                # Reward based on energy advantage
                                                energy_void_factor = (
                                                    info["full_params"][
                                                        "unit_energy_void_factor"
                                                    ]
                                                    or 0.125
                                                )  # Default
                                                potential_void_damage = (
                                                    unit_energy * energy_void_factor
                                                )
                                                unit_reward += (
                                                    0.2 * potential_void_damage / 100.0
                                                )  # Normalize

                        # Positional rewards - avoid being too close to friendly units (vulnerability to sap)
                        friendly_nearby = 0
                        for friend_idx in range(len(obs[self.player_id].units_mask)):
                            if friend_idx != i and obs[self.player_id].units_mask[friend_idx].any():
                                friend_pos = tuple(obs[self.player_id].units.position[friend_idx])
                                dist = abs(friend_pos[0] - new_x) + abs(friend_pos[1] - new_y)
                                if 1 <= dist <= 2:  # Close but not on same tile
                                    friendly_nearby += 1

                        # Small penalty for having too many units close together (sap vulnerability)
                        if friendly_nearby >= 3:
                            unit_reward -= 0.2 * (friendly_nearby - 2)

                    # Add refined exploration reward for discovering new tiles
                    if self.last_obs is not None:
                        previous_mask = np.array(self.last_obs[self.player_id].sensor_mask)
                        current_mask = np.array(obs[self.player_id].sensor_mask)
                        new_tiles_mask = current_mask & ~previous_mask
                        newly_discovered = np.sum(new_tiles_mask)

                        # Progressive exploration reward based on match stage
                        match_progress = self.current_step / GameConstants.MAX_STEPS_IN_MATCH
                        # Higher reward early in match, lower later
                        exploration_factor = max(0.1, 1.0 - match_progress)
                        unit_reward += 0.3 * exploration_factor * newly_discovered

            # Long-term planning reward: consecutive relic control bonus
            if self.consecutive_relic_control > 0 and unit_pos in self.relic_points_tiles:
                streak_bonus = min(self.consecutive_relic_control * 0.1, 1.0)  # Cap at +1.0
                unit_reward += streak_bonus


            sap_rewards.append(sap_reward)
            unit_rewards.append(unit_reward)
            unit_energies.append(unit_energy)
            unit_positions.append(unit_pos)

        # Calculate final reward
        rule_based_reward = sum(unit_rewards) + exploration_reward + sum(sap_rewards)

        # Dynamic reward weighting based on match progress
        match_progress = self.current_step / GameConstants.MAX_STEPS_IN_MATCH
        # Start with more emphasis on exploration, gradually shift to point accumulation
        point_weight = 0.3 + (0.4 * match_progress)  # 0.3 to 0.7
        rule_weight = 1.0 - point_weight  # 0.7 to 0.3

        final_reward = (reward * point_weight) + (rule_based_reward * rule_weight)

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

        # reward = float(reward_dict[self.player_id].item()) if isinstance(reward_dict, dict) else 0.0
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

        lux_metrics = {}
        lux_metrics["sap_actions_taken"] = sum(1 for a in actions if a == 5)
        lux_metrics["sap_reward"] = sum(sap_rewards)
        lux_metrics["rule_reward"] = rule_based_reward
        lux_metrics["point_reward"] = reward
        lux_metrics["final_reward"] = final_reward
        lux_metrics["relic_control_streak"] = self.consecutive_relic_control
        lux_metrics["visited_tiles_count"] = np.sum(self.visited_tiles == True)
        lux_metrics["relic_point_tiles_found"] = len(self.relic_points_tiles)
        lux_metrics["unit_positions"] = unit_positions

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
        total_tiles = GameConstants.MAP_SIZE * GameConstants.MAP_SIZE
        lux_metrics["map_coverage"] = (total_explored / total_tiles) * 100

        if self.last_obs is not None:
            previous_mask = np.array(self.last_obs[self.player_id].sensor_mask == True)
            new_tiles_mask = np.array(obs[self.player_id].sensor_mask) & ~previous_mask
            newly_discovered = np.sum(new_tiles_mask)
            lux_metrics["new_tiles_revealed"] = newly_discovered

        # Track pathfinding success
        if hasattr(self, "paths_started") and hasattr(self, "paths_completed"):
            if self.paths_started > 0:
                lux_metrics["path_completion_rate"] = (
                    self.paths_completed / self.paths_started
                )

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
        processed_obs = self._process_observation(obs, info)

        # Remember the current observation for next step
        self.last_obs = obs
        self.last_info = info
        return processed_obs, final_reward, terminated, truncated, info
    
class SB3LuxEnvStaticPlanner(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with SB3/Stable Baselines 3.
    This wrapper includes a static pathfinding algorithm for navigation.
    """
    
    def __init__(
        self, 
        env=None, 
        player_id="player_0", 
        opponent_strategy="random", 
        max_units=GameConstants.MAX_UNITS,
        replan_interval=100  # How often to replan paths (in steps)
    ):
        if env is None:
            env = LuxAIS3GymEnv()
        super().__init__(env)
        
        # Initialize the base wrapper
        self.base_wrapper = SB3LuxEnvBase(env, player_id, opponent_strategy, max_units)
        
        # Initialize the path planner
        self.path_planner = StaticPathPlanner()
        
        # Store parameters
        self.player_id = player_id
        self.replan_interval = replan_interval
        self.last_replan_step = -1
        
        # Define observation and action spaces from base wrapper
        self.observation_space = self.base_wrapper.observation_space
        self.action_space = self.base_wrapper.action_space
        
    def reset(self, **kwargs):
        """Reset the environment and replan paths."""
        obs, info = self.base_wrapper.reset(**kwargs)
        
        # Reset planning variables
        self.last_replan_step = -1
        
        # Reset path planner state
        self.path_planner.paths = {}  # Clear cached paths
        self.path_planner.targets = {}  # Clear targets
        
        # You might also want to initialize the cost map with the first observation
        self.path_planner.astar.update_cost_map(obs)
        
        # Immediately plan initial paths
        targets = self.path_planner.find_targets_for_units(obs, self.player_id)
        self.path_planner.compute_paths_for_all_units(obs, self.player_id, targets)
        
        return obs, info
    
    def step(self, actions):
        """
        Step the environment and use static path planning for unit movement.
        
        Args:
            actions: Actions from the agent, where action 5 represents a sap action
                    
        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get the current observation
        current_obs = self.base_wrapper.last_obs 
        
        if current_obs is not None:
            processed_obs = self.base_wrapper._process_observation(current_obs, self.base_wrapper.last_info)
            current_step = processed_obs["steps"][0]
            
            # Check if we need to replan (first step or replan interval)
            if self.last_replan_step == -1 or current_step - self.last_replan_step >= self.replan_interval:
                # Find targets for units
                targets = self.path_planner.find_targets_for_units(processed_obs, self.player_id)
                
                # Compute paths for all units
                self.path_planner.compute_paths_for_all_units(processed_obs, self.player_id, targets)
                
                # Update last replan step
                self.last_replan_step = current_step
            
            # Get the next actions for all units based on their paths
            path_actions = self.path_planner.get_next_actions(processed_obs, self.player_id)
            
            # Override with sap actions from the RL agent when appropriate
            # Here we assume actions is a MultiDiscrete space with 6 possible actions per unit
            unit_actions = []
            for unit_idx in range(min(len(path_actions), len(actions))):
                # Check if the RL agent wants to perform a sap action (action 5)
                if actions[unit_idx] == 5:
                    unit_actions.append(5)  # Use the sap action from the RL agent
                else:
                    unit_actions.append(path_actions[unit_idx])  # Use the path action
            
            # Pad actions if needed
            while len(unit_actions) < GameConstants.MAX_UNITS:
                unit_actions.append(ActionType.STAY.value)
            
            # Step the environment with the computed actions
            obs, reward, terminated, truncated, info = self.base_wrapper.step(unit_actions)
            
        else:
            # If no observation is available, just step the environment with the original actions
            obs, reward, terminated, truncated, info = self.base_wrapper.step(actions)
        
        return obs, reward, terminated, truncated, info
    
class SB3LuxEnvMAPPOHR(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with SBX/Stable Baselines 3.
    This wrapper focuses on training a single player while allowing the opponent to be controlled
    by a different strategy. 
    
    This wrapper includes a MAPPOHR implmenentation described in https://arxiv.org/pdf/2306.01270
    """
    
    # TODO: Implement MAPPOHR based on SB3LuxEnvBase.