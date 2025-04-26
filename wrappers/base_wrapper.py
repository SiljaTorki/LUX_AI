import numpy as np
import os
import gymnasium as gym
from gymnasium import spaces

from luxai_s3.wrappers import LuxAIS3GymEnv
from stable_baselines3 import PPO
from common.environment import GameConstants, ActionType

class SB3LuxEnvBase(gym.Wrapper):
    """
    A wrapper for the Lux S3 environment to make it compatible with SBX/Stable Baselines 3.
    This wrapper focuses on training a single player while allowing the opponent to be controlled
    by a different strategy.
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
        self.prev_positions = None

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
        self.base_model_path = None
        self.env_config = {
            "unit_move_cost": 1,
            "unit_sensor_range": 1,
            "nebula_tile_vision_reduction": 8,
            "nebula_tile_energy_reduction": 25,
            "unit_sap_cost": 51,
            "unit_sap_range": 3
        }

    def get_opponent_action(self, obs=None, info=None):
        """
        Generate actions for the opponent based on the selected strategy.
        Uses a mixture of strategies to create better training.
        """

        # 1. Simple strategy mixing - choose an opponent type based on training progress
        strategy_choice = np.random.random()

        # As training progresses, gradually shift from random to self-play opponents
        if self.base_model_path is None or (self.base_model_path is not None and not os.path.exists(
            self.base_model_path
        )):  # Early training
            # 80% random, 20% rule-based
            if strategy_choice < 0.8:
                return self.random_strategy()
            else:
                return self.rule_based_strategy(obs)
        elif self.current_step < 100000:  # Mid training
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

    def rule_based_strategy(self, obs):
        """A simple rule-based strategy that's better than random"""
        if obs is None:
            return self.random_strategy()

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

    def model_based_strategy(self, obs, info):
        """Use a previous model checkpoint as the opponent"""
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

    def process_observation(self, obs, info, remining_overage_time=60):
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
        defalt_pos = 0 if self.player_id == "player_0" else GameConstants.MAP_SIZE - 1
        # Basic unit information
        units_position = np.full(
            (GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2), defalt_pos, dtype=np.int32
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
            unit_sensor_range = info["full_params"]["unit_sensor_range"]
            map_width = info["full_params"]["map_width"]
            map_height = info["full_params"]["map_height"]
            nebula_tile_vision_reduction = info["full_params"]["nebula_tile_vision_reduction"]
            nebula_tile_energy_reduction = info["full_params"]["nebula_tile_energy_reduction"]
        else:
            # Defaults
            max_steps = GameConstants.MAX_STEPS_IN_MATCH
            unit_move_cost = 6
            unit_sap_cost = 51
            unit_sap_range = 3
            unit_sensor_range = 1
            nebula_tile_vision_reduction = 8
            nebula_tile_energy_reduction = 25
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
        """
        # Initialize actions for all units
        env_action = {
            player_id: np.zeros((GameConstants.MAX_UNITS, 3), dtype=np.int16),
        }

        for idx, action in enumerate(actions):

            if ActionType.MOVE_UP.value <= action <= ActionType.MOVE_LEFT.value:  # Do nothing or move actions
                # Direct mapping to action type
                env_action[player_id][idx, 0] = action
                env_action[player_id][idx, 1] = 0
                env_action[player_id][idx, 2] = 0

                self.last_move_direction = action

            elif action == ActionType.SAP.value:  # Sap action
                env_action[player_id][idx, 0] = 5  # Sap action type
                # Default to sapping in the direction of last movement
                # Get the sap range from parameters
                sap_range = obs.env_cfg_unit_sap_range[0] if obs is not None else 3
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
                            if  obs[player_id].units_mask[idx]:
                                enemy_pos =  obs[player_id].units.position[enemy_idx][idx]
                                
                                # Check if enemy is visible (within sensor range)
                                ex, ey = enemy_pos[0], enemy_pos[1]
                                if 0 <= ex < obs[player_id].sensor_mask.shape[1] and 0 <= ey < obs[player_id].sensor_mask.shape[0]:
                                    if obs[player_id].sensor_mask[ey][ex]:
                                        enemy_positions.append((enemy_pos[0], enemy_pos[1]))
                        
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
                            if abs(dx_target) <= sap_range and abs(dy_target) <= sap_range:
                                # Calculate a score based on number of enemies and distance
                                # Prioritize stacked enemies
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
                                elif self.last_move_direction == ActionType.MOVE_RIGHT.value:  # Right
                                    dx = sap_range
                                elif self.last_move_direction == ActionType.MOVE_DOWN.value:  # Down
                                    dy = sap_range
                                elif self.last_move_direction == ActionType.MOVE_LEFT.value:  # Left
                                    dx = -sap_range
                else:
                    # Fallback to using last movement direction
                    if self.last_move_direction is not None:
                        if self.last_move_direction == ActionType.MOVE_UP.value:  # Up
                            dy = -sap_range
                        elif self.last_move_direction == ActionType.MOVE_RIGHT.value:  # Right
                            dx = sap_range
                        elif self.last_move_direction == ActionType.MOVE_DOWN.value:  # Down
                            dy = sap_range
                        elif self.last_move_direction == ActionType.MOVE_LEFT.value:  # Left
                            dx = -sap_range

                env_action[player_id][idx, 1] = dx
                env_action[player_id][idx, 2] = dy

        return env_action

    def sap_reward(self, idx, obs, unit_pos, combined_action, unit_energy):
        any_enemy_in_range = False
        sap_range = self.env_config["unit_sap_range"]
        
        # Check if any enemy is within potential sap range
        for enemy in obs[self.player_id].units.position[1]:
            enemy_x, enemy_y = enemy[0].item(), enemy[1].item()
            # Check if enemy is in sensor range
            if obs[self.player_id].sensor_mask[enemy_y, enemy_x]:
                # Check if enemy is within reasonable sap range
                dist = abs(enemy_x - unit_pos[0]) + abs(enemy_y - unit_pos[1])
                if dist <= sap_range + 1:  # +1 for potential movement
                    any_enemy_in_range = True
                    break
        
        # Base energy check with smaller penalty
        energy_ratio = unit_energy / GameConstants.MAX_UNIT_ENERGY  
        if energy_ratio < 0.3:  # Low energy
            return -1.0  # Reduced penalty - we still want to discourage but not severely
        
        # Calculate target position
        target_x = unit_pos[0] + combined_action[self.player_id][idx, 1]
        target_y = unit_pos[1] + combined_action[self.player_id][idx, 2]
        
        # Check if hit enemy
        enemy_hit, enemy_count = self.check_sap_hit_enemy(obs, target_x, target_y)
        
        if enemy_hit:
            # Increased base reward for successful sap
            reward = 5.0  
            
            # Bonus for multiple enemies (strong positive reinforcement)
            if enemy_count > 1:
                reward += 3.0 * (enemy_count - 1)
            
            # Bonus for effective energy management (higher reward when your energy is higher)
            reward += energy_ratio * 2.0
            
            return reward
        else:
            # Only penalize misses if there were enemies nearby to hit
            # This way we don't discourage experimentation with sap
            if any_enemy_in_range:
                return -1.0  # Reduced miss penalty
            else:
                return -0.5  # Very small penalty when no enemies were in range anyway

    def check_sap_hit_enemy(self, obs, target_x, target_y):
        """Helper function to check if sap hits enemy"""
        enemy_count = 0
        enemy_hit = False
        # Adjacent hit check (AOE)
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
        """Calculate rewards for movement actions"""
        reward = 0.0

        if ActionType.MOVE_UP.value <= act <= ActionType.MOVE_LEFT.value:  # Movement actions
            # Get movement direction
            if act == ActionType.MOVE_UP.value:  # Up
                dx, dy = 0, -1
            elif act == ActionType.MOVE_RIGHT.value:  # Right
                dx, dy = 1, 0
            elif act == ActionType.MOVE_DOWN.value:  # Down
                dx, dy = 0, 1
            elif act == ActionType.MOVE_LEFT.value:  # Left
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
                    reward -= 0.3  # Increased penalty for invalid moves

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
        """Reward units for moving away from spawn"""
        dist_from_start = abs(unit_pos[0] - self.team_spawn[0]) + abs(
            unit_pos[1] - self.team_spawn[1]
        )
        normalized_dist = dist_from_start / (
            GameConstants.MAP_WIDTH + GameConstants.MAP_HEIGHT
        )
        return 0.5 * normalized_dist

    def calculate_stage_reward(self, unit_pos, unit_energy):
        """Calculate stage-specific rewards based on game progress"""
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
        """Reward for exploring new tiles"""
        reward = 0.0
        # Check if position is valid and not visited before
        if (0 <= unit_pos[1] < GameConstants.MAP_WIDTH and 
            0 <= unit_pos[0] < GameConstants.MAP_HEIGHT):
            
            # Much stronger reward for visiting new tiles
            if not self.visited_tiles[unit_pos[1], unit_pos[0]]:
                self.visited_tiles[unit_pos[1], unit_pos[0]] = True
                
                # Base exploration reward
                reward += 4.0  # Doubled from 2.0
                
                # Additional reward based on distance from already explored areas
                # Find distance to nearest explored tile
                max_dist = 1  # Start with minimum distance
                for dy in range(-5, 6):  # Check in a 11x11 area
                    for dx in range(-5, 6):
                        check_y = unit_pos[1] + dy
                        check_x = unit_pos[0] + dx
                        
                        if (0 <= check_y < GameConstants.MAP_HEIGHT and 
                            0 <= check_x < GameConstants.MAP_WIDTH and
                            (check_y != unit_pos[1] or check_x != unit_pos[0])):
                            
                            if self.visited_tiles[check_y, check_x]:
                                # Found an explored tile, calculate Manhattan distance
                                dist = abs(dx) + abs(dy)
                                max_dist = max(dist, max_dist)
                
                # Reward for exploring further from already explored areas
                reward += 0.5 * max_dist  # More reward for exploring further out

        return reward

    def calculate_relic_reward(self, idx, unit_pos, points_delta, obs, act):
        """Calculate rewards related to relics and point generation"""
        reward = 0.0
    
        # Check if this unit is near a relic node
        near_relic = False
        for j in range(len(obs[self.player_id].relic_nodes_mask)):
            if obs[self.player_id].relic_nodes_mask[j] == 1:
                relic_pos = tuple(obs[self.player_id].relic_nodes[j])
                relic_dist = abs(relic_pos[0] - unit_pos[0]) + abs(relic_pos[1] - unit_pos[1])
                
                if relic_dist <= 2:  # Within potential point-generating range
                    near_relic = True
                    
                    # Points were generated this step - strong positive reinforcement
                    if points_delta > 0:
                        reward += 8.0  # Significant reward
                        self.relic_points_tiles.add(unit_pos)  # Remember this position
                        
                        # Extra reward for discovering a new point tile
                        if unit_pos not in self.relic_points_tiles:
                            reward += 10.0  # Extra reward for discovery
                    
                    # No points yet, but encourage exploring different positions near the relic
                    # (but only if we haven't found point-generating tiles yet)
                    elif len(self.relic_points_tiles) == 0 and act >= 1 and act <= 4:  # Moving
                        reward += 1.0  # Modest reward for exploring relic vicinity
                    
                    break
        
        # If we're on a known point-generating tile - strong reward regardless of points this step
        if unit_pos in self.relic_points_tiles:
            reward += 3.0  # Good reward for staying on known point tiles
            
        if near_relic and unit_pos in self.prev_positions and points_delta == 0:
            # Been in this position before and not generating points
            reward -= 0.5  # Small penalty to encourage movement to find point tiles
            
        return reward

    def calculate_relic_exploration_reward(self, unit_pos, act, obs):
        """Reward for exploring different positions around relics"""
        if act < ActionType.MOVE_UP.value or act > ActionType.MOVE_LEFT.value:  # Not a movement action
            return 0.0
            
        reward = 0.0
        
        # Find nearby relics
        for j in range(len(obs[self.player_id].relic_nodes_mask)):
            if obs[self.player_id].relic_nodes_mask[j] == 1:
                relic_pos = tuple(obs[self.player_id].relic_nodes[j])
                relic_dist = abs(relic_pos[0] - unit_pos[0]) + abs(relic_pos[1] - unit_pos[1])
                
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
    def calculate_global_exploration_reward(self):
        """Calculate the reward for global team exploration progress"""
        current_coverage = np.sum(self.cumulative_sensor_mask) / (
            GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
        )

        if self.last_obs is not None:
            previous_coverage = np.sum(self.last_obs[self.player_id].sensor_mask) / (
                GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT
            )
            coverage_reward = 5.0 * (
                current_coverage - previous_coverage
            )  # Reward for improving coverage
        else:
            coverage_reward = 0.0

        return coverage_reward

    def calculate_dispersion_reward(self, unit_positions):
        """Calculate reward for team unit dispersion (avoiding clustering)"""
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
        """Calculate the final weighted reward based on game progress"""
        match_progress = self.current_step / GameConstants.MAX_STEPS_IN_MATCH

        # Get current map coverage percentage
        current_coverage = np.sum(self.visited_tiles) / (GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT)
    
        # If map coverage is very low, prioritize exploration even more
        if current_coverage < 0.15:  # Less than 15% map coverage
            point_weight = 0.1 + (0.3 * match_progress)  
            rule_weight = 1.0 - point_weight 
        else:
            # Normal progression
            point_weight = 0.3 + (0.4 * match_progress)  
            rule_weight = 1.0 - point_weight 

        return float((points_reward * point_weight) + (rule_based_reward * rule_weight))

    def calculate_map_coverage_reward(self, unit_pos, act):
        """Reward for actions that contribute to increasing map coverage"""
        # Only reward for movement actions
        if act < ActionType.MOVE_UP.value or act > ActionType.MOVE_LEFT.value:
            return 0.0
        
        # Check if this position is new (unexplored)
        if (0 <= unit_pos[1] < GameConstants.MAP_WIDTH and 
            0 <= unit_pos[0] < GameConstants.MAP_HEIGHT and
            not self.visited_tiles[unit_pos[1], unit_pos[0]]):
            
            # Calculate current map coverage percentage before marking this tile
            current_coverage = np.sum(self.visited_tiles) / (GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT)
            
            # Higher rewards for exploration when coverage is low
            if current_coverage < 0.10:
                return 3.0  # Higher reward early when map is mostly unexplored
            elif current_coverage < 0.25:
                return 2.0
            elif current_coverage < 0.50:
                return 1.0
            else:
                return 0.5  # Still some reward for late-game exploration
        
        return 0.0  # No reward if not exploring a new tile

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset all state variables
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

        # Reset exploration tracking
        self.cumulative_sensor_mask = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )
        self.visited_tiles = np.zeros(
            (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
        )

        # Reset game tracking
        self.relic_points_tiles = set()
        self.consecutive_relic_control = 0
        self.last_total_energy = 0
        self.last_total_points = 0
        self.prev_positions = set()

        # Reset step counter
        self.current_step = 0

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

    def step(self, actions, models_dir=None):
        self.last_action = actions
        self.current_step += 1
        if not models_dir:
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

            # Add distance from spawn reward
            unit_reward += self.calculate_distance_reward(unit_pos)

            # Add stage-based rewards (early/mid/late game specific)
            unit_reward += self.calculate_stage_reward(unit_pos, unit_energy)

            # Add exploration reward
            unit_reward += self.calculate_exploration_reward(unit_pos)
            
            # Add map coverage reward
            unit_reward += self.calculate_map_coverage_reward(unit_pos, act)

            # Add relic proximity and point generation reward
            unit_reward += self.calculate_relic_reward(
                idx, unit_pos, points_delta, obs, act
            )

            # Add relic exploration reward
            unit_reward += self.calculate_relic_exploration_reward(
                unit_pos, act, obs
            )

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
            self.prev_positions = set([unit_pos])

        # Calculate global team rewards
        exploration_reward = self.calculate_global_exploration_reward()
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

        lux_metrics = {}
        lux_metrics["sap_actions_taken"] = sum(1 for a in actions if a == 5)
        lux_metrics["sap_reward"] = sum(sap_rewards)
        lux_metrics["rule_reward"] = rule_based_reward
        lux_metrics["point_reward"] = reward
        lux_metrics["final_reward"] = final_reward
        lux_metrics["relic_control_streak"] = self.consecutive_relic_control
        lux_metrics["visited_tiles_count"] = np.sum(self.visited_tiles == True)
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
        total_tiles = GameConstants.MAP_SIZE * GameConstants.MAP_SIZE
        lux_metrics["map_coverage"] = (total_explored / total_tiles) * 100

        if self.last_obs is not None:
            previous_mask = np.array(self.last_obs[self.player_id].sensor_mask == True)
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
        return processed_obs, final_reward, terminated, truncated, info