import sys
import numpy as np
from environment import NodeType, ActionType, GameConstants
from a_star import a_star_search
from stable_baselines3 import PPO
import random


class Ship:
    def __init__(self, unit_id, position, energy):
        self.id = unit_id
        self.pos = tuple(position) # (x, y)
        self.energy = energy

    def get_position(self):
        return self.pos

    def get_energy(self):
        return self.energy

    def detect_obstacles(self, map_features):
        """Detect obstacles in the map."""
        return [
            (x, y)
            for x in range(GameConstants.MAP_WIDTH)
            for y in range(GameConstants.MAP_HEIGHT)
            if map_features[x, y] == NodeType.ASTEROID.value
        ]

    def detect_nebula(self, map_features):
        """Detect nebula tiles in the map."""
        return [
            (x, y)
            for x in range(GameConstants.MAP_WIDTH)
            for y in range(GameConstants.MAP_HEIGHT)
            if map_features[x, y] == NodeType.NEBULA.value
        ]

    def find_closest(self, node_type, map_features, team_vision):
        """Find closest visible node of specified type."""
        closest = None
        min_dist = float("inf")

        for x in range(GameConstants.MAP_WIDTH):
            for y in range(GameConstants.MAP_HEIGHT):
                if team_vision[x, y] and map_features[x, y] == node_type.value:
                    dist = abs(x - self.pos[0]) + abs(y - self.pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest = (x, y)

        return closest

    def get_direction(self, target):
        """Get action to move toward target position"""
        if self.pos == target:
            return ActionType.STAY.value

        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]

        # Priority: x-axis first (horizontal movement)
        if dx > 0:
            return ActionType.MOVE_RIGHT.value
        elif dx < 0:
            return ActionType.MOVE_LEFT.value
        elif dy > 0:
            return ActionType.MOVE_DOWN.value
        elif dy < 0:
            return ActionType.MOVE_UP.value
        else:
            return ActionType.STAY.value


# Class for a fleet that contorls all the ships
class Fleet:
    def __init__(self, team_id, units):
        self.team_id = team_id
        self.units = units
        self.num_units = len(units)

    def get_unit(self, unit_id):
        return self.units[unit_id]


class Agent:
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        np.random.seed(0)
        self.env_cfg = env_cfg

        self.discovered_relic_nodes_ids = set()
        self.grid = None
        self.enemy_positions = []
        self.obs = None
        self.explored_tiles = None

        try:
            self.model = PPO.load("./models_final/ppo_lux_final")
            print("Successfully loaded PPO model", file=sys.stderr)
            print(f"Model policy network: {self.model.policy}", file=sys.stderr)
        except Exception as e:
            print(
                f"Failed to load model: {e}. Using fallback strategy", file=sys.stderr
            )
            self.model = None

    def act(self, step: int, observation, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        self.process_observation(observation, remainingOverageTime)
        available_unit_ids = np.where(self.obs["units_mask"][self.team_id])[0]

        ships = [
            Ship(
                i,
                (
                    self.obs["units_position"][self.team_id][i][0],
                    self.obs["units_position"][self.team_id][i][1],
                ),
                self.obs["units_energy"][self.team_id][i][0],
            )
            for i in range(len(self.obs["units_position"][self.team_id]))
        ]
        fleet = Fleet(self.team_id, ships)
        
        try:
            # Get model prediction - the model is predicting an action (0-5)
            action, _ = self.model.predict(self.obs, deterministic=True)
            print(f"🤖 PPO model raw prediction: {action}, 🕵️: {_}", file=sys.stderr)

            # Handle scalar action value (0-5)
            if isinstance(action, (int, np.integer)) or (
                isinstance(action, np.ndarray) and action.size == 1
            ):
                action_value = int(action) if isinstance(action, np.ndarray) else action

                # Convert action to position based on the current unit position
                # This is a simple translation to demonstrate the concept
                unit_id = available_unit_ids[0] if len(available_unit_ids) > 0 else 0
                unit_pos = self.obs["units_position"][self.team_id][unit_id]

                if action_value == ActionType.STAY.value:
                    # Stay in place
                    goal_position = tuple(unit_pos)
                elif action_value == ActionType.MOVE_UP.value:
                    # Move up on the map
                    goal_position = (int(unit_pos[0]), max(0, int(unit_pos[1]) - 3))
                elif action_value == ActionType.MOVE_RIGHT.value:
                    # Move right on the map
                    goal_position = (
                        min(GameConstants.MAP_WIDTH - 1, int(unit_pos[0]) + 3),
                        int(unit_pos[1]),
                    )
                elif action_value == ActionType.MOVE_DOWN.value:
                    # Move down on the map
                    goal_position = (
                        int(unit_pos[0]),
                        min(GameConstants.MAP_HEIGHT - 1, int(unit_pos[1]) + 3),
                    )
                elif action_value == ActionType.MOVE_LEFT.value:
                    # Move left on the map
                    goal_position = (max(0, int(unit_pos[0]) - 3), int(unit_pos[1]))
                elif action_value == ActionType.SAP.value:
                    # For sap action, we still need a movement goal - go to nearest enemy
                    enemy_pos = self.find_nearest_enemy(unit_pos)
                    goal_position = enemy_pos if enemy_pos else tuple(unit_pos)

                print(
                    f"Converted action {action_value} to goal: {goal_position}",
                    file=sys.stderr,
                )
            # Handle coordinate-based prediction if your model actually produces coordinates
            elif isinstance(action, np.ndarray) and action.shape == (2,):
                goal_x = min(max(int(action[0]), 0), GameConstants.MAP_WIDTH - 1)
                goal_y = min(max(int(action[1]), 0), GameConstants.MAP_HEIGHT - 1)
                goal_position = (goal_x, goal_y)
            else:
                print(
                    f"⚠️ Unexpected prediction shape: {action.shape if hasattr(action, 'shape') else type(action)}",
                    file=sys.stderr,
                )
                goal_position = None
        except Exception as e:
            print(f"⚠️ Error in model prediction: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)
            goal_position = None

        # Fall back to rule-based goal selection if needed
        if goal_position is None:
            goal_position = self.rule_based_goals()
            print(f"Rule-based goal selection: {goal_position}", file=sys.stderr)

        strategy = self.determine_strategy(goal_position)
        unit_roles = self.determine_unit_roles(available_unit_ids)
        print(f"Strategy: {strategy}, Roles: {unit_roles}", file=sys.stderr)

        # Use A* to plan paths to the goal
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in available_unit_ids:
            ship = fleet.get_unit(unit_id)
            role = unit_roles[unit_id] if unit_id < len(unit_roles) else 0

            # Handle unit based on its role
            if role == 0:  # Explorer
                # Explorers prioritize new areas
                target = self.get_explorer_target(ship, goal_position)
            elif role == 1:  # Energy collector
                # Energy collectors prioritize energy tiles
                target = self.get_energy_target(ship)
            elif role == 2:  # Relic seeker
                # Relic seekers prioritize relic nodes
                target = self.get_relic_target(ship)
            elif role == 3:  # Attacker/defender
                # Attackers prioritize enemy units
                sap_target = self.find_best_sap_target(ship)
                if sap_target:
                    actions[unit_id] = [
                        ActionType.SAP.value,
                        sap_target[0],
                        sap_target[1],
                    ]
                    continue
                target = self.get_attacker_target(ship)
            else:
                # Default to the main goal
                target = goal_position

            print(f"Unit {unit_id} role: {role}, target: {target}", file=sys.stderr)

            # Plan path using A*
            path = a_star_search(
                ship=ship,
                goal=target,  # Use model-selected goal
                map_features=self.obs["map_features_tile_type"],
                team_vision=self.obs["sensor_mask"],
                enemy_positions=self.enemy_positions,
                step_count=step,
            )

            # Convert path to action
            if len(path) > 1:
                next_position = path[1]
                direction = ship.get_direction(next_position)
                actions[unit_id] = [direction, 0, 0]
        # print(f"🕺🏻 Actions: {actions}", file=sys.stderr)
        return actions

    def find_nearest_enemy(self, unit_pos):
        """Find the nearest visible enemy position"""
        nearest_enemy = None
        min_dist = float("inf")

        for i in range(len(self.obs["units_position"][self.opp_team_id])):
            if self.obs["units_mask"][self.opp_team_id][i] > 0:
                enemy_pos = self.obs["units_position"][self.opp_team_id][i]
                if enemy_pos[0] >= 0 and enemy_pos[1] >= 0:  # Valid position
                    dist = abs(unit_pos[0] - enemy_pos[0]) + abs(
                        unit_pos[1] - enemy_pos[1]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        nearest_enemy = tuple(int(x) for x in enemy_pos)

        return (
            nearest_enemy
            if nearest_enemy
            else (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)
        )

    def determine_strategy(self, goal_position):
        """Determine appropriate strategy based on goal position and game state"""
        # Handle case where goal_position is None or not a valid tuple
        if goal_position is None:
            return 1  # Default to exploration focus

        if not isinstance(goal_position, tuple) or len(goal_position) != 2:
            print(f"⚠️ Invalid goal position: {goal_position}", file=sys.stderr)
            return 1  # Default to exploration focus

        # Now we have a valid tuple
        x, y = goal_position

        steps = self.obs["steps"][0]
        max_steps = self.obs["match_steps"][0]
        match_progress = steps / max_steps if max_steps > 0 else 0

        # Early match or early in sequence - exploration focus
        if match_progress < 0.3 or steps < 10:
            return 1 
        
        # Check if goal is on or near energy tile
        x, y = goal_position
        print(f"‼️Goal position: {goal_position}, X  {x},  ", file=sys.stderr)
        if 0 <= x < GameConstants.MAP_WIDTH and 0 <= y < GameConstants.MAP_HEIGHT:
            if self.obs["map_features_energy"][y, x] > 0:
                return 2  # Energy focus

        # Check if goal is near a relic
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                rx, ry = self.obs["relic_nodes"][i]
                if abs(x - rx) <= 3 and abs(y - ry) <= 3:
                    return 3  # Relic focus

        # Default to balanced
        return 0

    def determine_unit_roles(self, available_unit_ids):
        """Assign roles to units based on goal position and game state"""
        unit_roles = np.zeros(GameConstants.MAX_UNITS, dtype=np.int32)

        # Calculate unit counts based on game phase
        steps = self.obs["steps"][0]
        max_steps = self.obs["match_steps"][0]
        match_progress = steps / max_steps if max_steps > 0 else 0

        # Calculate role percentages based on game phase
        if match_progress < 0.3:  # Early game
            explorer_pct = 0.5
            energy_pct = 0.3
            relic_pct = 0.1
            attacker_pct = 0.1
        elif match_progress < 0.7:  # Mid game
            explorer_pct = 0.3
            energy_pct = 0.2
            relic_pct = 0.4
            attacker_pct = 0.1
        else:  # Late game
            explorer_pct = 0.1
            energy_pct = 0.1
            relic_pct = 0.6
            attacker_pct = 0.2

        # Calculate role counts
        num_units = len(available_unit_ids)
        explorer_count = max(1, int(num_units * explorer_pct))
        energy_count = max(1, int(num_units * energy_pct))
        relic_count = max(1, int(num_units * relic_pct))
        attacker_count = max(0, num_units - explorer_count - energy_count - relic_count)

        # Assign roles to units
        role_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Track assigned roles

        for unit_id in available_unit_ids:
            if unit_id >= GameConstants.MAX_UNITS:
                continue

            unit_pos = self.obs["units_position"][self.team_id][unit_id]
            units_energy = self.obs["units_energy"][self.team_id][unit_id][0]

            # Assign role based on unit state and role counts
            if units_energy < 80 and role_counts[1] < energy_count:
                # Low energy - assign as energy collector
                unit_roles[unit_id] = 1
                role_counts[1] += 1
            elif role_counts[2] < relic_count:
                # Assign as relic seeker
                unit_roles[unit_id] = 2
                role_counts[2] += 1
            elif role_counts[0] < explorer_count:
                # Assign as explorer
                unit_roles[unit_id] = 0
                role_counts[0] += 1
            elif role_counts[3] < attacker_count:
                # Assign as attacker
                unit_roles[unit_id] = 3
                role_counts[3] += 1
            else:
                # Default to explorer
                unit_roles[unit_id] = 0
                role_counts[0] += 1

        return unit_roles

    def get_explorer_target(self, ship, goal_position):
        """Get target position for an explorer unit"""
        # Try to find unexplored tiles in the direction of the goal
        unexplored_pos = self.find_closest_unexplored_tile(ship, self.explored_tiles)

        if unexplored_pos:
            return unexplored_pos

        # Fallback to the goal position
        return goal_position

    def get_energy_target(self, ship):
        """Get target position for an energy collector unit"""
        energy_goal = ship.find_closest(
            NodeType.ENERGY_NODE,
            self.obs["map_features_tile_type"],
            self.obs["sensor_mask"],
        )

        if energy_goal:
            return energy_goal

        # Fallback to exploring for energy
        return self.find_closest_unexplored_tile(ship, self.explored_tiles)

    def get_relic_target(self, ship):
        """Get target position for a relic seeker unit"""
        visible_relics = []
        for i in range(len(self.obs["relic_nodes_mask"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                visible_relics.append(tuple(self.obs["relic_nodes"][i]))
        
        # If there are visible relics, prioritize actions
        if visible_relics:
            # 1. First priority: Go to known point-yielding tiles
            if hasattr(self, "point_yielding_tiles") and len(self.point_yielding_tiles) > 0:
                # Find closest point-yielding tile
                closest_point_tile = None
                min_dist = float("inf")
                
                for tile in self.point_yielding_tiles:
                    # Check if the tile is near a visible relic
                    is_near_relic = any(
                        abs(tile[0] - relic[0]) <= 2 and abs(tile[1] - relic[1]) <= 2
                        for relic in visible_relics
                    )
                    
                    if is_near_relic:
                        dist = abs(tile[0] - ship.pos[0]) + abs(tile[1] - ship.pos[1])
                        if dist < min_dist:
                            min_dist = dist
                            closest_point_tile = tile
                
                if closest_point_tile:
                    return closest_point_tile
            
            # 2. Second priority: Explore tiles around relics we haven't visited
            closest_unexplored_tile = None
            min_dist = float("inf")
            
            for relic_pos in visible_relics:
                rx, ry = relic_pos
                
                # Check all tiles in 5x5 grid around relic
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        x, y = rx + dx, ry + dy
                        
                        # Ensure tile is within bounds
                        if 0 <= x < GameConstants.MAP_WIDTH and 0 <= y < GameConstants.MAP_HEIGHT:
                            tile_pos = (x, y)
                            
                            # Skip if this is a known point-yielding tile (handled above)
                            if hasattr(self, "point_yielding_tiles") and tile_pos in self.point_yielding_tiles:
                                continue
                            
                            # Check if we've already visited this tile (using explored_tiles)
                            if not self.explored_tiles[x, y]:
                                dist = abs(x - ship.pos[0]) + abs(y - ship.pos[1])
                                if dist < min_dist:
                                    min_dist = dist
                                    closest_unexplored_tile = tile_pos
            
            if closest_unexplored_tile:
                return closest_unexplored_tile
            
            # 3. Third priority: Go to the relic itself
            closest_relic = min(
                visible_relics,
                key=lambda pos: abs(pos[0] - ship.pos[0]) + abs(pos[1] - ship.pos[1])
            )
            return closest_relic
        
        # If no relics are visible, fall back to exploration
        return self.find_closest_unexplored_tile(ship, self.explored_tiles)
        # for i in range(len(self.obs["relic_nodes"])):
        #     if self.obs["relic_nodes_mask"][i] > 0:
        #         return tuple(self.obs["relic_nodes"][i])

        # # If no relic visible, check for any remembered relics
        # if hasattr(self, "discovered_relics") and self.discovered_relics:
        #     closest_relic = min(
        #         self.discovered_relics,
        #         key=lambda pos: abs(pos[0] - ship.pos[0]) + abs(pos[1] - ship.pos[1]),
        #     )
        #     return closest_relic

        # # Fallback to exploring
        # return self.find_closest_unexplored_tile(ship, self.explored_tiles)

    def get_attacker_target(self, ship):
        """Get target position for an attacker unit"""
        # Find closest enemy
        enemy_positions = []
        for i in range(len(self.obs["units_position"][self.opp_team_id])):
            if self.obs["units_mask"][self.opp_team_id][i] > 0:
                enemy_pos = tuple(self.obs["units_position"][self.opp_team_id][i])
                if enemy_pos[0] >= 0 and enemy_pos[1] >= 0:  # Valid position
                    enemy_positions.append(enemy_pos)

        if enemy_positions:
            closest_enemy = min(
                enemy_positions,
                key=lambda pos: abs(pos[0] - ship.pos[0]) + abs(pos[1] - ship.pos[1]),
            )
            return closest_enemy

        # If no enemies visible, check relics (defend them)
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                return tuple(self.obs["relic_nodes"][i])

        # Fallback to exploring
        return self.find_closest_unexplored_tile(ship, self.explored_tiles)
    
    def process_observation(self, observation, remainingOverageTime: int = 60):
        """Process observation data for the agent"""
        self.obs = {
            "units_position": np.array(
                observation.get("units", {}).get(
                    "position", GameConstants.DEFAULT_UNIT_POS
                ),
                dtype=np.int8,
            ).reshape(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 2),
            "units_energy": np.array(
                observation.get("units", {}).get(
                    "energy", GameConstants.DEFAULT_UNIT_ENERGY
                ),
                dtype=np.int32,
            ).reshape(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS, 1),
            "units_mask": np.array(
                observation.get("units_mask", GameConstants.DEFAULT_UNITS_MASK),
                dtype=np.int8,
            ).reshape(GameConstants.NUM_TEAMS, GameConstants.MAX_UNITS),
            "sensor_mask": np.array(
                observation.get("sensor_mask", GameConstants.DEFAULT_SENSOR_MASK),
                dtype=np.int8,
            ).reshape(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
            "map_features_tile_type": np.array(
                observation.get("map_features", {}).get(
                    "tile_type", GameConstants.DEFAULT_TILE_TYPE
                ),
                dtype=np.int8,
            ).reshape(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
            "map_features_energy": np.array(
                observation.get("map_features", {}).get(
                    "energy", GameConstants.DEFAULT_MAP_FEATURES_ENERGY
                ),
                dtype=np.float64,
            ).reshape(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT),
            "relic_nodes": np.array(
                observation.get("relic_nodes", GameConstants.DEFAULT_RELIC_NODES),
                dtype=np.int32,
            ).reshape(GameConstants.MAX_RELIC_NODES, 2),
            "relic_nodes_mask": np.array(
                observation.get("relic_nodes_mask", GameConstants.DEFAULT_RELIC_MASK),
                dtype=np.float32,
            ).reshape(
                GameConstants.MAX_RELIC_NODES,
            ),
            "team_points": np.array(
                observation.get("team_points", GameConstants.DEFAULT_TEAM_POINTS),
                dtype=np.int32,
            ).reshape(
                GameConstants.NUM_TEAMS,
            ),
            "team_wins": np.array(
                observation.get("team_wins", GameConstants.DEFAULT_TEAM_WINS),
                dtype=np.int32,
            ).reshape(
                GameConstants.NUM_TEAMS,
            ),
            "steps": np.array([observation.get("steps", 0)], dtype=np.float32).reshape(
                1,
            ),
            "match_steps": np.array(
                [observation.get("match_steps", 0)], dtype=np.int32
            ).reshape(
                1,
            ),
            "remainingOverageTime": np.array(
                [observation.get("remainingOverageTime", remainingOverageTime)],
                dtype=np.int32,
            ).reshape(
                1,
            ),
        }

        self.enemy_positions = self.obs["units_position"][self.opp_team_id].tolist()
        self.grid = self.obs["map_features_tile_type"]
        visible_relic_node_ids = np.where(self.obs["relic_nodes_mask"])[0]
        for id in visible_relic_node_ids:
            if (
                id not in self.discovered_relic_nodes_ids
                and self.obs["relic_nodes"][id] not in self.obs["relic_nodes"]
            ):
                self.discovered_relic_nodes_ids.add(id)
                self.obs["relic_nodes"].append(self.obs["relic_nodes"][id])

        if not hasattr(self, "explored_tiles"):
            self.explored_tiles = np.zeros(
                (GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT), dtype=bool
            )

        visible_mask = self.obs["sensor_mask"] > 0
        self.explored_tiles = np.logical_or(self.explored_tiles, visible_mask)
        

    def rule_based_goals(self):
        """
        Fallback strategy using rule-based decision making when PPO model isn't available.
        Returns a single goal position for pathfinding.
        """
        active_unit_ids = np.where(self.obs["units_mask"][self.team_id])[0]
        if len(active_unit_ids) == 0:
            return (
                GameConstants.MAP_WIDTH // 2,
                GameConstants.MAP_HEIGHT // 2,
            )  # Default to center

        # Just pick the first unit for determining the goal
        unit_id = active_unit_ids[0]
        unit_pos = self.obs["units_position"][self.team_id][unit_id]
        units_energy = self.obs["units_energy"][self.team_id][unit_id][0]
        ship = Ship(unit_id, unit_pos, units_energy)

        # Rule-based goal selection logic
        game_step = self.obs["steps"][0]

        # Priority 1: Relic node if visible
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                return tuple(int(x) for x in self.obs["relic_nodes"][i])

        # Priority 2: Energy node if low energy
        if units_energy < 100:
            energy_pos = ship.find_closest(
                NodeType.ENERGY_NODE,
                self.obs["map_features_tile_type"],
                self.obs["sensor_mask"],
            )
            if energy_pos:
                return energy_pos

        # Priority 3: Exploration
        unexplored_pos = self.find_closest_unexplored_tile(ship, self.explored_tiles)
        if unexplored_pos:
            return unexplored_pos

        # Fallback to center
        return (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)

    # def find_best_sap_target(self, ship):
    #     """Find best target for SAP action with enhanced aggressiveness"""
    #     ship_pos = ship.get_position()
    #     ship_energy = ship.get_energy()
    #     sap_cost = int(self.env_cfg.get("unit_sap_cost", 30))
        
    #     # Must have enough energy to sap
    #     if ship_energy < sap_cost:
    #         return None
        
    #     # Lower energy threshold for sapping in late game
    #     step = self.obs["steps"][0]
    #     max_steps = self.obs["match_steps"][0]
    #     match_progress = step / max_steps if max_steps > 0 else 0
        
    #     # Find adjacent enemies
    #     potential_targets = []
        
    #     for i in range(len(self.obs["units_position"][self.opp_team_id])):
    #         if self.obs["units_mask"][self.opp_team_id][i] > 0:
    #             enemy_pos = tuple(int(val) for val in self.obs["units_position"][self.opp_team_id][i])
                
    #             # Skip invalid positions
    #             if enemy_pos[0] < 0 or enemy_pos[1] < 0:
    #                 continue
                
    #             enemy_energy = int(self.obs["units_energy"][self.opp_team_id][i][0])
                
    #             # Calculate Manhattan distance
    #             manhattan_dist = abs(ship_pos[0] - enemy_pos[0]) + abs(ship_pos[1] - enemy_pos[1])
                
    #             # If adjacent to enemy
    #             if manhattan_dist <= 1:
    #                 # Calculate priority for this target
    #                 priority = 1.0
                    
    #                 # Check if enemy is on a relic node (highest priority)
    #                 for j in range(len(self.obs["relic_nodes"])):
    #                     if self.obs["relic_nodes_mask"][j] > 0:
    #                         relic_pos = tuple(self.obs["relic_nodes"][j])
    #                         if relic_pos == enemy_pos:
    #                             priority = 10.0  # Extremely high priority
    #                             break
                    
    #                 # High energy enemies are high priority
    #                 if enemy_energy > 200:
    #                     priority += 3.0
    #                 elif enemy_energy > 100:
    #                     priority += 1.5
                    
    #                 # If we're in late game, be more aggressive
    #                 if match_progress > 0.7:
    #                     priority *= 1.5
                    
    #                 # Store target direction with priority
    #                 dx = enemy_pos[0] - ship_pos[0]
    #                 dy = enemy_pos[1] - ship_pos[1]
    #                 potential_targets.append((priority, (dx, dy)))
        
    #     # Return highest priority target if any
    #     if potential_targets:
    #         potential_targets.sort(reverse=True)
    #         print(f"Ship {ship.id} found sap target with priority {potential_targets[0][0]}", file=sys.stderr)
    #         return potential_targets[0][1]
        
    #     return None
    def find_best_sap_target(self, ship):
        """
        Enhanced SAP targeting that prioritizes enemies near relics.
        Replace your current find_best_sap_target method with this.
        """
        ship_pos = ship.get_position()
        ship_energy = ship.get_energy()
        sap_cost = int(self.env_cfg.get("unit_sap_cost", 30))
        sap_range = 1  # Adjacent tiles
        
        # Must have enough energy to sap
        if ship_energy < sap_cost:
            return None
        
        # Find enemy ships within range
        potential_targets = []
        
        for i in range(len(self.obs["units_position"][self.opp_team_id])):
            if self.obs["units_mask"][self.opp_team_id][i] > 0:
                enemy_pos = tuple(int(val) for val in self.obs["units_position"][self.opp_team_id][i])
                
                # Skip invalid positions
                if enemy_pos[0] < 0 or enemy_pos[1] < 0:
                    continue
                
                enemy_energy = int(self.obs["units_energy"][self.opp_team_id][i][0])
                
                # Calculate Manhattan distance
                manhattan_dist = abs(ship_pos[0] - enemy_pos[0]) + abs(ship_pos[1] - enemy_pos[1])
                
                # If enemy is in range
                if manhattan_dist <= sap_range:
                    # Calculate priority score for this target
                    priority = 1.0
                    
                    # HIGHEST PRIORITY: Enemy on a known point-yielding tile
                    if hasattr(self, "point_yielding_tiles") and enemy_pos in self.point_yielding_tiles:
                        priority = 10.0
                    
                    # HIGH PRIORITY: Enemy near any relic
                    if hasattr(self, "tiles_around_relics") and enemy_pos in self.tiles_around_relics:
                        priority = max(priority, 5.0)
                    
                    # Medium priority: High energy enemies
                    if enemy_energy > 150:
                        priority = max(priority, 2.0)
                    
                    # Store target direction with priority
                    dx = enemy_pos[0] - ship_pos[0]
                    dy = enemy_pos[1] - ship_pos[1]
                    potential_targets.append((priority, (dx, dy)))
        
        # Return highest priority target if any
        if potential_targets:
            potential_targets.sort(reverse=True)
            print(f"Ship {ship.id} found sap target with priority {potential_targets[0][0]}", file=sys.stderr)
            return potential_targets[0][1]
        
        return None

    def find_closest_unexplored_tile(self, ship, explored_tiles):
        closest = None
        min_dist = float("inf")
        ship_pos = ship.get_position()

        print(f"Ship position: {ship_pos}", file=sys.stderr)
        print(
            f"Custom explored tiles count: {np.sum(explored_tiles)}/{GameConstants.MAP_WIDTH * GameConstants.MAP_HEIGHT}",
            file=sys.stderr,
        )

        for x in range(GameConstants.MAP_WIDTH):
            for y in range(GameConstants.MAP_HEIGHT):
                # Skip the ship's current position
                if (x, y) == ship_pos:
                    continue

                # Check if tile is unexplored in our custom tracker
                if not explored_tiles[x, y]:
                    dist = abs(x - ship_pos[0]) + abs(y - ship_pos[1])
                    if dist < min_dist:
                        min_dist = dist
                        closest = (x, y)

        # If all tiles are explored or no valid target found
        if closest is None:
            # Try to find the furthest point from the ship
            max_dist = -1
            for x in range(GameConstants.MAP_WIDTH):
                for y in range(GameConstants.MAP_HEIGHT):
                    if (x, y) != ship_pos:
                        dist = abs(x - ship_pos[0]) + abs(y - ship_pos[1])
                        if dist > max_dist:
                            max_dist = dist
                            closest = (x, y)

        return closest

