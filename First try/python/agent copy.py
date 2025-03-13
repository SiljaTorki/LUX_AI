import sys
import numpy as np
from environment import NodeType, ActionType, GameConstants
from a_star import a_star_search
from stable_baselines3 import PPO

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
            self.model = PPO.load("./models_final/best_model/best_model")
            print("Successfully loaded PPO model", file=sys.stderr)
            print(f"Model policy network: {self.model.policy}", file=sys.stderr)
        except Exception as e:
            print(
                f"Failed to load model: {e}. Using fallback strategy", file=sys.stderr
            )
            self.model = None

    def act(self, step: int, observation, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit."""
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
        
        # Initialize actions array
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        
        # Process enemy positions with energy for A* pathfinding
        enemy_positions_with_energy = []
        for i in range(len(self.obs["units_position"][self.opp_team_id])):
            if self.obs["units_mask"][self.opp_team_id][i] > 0:
                pos_x = int(self.obs["units_position"][self.opp_team_id][i][0])
                pos_y = int(self.obs["units_position"][self.opp_team_id][i][1])
                
                if pos_x >= 0 and pos_y >= 0:
                    enemy_energy = int(self.obs["units_energy"][self.opp_team_id][i][0])
                    enemy_positions_with_energy.append(((pos_x, pos_y), enemy_energy))
        
        try:
            # Make a single prediction with your existing model
            action, _ = self.model.predict(self.obs, deterministic=True)
            print(f"🤖 PPO model raw prediction: {action}", file=sys.stderr)

            # Convert to goal position
            # This will be used as a primary target for your ships
            if isinstance(action, (int, np.integer)) or (isinstance(action, np.ndarray) and action.size == 1):
                action_value = int(action) if isinstance(action, np.ndarray) else action
                
                # Get the first unit's position for calculating the goal
                unit_id = available_unit_ids[0] if len(available_unit_ids) > 0 else 0
                unit_pos = self.obs["units_position"][self.team_id][unit_id]
                
                if action_value == ActionType.STAY.value:
                    goal_position = tuple(unit_pos)
                elif action_value == ActionType.MOVE_UP.value:
                    goal_position = (int(unit_pos[0]), max(0, int(unit_pos[1]) - 3))
                elif action_value == ActionType.MOVE_RIGHT.value:
                    goal_position = (
                        min(GameConstants.MAP_WIDTH - 1, int(unit_pos[0]) + 3),
                        int(unit_pos[1]),
                    )
                elif action_value == ActionType.MOVE_DOWN.value:
                    goal_position = (
                        int(unit_pos[0]),
                        min(GameConstants.MAP_HEIGHT - 1, int(unit_pos[1]) + 3),
                    )
                elif action_value == ActionType.MOVE_LEFT.value:
                    goal_position = (max(0, int(unit_pos[0]) - 3), int(unit_pos[1]))
                elif action_value == ActionType.SAP.value:
                    enemy_pos = self.find_nearest_enemy(unit_pos)
                    goal_position = enemy_pos if enemy_pos else tuple(unit_pos)

            # Handle coordinate-based prediction if your model produces coordinates
            elif isinstance(action, np.ndarray) and action.shape == (2,):
                goal_x = min(max(int(action[0]), 0), GameConstants.MAP_WIDTH - 1)
                goal_y = min(max(int(action[1]), 0), GameConstants.MAP_HEIGHT - 1)
                goal_position = (goal_x, goal_y)
            else:
                print(f"⚠️ Unexpected prediction shape: {action.shape if hasattr(action, 'shape') else type(action)}", file=sys.stderr)
                goal_position = self.rule_based_goals()

            
            # Make strategy decisions based on game state
            strategy = self.determine_strategy(goal_position)
            unit_roles = self.determine_unit_roles(available_unit_ids)
            print(f"Strategy: {strategy}, Roles: {unit_roles}", file=sys.stderr)
            
            self.coordinate_relic_control(unit_roles, available_unit_ids)
            
            
            # Assign actions to each unit based on their roles and the main goal
            for unit_id in available_unit_ids:
                ship = fleet.get_unit(unit_id)
                unit_pos = ship.pos
                role = unit_roles[unit_id] if unit_id < len(unit_roles) else 0
                max_steps = self.obs["match_steps"][0]
                
                sap_target = None
                if role == 3 or step > max_steps * 0.7:  # All units can sap in late game
                    sap_target = self.find_best_sap_target(ship)
                    
                if sap_target:
                    print(f"🔋 Ship {unit_id} will SAP in direction {sap_target}", file=sys.stderr)
                    actions[unit_id] = [ActionType.SAP.value, sap_target[0], sap_target[1]]
                    continue
                
                # Determine unit's target based on role and game state
                if role == 0:  # Explorer
                    target = self.get_explorer_target(ship, goal_position)
                elif role == 1:  # Energy collector
                    target = self.get_energy_target(ship)
                elif role == 2:  # Relic seeker
                    target = self.get_relic_target(ship)
                elif role == 3:  # Attacker/defender
                    target = self.get_attacker_target(ship)
                else:
                    target = goal_position
                    
                print(f"Unit {unit_id} role: {role}, target: {target}", file=sys.stderr)
                
                # Use improved A* for pathfinding with dynamic obstacle avoidance
                path = a_star_search(
                    ship=ship,
                    goal=target,
                    map_features=self.obs["map_features_tile_type"],
                    team_vision=self.obs["sensor_mask"],
                    enemy_positions=enemy_positions_with_energy,  # Pass positions with energy
                    step_count=step,
                )
                
                # Convert path to action
                if len(path) > 1:
                    next_position = path[1]
                    direction = ship.get_direction(next_position)
                    actions[unit_id] = [direction, 0, 0]
                
        except Exception as e:
            print(f"⚠️ Error in model prediction: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            
            # Fall back to rule-based approach
            for unit_id in available_unit_ids:
                ship = fleet.get_unit(unit_id)
                
                # Get the fallback goal
                goal_position = self.rule_based_goals()
                print(f"Rule-based goal for unit {unit_id}: {goal_position}", file=sys.stderr)
                
                # Use A* to plan path
                path = a_star_search(
                    ship=ship,
                    goal=goal_position,
                    map_features=self.obs["map_features_tile_type"],
                    team_vision=self.obs["sensor_mask"],
                    enemy_positions=enemy_positions_with_energy,
                    step_count=step,
                )
                
                # Convert path to action
                if len(path) > 1:
                    next_position = path[1]
                    direction = ship.get_direction(next_position)
                    actions[unit_id] = [direction, 0, 0]
        print(f"🤖 Final actions: {actions}", file=sys.stderr)
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

        # Get game state information
        steps = self.obs["steps"][0]
        max_steps = self.obs["match_steps"][0]
        match_progress = steps / max_steps if max_steps > 0 else 0
        
        # Count active units on both teams
        team0_units = sum(1 for i in range(len(self.obs["units_mask"][self.team_id])) 
                        if self.obs["units_mask"][self.team_id][i] > 0)
        team1_units = sum(1 for i in range(len(self.obs["units_mask"][self.opp_team_id])) 
                        if self.obs["units_mask"][self.opp_team_id][i] > 0)
        
        # Calculate energy totals
        team0_energy = sum(self.obs["units_energy"][self.team_id][i][0] 
                        for i in range(len(self.obs["units_energy"][self.team_id])) 
                        if self.obs["units_mask"][self.team_id][i] > 0)
        team1_energy = sum(self.obs["units_energy"][self.opp_team_id][i][0] 
                        for i in range(len(self.obs["units_energy"][self.opp_team_id])) 
                        if self.obs["units_mask"][self.opp_team_id][i] > 0)
        
        # Check unit and energy advantages
        unit_advantage = team0_units >= team1_units
        energy_advantage = team0_energy >= team1_energy * 1.2  # 20% advantage
        energy_disadvantage = team0_energy * 1.2 <= team1_energy
        
        # Count visible relics
        visible_relics = sum(1 for i in range(len(self.obs["relic_nodes_mask"])) 
                        if self.obs["relic_nodes_mask"][i] > 0)
    
        if visible_relics > 0:
            print(f"Visible relics detected: {visible_relics}. Switching to relic focus strategy.", file=sys.stderr)
            return 3  # Relic focus
        
        # Early game strategies (0-30%)
        if match_progress < 0.3:
            if steps < 10:
                return 1  # Initial exploration
            elif not unit_advantage:
                return 2  # Energy focus if we have fewer units
            elif visible_relics > 0:
                return 3  # Early relic focus if visible
            else:
                return 1  # Default to exploration in early game
        
        # Mid game strategies (30-70%)
        elif match_progress < 0.7:
            if energy_disadvantage:
                return 2  # Energy focus if disadvantaged
            elif visible_relics > 0:
                return 3  # Relic focus if visible
            elif not unit_advantage:
                return 4  # Defensive/aggressive based on position
            else:
                # Check if goal is on or near energy tile
                x, y = goal_position
                if (0 <= x < GameConstants.MAP_WIDTH and 
                    0 <= y < GameConstants.MAP_HEIGHT and
                    self.obs["map_features_energy"][y, x] > 0):
                    return 2  # Energy focus
                    
                # Check if goal is near a relic
                for i in range(len(self.obs["relic_nodes"])):
                    if self.obs["relic_nodes_mask"][i] > 0:
                        rx, ry = self.obs["relic_nodes"][i]
                        if abs(x - rx) <= 3 and abs(y - ry) <= 3:
                            return 3  # Relic focus
                
                return 0  # Balanced approach in mid-game
        
        # Late game strategies (70-100%)
        else:
            if visible_relics > 0:
                return 3  # Relic focus is highest priority in late game
            elif energy_advantage:
                return 4  # Aggressive/hunting if energy advantage
            elif energy_disadvantage:
                return 2  # Energy focus if disadvantaged
            else:
                return 0  # Balanced approach

    def determine_unit_roles(self, available_unit_ids):
        """Assign roles to units with enhanced sapping focus"""
        unit_roles = np.zeros(GameConstants.MAX_UNITS, dtype=np.int32)

        # Calculate unit counts based on game phase
        steps = self.obs["steps"][0]
        max_steps = self.obs["match_steps"][0]
        match_progress = steps / max_steps if max_steps > 0 else 0

        # Calculate role percentages with more attackers in late game
        if match_progress < 0.3: 
            explorer_pct = 0.5
            energy_pct = 0.3
            relic_pct = 0.1
            attacker_pct = 0.1
        elif match_progress < 0.7: 
            explorer_pct = 0.2
            energy_pct = 0.3
            relic_pct = 0.3
            attacker_pct = 0.2
        else:  
            explorer_pct = 0.1
            energy_pct = 0.1
            relic_pct = 0.4
            attacker_pct = 0.4 

        # Calculate role counts
        num_units = len(available_unit_ids)
        explorer_count = max(1, int(num_units * explorer_pct))
        energy_count = max(1, int(num_units * energy_pct))
        relic_count = max(1, int(num_units * relic_pct))
        attacker_count = max(0, num_units - explorer_count - energy_count - relic_count)

        # Check if any relics are visible
        visible_relics = sum(1 for i in range(len(self.obs["relic_nodes_mask"])) 
                       if self.obs["relic_nodes_mask"][i] > 0)
    
        # In the late game with visible relics, massively prioritize relic seekers
        if match_progress > 0.6 and visible_relics > 0:
            relic_count = max(visible_relics, int(num_units * 0.6))  # 60% of units for relics
            attacker_count = max(2, int(num_units * 0.3))  # 30% for attack/defense
            energy_count = max(1, num_units - relic_count - attacker_count)
            explorer_count = 0  # No need for explorers with visible relics
            
            print(f"Late game relic focus: {relic_count} relic seekers, {attacker_count} attackers", file=sys.stderr)
        
        
        # If relics are visible, increase attackers and relic seekers
        if visible_relics > 0:
            relic_count = max(2, int(num_units * 0.4))
            attacker_count = max(2, int(num_units * 0.4))
            energy_count = max(1, num_units - relic_count - attacker_count)
            explorer_count = 0  # No need for explorers once relics are found
        
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
            elif role_counts[3] < attacker_count:
                # Assign as attacker
                unit_roles[unit_id] = 3
                role_counts[3] += 1
            elif role_counts[0] < explorer_count:
                # Assign as explorer
                unit_roles[unit_id] = 0
                role_counts[0] += 1
            else:
                # Default to attackers in late game, explorers in early game
                if match_progress > 0.5:
                    unit_roles[unit_id] = 3  # More attackers in late game
                    role_counts[3] += 1
                else:
                    unit_roles[unit_id] = 0
                    role_counts[0] += 1

        print(f"Role assignment: Explorers={role_counts[0]}, Energy={role_counts[1]}, Relic={role_counts[2]}, Attackers={role_counts[3]}", file=sys.stderr)
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
        ship_pos = ship.pos
    
        # Find energy nodes within view
        energy_spots = []
        for x in range(GameConstants.MAP_WIDTH):
            for y in range(GameConstants.MAP_HEIGHT):
                if (self.obs["sensor_mask"][x, y] and 
                    self.obs["map_features_tile_type"][x, y] == NodeType.ENERGY_NODE.value):
                    # Calculate distance
                    dist = abs(x - ship_pos[0]) + abs(y - ship_pos[1])
                    # Get energy value
                    energy_value = self.obs["map_features_energy"][y, x]
                    # Score based on energy/distance ratio
                    score = energy_value / max(1, dist)
                    energy_spots.append((score, (x, y)))
        
        if energy_spots:
            energy_spots.sort(reverse=True)  # Sort by score (highest first)
            return energy_spots[0][1]
        
        # Fallback to standard closest energy node
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
        """Get target position for a relic seeker unit with improved positioning"""
        ship_pos = ship.pos
        ship_id = ship.id
        
        # First, check for visible relics
        visible_relics = []
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                relic_pos = tuple(self.obs["relic_nodes"][i])
                
                # Count units already targeting this relic
                units_at_relic = 0
                for j in range(len(self.obs["units_position"][self.team_id])):
                    if self.obs["units_mask"][self.team_id][j] > 0 and j != ship.id:
                        unit_pos = tuple(self.obs["units_position"][self.team_id][j])
                        if unit_pos == relic_pos:
                            units_at_relic += 1
                
                # Calculate base score (lower distance = higher score)
                dist = abs(relic_pos[0] - ship_pos[0]) + abs(relic_pos[1] - ship_pos[1])
                score = 100.0 - dist
                
                # Penalize relics that already have many units
                if units_at_relic >= 2:
                    score -= 50  # Heavily penalize overcrowded relics
                
                visible_relics.append((score, relic_pos, units_at_relic))
        
        # Sort relics by score
        if visible_relics:
            visible_relics.sort(reverse=True)
            
            # Use ship ID to distribute units more evenly
            # Ships with different IDs will tend to pick different relics if available
            relic_index = ship_id % min(len(visible_relics), 3)  # Cycle through top 3 relics
            
            # But if a relic has no units, prioritize it regardless
            for score, pos, units in visible_relics:
                if units == 0:
                    return pos
            
            # Otherwise use the distributed index
            return visible_relics[relic_index][1]
        
        # If no visible relics, check discovered relics
        if hasattr(self, "discovered_relics") and self.discovered_relics:
            # Convert to list and sort by distance
            known_relics = [(pos, abs(pos[0] - ship_pos[0]) + abs(pos[1] - ship_pos[1])) 
                        for pos in self.discovered_relics.keys()]
            known_relics.sort(key=lambda x: x[1])  # Sort by distance
            
            # Again, use ship ID to distribute units among known relics
            relic_index = ship_id % min(len(known_relics), 3)  # Cycle through closest 3 relics
            return known_relics[relic_index][0]
        
        # No known relics - focus on exploration of likely relic areas
        # Look for unexplored center regions (relics are often placed more centrally)
        center_x, center_y = GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2
        center_unexplored = []
        
        # Prioritize unexplored tiles closer to center
        for x in range(GameConstants.MAP_WIDTH):
            for y in range(GameConstants.MAP_HEIGHT):
                if not self.explored_tiles[x, y]:
                    # Calculate distance from center
                    center_dist = abs(x - center_x) + abs(y - center_y)
                    # Score is higher for tiles closer to center
                    score = 50 - center_dist
                    center_unexplored.append((score, (x, y)))
        
        if center_unexplored:
            center_unexplored.sort(reverse=True)  # Sort by score
            return center_unexplored[0][1]
        
        # Fallback to regular exploration
        return self.find_closest_unexplored_tile(ship, self.explored_tiles)

    def get_attacker_target(self, ship):
        """Get target position for an attacker unit"""
        hunting_target = self.get_sap_hunting_target(ship)
        if hunting_target:
            return hunting_target
        
        # Second priority: check for relic positions to defend/attack
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                return tuple(self.obs["relic_nodes"][i])
        
        # Third priority: explore for relics or energy if we're low
        if ship.energy < 100:
            return self.get_energy_target(ship)
        
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
        
        if not hasattr(self, "discovered_relics"):
            self.discovered_relics = {}  # Use a dictionary to store position and last seen time
        
        visible_relic_node_ids = np.where(self.obs["relic_nodes_mask"])[0]
        current_step = self.obs["steps"][0]
        
        # Add or update visible relics
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                relic_pos = tuple(self.obs["relic_nodes"][i])
                self.discovered_relics[relic_pos] = current_step  # Update last seen time
                
                print(f"Tracking relic at {relic_pos}, step {current_step}", file=sys.stderr)

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
    
    def coordinate_relic_control(self, unit_roles, available_unit_ids):
        """Coordinate units to effectively control relic nodes"""
        # Detect visible relics
        visible_relics = []
        for i in range(len(self.obs["relic_nodes"])):
            if self.obs["relic_nodes_mask"][i] > 0:
                relic_pos = tuple(self.obs["relic_nodes"][i])
                
                # Count enemy units near this relic
                enemies_near = 0
                for j in range(len(self.obs["units_position"][self.opp_team_id])):
                    if self.obs["units_mask"][self.opp_team_id][j] > 0:
                        enemy_pos = tuple(self.obs["units_position"][self.opp_team_id][j])
                        if enemy_pos[0] >= 0 and enemy_pos[1] >= 0:
                            enemy_dist = abs(enemy_pos[0] - relic_pos[0]) + abs(enemy_pos[1] - relic_pos[1])
                            if enemy_dist <= 2:
                                enemies_near += 1
                
                # Count our units near this relic
                allies_near = 0
                allies_ids = []
                for j, unit_id in enumerate(available_unit_ids):
                    unit_pos = tuple(self.obs["units_position"][self.team_id][unit_id])
                    unit_dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
                    if unit_dist <= 3:
                        allies_near += 1
                        allies_ids.append(unit_id)
                
                visible_relics.append((relic_pos, enemies_near, allies_near, allies_ids))
        
        # If no visible relics, don't change anything
        if not visible_relics:
            return
        
        # For each visible relic, determine how many units we need
        for relic_pos, enemies_near, allies_near, allies_ids in visible_relics:
            # Calculate how many units we should assign
            target_units = min(enemies_near + 1, 3)  # At least one more than enemies, up to 3
            
            # If we need more units, assign more relic seekers
            if allies_near < target_units:
                # Find closest units not already assigned to this relic
                potential_units = []
                for unit_id in available_unit_ids:
                    if unit_id not in allies_ids:
                        unit_pos = tuple(self.obs["units_position"][self.team_id][unit_id])
                        unit_dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
                        potential_units.append((unit_dist, unit_id))
                
                # Sort by distance and assign closest units
                if potential_units:
                    potential_units.sort()  # Sort by distance
                    units_to_assign = min(target_units - allies_near, len(potential_units))
                    
                    for i in range(units_to_assign):
                        unit_id = potential_units[i][1]
                        unit_roles[unit_id] = 2  # Assign as relic seeker
                        print(f"Reassigning unit {unit_id} to relic at {relic_pos}", file=sys.stderr)
            
            # If we have too many units, reassign some
            elif allies_near > target_units + 1:
                # Sort allies by distance to relic (keep closest ones)
                allies_by_dist = []
                for unit_id in allies_ids:
                    unit_pos = tuple(self.obs["units_position"][self.team_id][unit_id])
                    unit_dist = abs(unit_pos[0] - relic_pos[0]) + abs(unit_pos[1] - relic_pos[1])
                    allies_by_dist.append((unit_dist, unit_id))
                
                allies_by_dist.sort()  # Sort by distance
                
                # Reassign excess units to attackers
                for i in range(target_units, len(allies_by_dist)):
                    unit_id = allies_by_dist[i][1]
                    unit_roles[unit_id] = 3  # Reassign as attacker
                    print(f"Reassigning excess unit {unit_id} from relic to attacker", file=sys.stderr)

    def find_best_sap_target(self, ship):
        """Find best target for SAP action with enhanced aggressiveness"""
        ship_pos = ship.get_position()
        ship_energy = ship.get_energy()
        sap_cost = int(self.env_cfg.get("unit_sap_cost", 30))
        
        # Must have enough energy to sap
        if ship_energy < sap_cost:
            return None
        
        # Lower energy threshold for sapping in late game
        step = self.obs["steps"][0]
        max_steps = self.obs["match_steps"][0]
        match_progress = step / max_steps if max_steps > 0 else 0
        
        # Find adjacent enemies
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
                
                # If adjacent to enemy
                if manhattan_dist <= 1:
                    # Calculate priority for this target
                    priority = 1.0
                    
                    # Check if enemy is on a relic node (highest priority)
                    for j in range(len(self.obs["relic_nodes"])):
                        if self.obs["relic_nodes_mask"][j] > 0:
                            relic_pos = tuple(self.obs["relic_nodes"][j])
                            if relic_pos == enemy_pos:
                                priority = 10.0  # Extremely high priority
                                break
                    
                    # High energy enemies are high priority
                    if enemy_energy > 200:
                        priority += 3.0
                    elif enemy_energy > 100:
                        priority += 1.5
                    
                    # If we're in late game, be more aggressive
                    if match_progress > 0.7:
                        priority *= 1.5
                    
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
        
    def get_sap_hunting_target(self, ship):
        """Get a target position for actively hunting enemies to sap"""
        ship_pos = ship.pos
        ship_energy = ship.energy
        sap_cost = int(self.env_cfg.get("unit_sap_cost", 30))
        
        # Only hunt if we have sufficient energy
        if ship_energy < sap_cost * 2:
            return self.get_energy_target(ship)  # Get energy first
        
        # Find nearby enemies that are worth hunting
        huntable_enemies = []
        
        for i in range(len(self.obs["units_position"][self.opp_team_id])):
            if self.obs["units_mask"][self.opp_team_id][i] > 0:
                enemy_pos = tuple(int(val) for val in self.obs["units_position"][self.opp_team_id][i])
                
                # Skip invalid positions
                if enemy_pos[0] < 0 or enemy_pos[1] < 0:
                    continue
                    
                enemy_energy = int(self.obs["units_energy"][self.opp_team_id][i][0])
                manhattan_dist = abs(ship_pos[0] - enemy_pos[0]) + abs(ship_pos[1] - enemy_pos[1])
                
                # Hunt enemies with low energy or that are near relics
                hunt_score = 0
                
                # Low energy enemies are easier targets
                if enemy_energy < ship_energy:
                    hunt_score += 2.0
                
                # Enemies near relics are high-value targets
                for j in range(len(self.obs["relic_nodes"])):
                    if self.obs["relic_nodes_mask"][j] > 0:
                        relic_pos = tuple(self.obs["relic_nodes"][j])
                        relic_dist = abs(enemy_pos[0] - relic_pos[0]) + abs(enemy_pos[1] - relic_pos[1])
                        if relic_dist <= 3:
                            hunt_score += 5.0 / (relic_dist + 1)  # Higher score for closer to relics
                
                # Distance penalty - prefer closer enemies
                distance_factor = max(1, manhattan_dist) ** 0.75  # Non-linear scaling
                final_score = hunt_score / distance_factor
                
                if final_score > 0:
                    huntable_enemies.append((final_score, enemy_pos))
        
        # Sort by score and return the best target
        if huntable_enemies:
            huntable_enemies.sort(reverse=True)
            return huntable_enemies[0][1]
        
        # If no good hunting targets, target the nearest enemy
        return self.find_nearest_enemy(ship_pos)

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