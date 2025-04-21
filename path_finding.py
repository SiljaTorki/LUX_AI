from astar import AStar
import numpy as np
from environment import GameConstants, ActionType

def manhattan_distance(a, b):
    """
    Calculate the Manhattan distance between two points.
    
    Args:
        a: Tuple (x, y) for first point
        b: Tuple (x, y) for second point
        
    Returns:
        Manhattan distance between the points
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

class LuxAStarPlanner(AStar):
    """
    A* implementation for Lux AI S3 using the astar library.
    This class handles pathfinding on the game map.
    """
    
    def __init__(self, cost_map=None, map_width=GameConstants.MAP_WIDTH, map_height=GameConstants.MAP_HEIGHT):
        """
        Initialize the A* pathfinder with a cost map.
        
        Args:
            cost_map: 2D array with costs for each position (None for default)
            map_width: Width of the map
            map_height: Height of the map
        """
        self.map_width = map_width
        self.map_height = map_height
        
        # Initialize cost map with default values if not provided
        if cost_map is None:
            self.cost_map = np.ones((map_height, map_width), dtype=float)
        else:
            self.cost_map = cost_map
    
    def update_cost_map(self, obs):
        """
        Update the cost map based on the observation.
        
        Args:
            obs: Observation from the environment
        """
        # Reset the cost map
        self.cost_map = np.ones((self.map_height, self.map_width), dtype=float)
        
        # Mark asteroid tiles as unwalkable (infinite cost)
        # And add higher costs for nebula tiles
        for y in range(self.map_height):
            for x in range(self.map_width):
                # Only process tiles we can see
                if obs["sensor_mask"][y, x] == 1:
                    tile_type = obs["map_features_tile_type"][y, x]
                    
                    # Asteroid (unwalkable)
                    if tile_type == 2:
                        self.cost_map[y, x] = float('inf')
                    
                    # Nebula (higher cost)
                    elif tile_type == 1:
                        self.cost_map[y, x] = 5.0
                    
                    # If it has energy, reduce the cost
                    energy = obs["map_features_energy"][y, x]
                    if energy > 0:
                        # Scale discount by energy amount, but don't go below 0.5
                        discount = min(0.5, energy / 100.0)
                        if self.cost_map[y, x] != float('inf'):  # Don't discount unwalkable tiles
                            self.cost_map[y, x] = max(0.5, self.cost_map[y, x] - discount)
    
    def heuristic_cost_estimate(self, n1, n2):
        """
        Estimate the cost from n1 to n2 (Manhattan distance).
        
        Args:
            n1: First node (x, y) tuple
            n2: Second node (x, y) tuple
            
        Returns:
            Estimated cost
        """
        return manhattan_distance(n1, n2)
    
    def distance_between(self, n1, n2):
        """
        Calculate the actual cost between adjacent nodes.
        
        Args:
            n1: First node (x, y) tuple
            n2: Second node (x, y) tuple
            
        Returns:
            Cost between the nodes
        """
        # Make sure we're dealing with adjacent nodes
        dx = abs(n1[0] - n2[0])
        dy = abs(n1[1] - n2[1])
        
        if dx + dy != 1:
            # Not adjacent nodes, return a high cost
            return float('inf')
        
        # Return the cost from the cost map for the destination node
        return self.cost_map[n2[1], n2[0]]
    
    def neighbors(self, node):
        """
        Get the neighbors of a node.
        
        Args:
            node: Current node (x, y) tuple
            
        Returns:
            List of neighbor nodes
        """
        x, y = node
        neighbors = []
        
        # Check the four cardinal directions
        for dx, dy in [(0, -1), (1, 0), (0, 1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check if within map bounds
            if 0 <= nx < self.map_width and 0 <= ny < self.map_height:
                # Only add if not an obstacle (infinite cost)
                if self.cost_map[ny, nx] < float('inf'):
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def is_goal_reached(self, current, goal):
        """
        Check if the goal is reached.
        
        Args:
            current: Current node (x, y) tuple
            goal: Goal node (x, y) tuple
            
        Returns:
            True if goal is reached, False otherwise
        """
        return current == goal


class StaticPathPlanner:
    """
    A static path planner that uses A* to find paths for units.
    This planner does not consider dynamic obstacles and only plans once at initialization.
    """
    
    def __init__(self, map_width=GameConstants.MAP_WIDTH, map_height=GameConstants.MAP_HEIGHT):
        """
        Initialize the path planner with the map dimensions.
        
        Args:
            map_width: Width of the map
            map_height: Height of the map
        """
        self.map_width = map_width
        self.map_height = map_height
        self.astar = LuxAStarPlanner(map_width=map_width, map_height=map_height)
        
        # Paths cache - stores pre-computed paths for each unit to their targets
        self.paths = {}
        # Target cache - stores the current target for each unit
        self.targets = {}
    
    def path_to_action(self, current_pos, next_pos):
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
            return ActionType.MOVE_RIGHT.value  # Move Right
        elif dx == 0 and dy == 1:
            return ActionType.MOVE_DOWN.value  # Move Down
        elif dx == -1 and dy == 0:
            return ActionType.MOVE_LEFT.value  # Move Left
        else:
            # Default to no action if not a cardinal direction
            return ActionType.STAY.value
    
    def compute_paths_for_all_units(self, obs, player_id, targets_dict):
        """
        Compute paths for all units to their assigned targets.
        
        Args:
            obs: Observation from the environment
            player_id: ID of the player
            targets_dict: Dictionary mapping unit_idx to target position
            
        Returns:
            Dictionary mapping unit_idx to paths
        """
        
        team_id = 0 if player_id == "player_0" else 1
        # Update the cost map based on the current observation
        self.astar.update_cost_map(obs)
        
        # Store the targets
        self.targets = targets_dict
        
        # Compute paths for all units
        self.paths = {}
        for unit_idx, target in targets_dict.items():
            # Check if the unit exists and is visible
            if not obs["units_mask"][team_id][unit_idx]:
                continue
            
            # Get unit position
            unit_pos = (
                obs["units_position"][team_id][unit_idx][0].item(),
                obs["units_position"][team_id][unit_idx][1].item()
            )
            
            # Debug: Check if target is valid
            tx, ty = target
            if tx < 0 or tx >= self.map_width or ty < 0 or ty >= self.map_height:
                print(f"Invalid target position: {target}")
                self.paths[unit_idx] = [unit_pos]
                continue
            
            # Debug: Check if target is visible
            if not obs["sensor_mask"][ty, tx]:
                print(f"Target {target} not visible in sensor mask")
            
            # Debug: Check if target is on obstacle
            if self.astar.cost_map[ty, tx] == float('inf'):
                print(f"Target {target} is on an obstacle")
                # Find closest non-obstacle position
                target = self._find_closest_walkable_position(target, self.astar.cost_map)
                if target is None:
                    print(f"No walkable position found near target")
                    self.paths[unit_idx] = [unit_pos]
                    continue
            
            # Compute path - astar.astar() may return None if no path found
            path_result = self.astar.astar(unit_pos, target)
            
            # Check if path was found before converting to list
            if path_result is not None:
                path = list(path_result)
                if path:  # Make sure the path is not empty
                    self.paths[unit_idx] = path
            else:
                print(f"Path not found for unit {unit_idx} from {unit_pos} to {target}")
                # If no path found, store just the current position
                # This will make the unit stay put until a valid path is found
                self.paths[unit_idx] = [unit_pos]
        
        return self.paths
    
    def get_next_actions(self, obs, player_id):
        """
        Get the next actions for all units based on their pre-computed paths.
        
        Args:
            obs: Observation from the environment
            player_id: ID of the player
            
        Returns:
            List of actions for each unit
        """
        actions = []
        team_id = 0 if player_id == "player_0" else 1
        
        for unit_idx in range(len(obs["units_mask"][team_id])):
            # Check if the unit exists and is visible
            if not obs["units_mask"][team_id][unit_idx]:
                # Unit doesn't exist, stay in place
                actions.append(ActionType.STAY.value)
                continue
            
            # Get unit position
            unit_pos = (
                obs["units_position"][team_id][unit_idx][0].item(),
                obs["units_position"][team_id][unit_idx][1].item()
            )
            
            # Check if we have a path for this unit
            if unit_idx in self.paths and len(self.paths[unit_idx]) > 1:
                # Get the next position in the path
                next_pos = self.paths[unit_idx][1]
                
                # Calculate the action to take
                action = self.path_to_action(unit_pos, next_pos)
                
                # Append action
                actions.append(action)
                
                # Remove the current position from the path
                self.paths[unit_idx] = self.paths[unit_idx][1:]
            else:
                # No path or reached destination, stay in place
                actions.append(ActionType.STAY.value)
        
        # Pad actions to max units if needed
        while len(actions) < GameConstants.MAX_UNITS:
            actions.append(ActionType.STAY.value)
        
        return actions
    
    def find_targets_for_units(self, obs, player_id):
        """
        Find suitable targets for units based on current observation.
        
        Args:
            obs: Observation from the environment
            player_id: ID of the player
            
        Returns:
            Dictionary mapping unit indices to target positions
        """
        targets = {}
        team_id = 0 if player_id == "player_0" else 1
        
        # Get positions of active units
        active_units = []
        for i in range(GameConstants.MAX_UNITS):
            if obs["units_mask"][team_id][i]:
                unit_pos = (
                    obs["units_position"][team_id][i][0].item(),
                    obs["units_position"][team_id][i][1].item()
                )
                unit_energy = obs["units_energy"][team_id][i][0]
                active_units.append((i, unit_pos, unit_energy))
    
        # Find visible relic nodes (highest priority)
        visible_relics = []
        for i in range(len(obs["relic_nodes_mask"])):
            if obs["relic_nodes_mask"][i]:
                relic_pos = (
                    obs["relic_nodes"][i][0].item(),
                    obs["relic_nodes"][i][1].item()
                )
                # Check if relic is visible and has valid coordinates
                if (relic_pos[0] >= 0 and relic_pos[1] >= 0 and 
                    obs["sensor_mask"][relic_pos[1], relic_pos[0]]):
                    visible_relics.append(relic_pos)
        
        # Find visible energy nodes (second priority)
        visible_energy = []
        energy_map = np.array(obs["map_features_energy"])
        
        for y in range(self.map_height):
            for x in range(self.map_width):
                # Check if tile is visible, has energy, and is not an asteroid
                if (obs["sensor_mask"][y, x] and energy_map[y, x] > 0 and
                    obs["map_features_tile_type"][y, x] != 2):
                    visible_energy.append((x, y, energy_map[y, x]))
    
        # Sort energy targets by value
        visible_energy.sort(key=lambda x: x[2], reverse=True)
        
        # Determine enemy spawn location for exploration
        enemy_spawn = (
            (self.map_width - 1, self.map_height - 1) 
            if team_id == 0 
            else (0, 0)
        )
        
        # Assign targets to units
        assigned_units = set()
        
        # First assign to visible relics
        if visible_relics:
            for unit_idx, unit_pos, _ in active_units:
                if unit_idx in assigned_units:
                    continue
                
                # Find closest visible relic
                closest_relic = min(
                    visible_relics, 
                    key=lambda r: manhattan_distance(unit_pos, r)
                )
                
                targets[unit_idx] = closest_relic
                assigned_units.add(unit_idx)
                
                if len(assigned_units) == len(active_units):
                    return targets
    
        # Then assign to visible energy nodes
        if visible_energy:
            for unit_idx, unit_pos, unit_energy in active_units:
                if unit_idx in assigned_units:
                    continue
                
                # Find best energy node (closest if unit has high energy, most valuable if low)
                energy_nodes = [(x, y) for x, y, _ in visible_energy]
                if unit_energy < 50:  # Low energy - prioritize value
                    best_energy = max(
                        energy_nodes,
                        key=lambda e: visible_energy[energy_nodes.index(e)][2]
                    )
                else:  # High energy - prioritize proximity
                    best_energy = min(
                        energy_nodes,
                        key=lambda e: manhattan_distance(unit_pos, e)
                    )
                
                targets[unit_idx] = best_energy
                assigned_units.add(unit_idx)
                
                if len(assigned_units) == len(active_units):
                    return targets
    
        # If no visible targets, explore toward enemy
        for unit_idx, unit_pos, _ in active_units:
            if unit_idx in assigned_units:
                continue
            
            # Find farthest visible tile toward enemy
            best_exploration = unit_pos  # Default to current position
            max_distance = 0
            
            for y in range(self.map_height):
                for x in range(self.map_width):
                    if obs["sensor_mask"][y, x]:
                        dist_to_enemy = manhattan_distance((x, y), enemy_spawn)
                        if dist_to_enemy > max_distance:
                            max_distance = dist_to_enemy
                            best_exploration = (x, y)
            
            targets[unit_idx] = best_exploration
            assigned_units.add(unit_idx)
        
        return targets