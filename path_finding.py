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
            
            # Compute path
            path = list(self.astar.astar(unit_pos, target))
            
            # Store path if found
            if path:
                self.paths[unit_idx] = path
        
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
        
        # Prioritize relic nodes
        relic_targets = []
        for i in range(len(obs["relic_nodes_mask"])):
            if obs["relic_nodes_mask"][i]:
                relic_pos = (
                    obs["relic_nodes"][i][0].item(),
                    obs["relic_nodes"][i][1].item()
                )
                if relic_pos[0] >= 0 and relic_pos[1] >= 0:  # Valid coordinates
                    relic_targets.append(relic_pos)
        
        # Find high energy areas
        energy_map = np.array(obs["map_features_energy"])
        high_energy_targets = []
        
        # Find positions with energy > 0 that we can see
        for y in range(self.map_height):
            for x in range(self.map_width):
                if (obs["sensor_mask"][y, x] and
                    energy_map[y, x] > 0 and
                    obs["map_features_tile_type"][y, x] != 2):  # Not an asteroid
                    high_energy_targets.append((x, y, energy_map[y, x]))
        
        # Sort high energy targets by energy value (highest first)
        high_energy_targets.sort(key=lambda x: x[2], reverse=True)
        
        # Only keep the top N high energy targets
        max_energy_targets = 5
        high_energy_targets = [(x, y) for x, y, _ in high_energy_targets[:max_energy_targets]]
        
        # Exploration targets (corners and center)
        exploration_targets = []
        center = (self.map_width // 2, self.map_height // 2)
        corners = [
            (2, 2),
            (2, self.map_height - 3),
            (self.map_width - 3, 2),
            (self.map_width - 3, self.map_height - 3)
        ]
        exploration_targets.extend(corners)
        exploration_targets.append(center)
        
        # Assign targets based on priority: relic nodes > high energy > exploration
        assigned_units = set()
        
        # First, assign units to relic nodes if we have any
        if relic_targets:
            # Calculate distances between units and relic targets
            distances = []
            for unit_idx, unit_pos, _ in active_units:
                for target_pos in relic_targets:
                    dist = manhattan_distance(unit_pos, target_pos)
                    distances.append((dist, unit_idx, target_pos))
            
            # Sort by distance
            distances.sort()
            
            # Assign units to relic targets
            assigned_targets = set()
            for _, unit_idx, target_pos in distances:
                if unit_idx not in assigned_units and target_pos not in assigned_targets:
                    targets[unit_idx] = target_pos
                    assigned_units.add(unit_idx)
                    assigned_targets.add(target_pos)
                    
                    # Break if all units assigned
                    if len(assigned_units) == len(active_units):
                        break
        
        # Then, assign remaining units to high energy targets
        if high_energy_targets and len(assigned_units) < len(active_units):
            remaining_units = [(i, pos, energy) for i, pos, energy in active_units if i not in assigned_units]
            
            # Calculate distances between remaining units and energy targets
            distances = []
            for unit_idx, unit_pos, unit_energy in remaining_units:
                for target_pos in high_energy_targets:
                    # Closer is better, but prioritize units with lower energy
                    dist = manhattan_distance(unit_pos, target_pos)
                    energy_factor = max(1, 100 / (unit_energy + 1))  # Lower energy = higher priority
                    score = dist / energy_factor
                    distances.append((score, unit_idx, target_pos))
            
            # Sort by score
            distances.sort()
            
            # Assign units to energy targets
            assigned_energy_targets = set()
            for _, unit_idx, target_pos in distances:
                if unit_idx not in assigned_units and target_pos not in assigned_energy_targets:
                    targets[unit_idx] = target_pos
                    assigned_units.add(unit_idx)
                    assigned_energy_targets.add(target_pos)
                    
                    # Break if all units assigned
                    if len(assigned_units) == len(active_units):
                        break
        
        # Finally, assign any remaining units to exploration targets
        if exploration_targets and len(assigned_units) < len(active_units):
            remaining_units = [(i, pos, energy) for i, pos, energy in active_units if i not in assigned_units]
            
            # Calculate distances between remaining units and exploration targets
            distances = []
            for unit_idx, unit_pos, _ in remaining_units:
                for target_pos in exploration_targets:
                    dist = manhattan_distance(unit_pos, target_pos)
                    distances.append((dist, unit_idx, target_pos))
            
            # Sort by distance
            distances.sort()
            
            # Assign units to exploration targets
            assigned_exploration_targets = set()
            for _, unit_idx, target_pos in distances:
                if unit_idx not in assigned_units and target_pos not in assigned_exploration_targets:
                    targets[unit_idx] = target_pos
                    assigned_units.add(unit_idx)
                    assigned_exploration_targets.add(target_pos)
                    
                    # Break if all units assigned
                    if len(assigned_units) == len(active_units):
                        break
        
        return targets