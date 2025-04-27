from astar import AStar
import numpy as np
from common.environment import GameConstants, ActionType
import heapq
from collections import defaultdict


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

    def __init__(
        self,
        cost_map=None,
        map_width=GameConstants.MAP_WIDTH,
        map_height=GameConstants.MAP_HEIGHT,
    ):
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
                        self.cost_map[y, x] = float("inf")

                    # Nebula (higher cost)
                    elif tile_type == 1:
                        self.cost_map[y, x] = 5.0

                    # If it has energy, reduce the cost
                    energy = obs["map_features_energy"][y, x]
                    if energy > 0:
                        # Scale discount by energy amount, but don't go below 0.5
                        discount = min(0.5, energy / 100.0)
                        if self.cost_map[y, x] != float(
                            "inf"
                        ):  # Don't discount unwalkable tiles
                            self.cost_map[y, x] = max(
                                0.5, self.cost_map[y, x] - discount
                            )

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
            return float("inf")

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
                if self.cost_map[ny, nx] < float("inf"):
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

    def __init__(
        self, map_width=GameConstants.MAP_WIDTH, map_height=GameConstants.MAP_HEIGHT
    ):
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
            return ActionType.MOVE_UP.value 
        elif dx == 1 and dy == 0:
            return ActionType.MOVE_RIGHT.value  
        elif dx == 0 and dy == 1:
            return ActionType.MOVE_DOWN.value 
        elif dx == -1 and dy == 0:
            return ActionType.MOVE_LEFT.value 
        else:
            return ActionType.STAY.value
    
    def _find_closest_walkable_position(self, target, cost_map):
        """
        Find the closest walkable position to the target.

        Args:
            target: Target position (x, y)
            cost_map: Cost map of the environment

        Returns:
            Closest walkable position (x, y) or None if not found
        """
        x, y = target
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.map_width
                    and 0 <= ny < self.map_height
                    and cost_map[ny, nx] != float("inf")
                ):
                    return (nx, ny)
        return None

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
                obs["units_position"][team_id][unit_idx][1].item(),
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
            if self.astar.cost_map[ty, tx] == float("inf"):
                print(f"Target {target} is on an obstacle")
                # Find closest non-obstacle position
                target = self._find_closest_walkable_position(
                    target, self.astar.cost_map
                )
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
                continue

            # Get unit position
            unit_pos = (
                obs["units_position"][team_id][unit_idx][0].item(),
                obs["units_position"][team_id][unit_idx][1].item(),
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
                    obs["units_position"][team_id][i][1].item(),
                )
                unit_energy = obs["units_energy"][team_id][i][0]
                active_units.append((i, unit_pos, unit_energy))

        # Find visible relic nodes (highest priority)
        visible_relics = []
        for i in range(len(obs["relic_nodes_mask"])):
            if obs["relic_nodes_mask"][i]:
                relic_pos = (
                    obs["relic_nodes"][i][0].item(),
                    obs["relic_nodes"][i][1].item(),
                )
                # Check if relic is visible and has valid coordinates
                if (
                    relic_pos[0] >= 0
                    and relic_pos[1] >= 0
                    and obs["sensor_mask"][relic_pos[1], relic_pos[0]]
                ):
                    visible_relics.append(relic_pos)

        # Find visible energy nodes (second priority)
        visible_energy = []
        energy_map = np.array(obs["map_features_energy"])

        for y in range(self.map_height):
            for x in range(self.map_width):
                # Check if tile is visible, has energy, and is not an asteroid
                if (
                    obs["sensor_mask"][y, x]
                    and energy_map[y, x] > 0
                    and obs["map_features_tile_type"][y, x] != 2
                ):
                    visible_energy.append((x, y, energy_map[y, x]))

        # Sort energy targets by value
        visible_energy.sort(key=lambda x: x[2], reverse=True)

        # Determine enemy spawn location for exploration
        enemy_spawn = (
            (self.map_width - 1, self.map_height - 1) if team_id == 0 else (0, 0)
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
                    visible_relics, key=lambda r: manhattan_distance(unit_pos, r)
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
                        key=lambda e: visible_energy[energy_nodes.index(e)][2],
                    )
                else:  # High energy - prioritize proximity
                    best_energy = min(
                        energy_nodes, key=lambda e: manhattan_distance(unit_pos, e)
                    )

                targets[unit_idx] = best_energy
                assigned_units.add(unit_idx)

                if len(assigned_units) == len(active_units):
                    return targets

        # If no visible targets, explore toward enemy
        if active_units:
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
        else:
            # No active units, assing enemy start position
            for unit_idx in range(GameConstants.MAX_UNITS):
                targets[unit_idx] = enemy_spawn
                assigned_units.add(unit_idx)

        return targets
    
class State:
    """
    Represents a state (position) in the grid for D* Lite algorithm.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __str__(self):
        return f"({self.x}, {self.y})"
    

class DStarLitePlanner:
    """
    Dynamic local planner using D* Lite algorithm for path planning
    with moving obstacles and fog of war in the Lux AI environment.
    """  
    def __init__(self):
            self.accumylated_cost_changes = 0  # Accumulated cost changes
            self.update_priority_queue = []  # Priority queue for states to update
            self.right_hand_side = defaultdict(lambda: float('inf'))  # Right-hand side values
            self.cost_to_go = defaultdict(lambda: float('inf'))  # Cost-to-go values
            self.paths = {}  # Cached paths for units
            self.targets = {}  # Target positions for units
            self.last_known_map = None  # Last observed map state
            self.cost_map = None  # Cost map for path planning
            self.dynamic_obstacles = {}  # Track dynamic obstacles
            self.unit_positions = {}  # Track unit positions
            
    def _heuristic(self, s1, s2):
        """
        Heuristic function for D* Lite.
        
        Args:
            s1: First state (x, y)
            s2: Second state (x, y)
            
        Returns:
            Heuristic cost (Manhattan distance)
        """
        return manhattan_distance((s1.x, s1.y), (s2.x, s2.y))
    
    def calculate_key(self, state, start):
        """
        Calculate priority key for a state.
        
        Args:
            state: State (x, y)
            start: Start state (x, y)
            
        Returns:
            List of two values: [key1, key2]
        """
        
        if self.cost_to_go[state] > self.right_hand_side[state]:
            return [self.right_hand_side[state] + self._heuristic(start, state) + self.accumylated_cost_changes, self.right_hand_side[state]]
        else:
            return [self.cost_to_go[state] + self._heuristic(start, state) + self.accumylated_cost_changes, self.cost_to_go[state]]
    
    def update_vertex(self, update_state, start):
        """
        Update the vertex update_state in the priority queue.
        
        Args:
            update_state: State (x, y)
            start: Start state (x, y)
        """
        
        if update_state != self.goal:
            min_cost = float('inf')
            for s_next in self.get_neighbors(update_state):
                cost = self.get_cost(update_state, s_next)
                if cost + self.cost_to_go[s_next] < min_cost:
                    min_cost = cost + self.cost_to_go[s_next]
            self.right_hand_side[update_state] = min_cost
        
        # Remove u from U if it exists
        for i, (_, state) in enumerate(self.update_priority_queue):
            if state == update_state:
                self.update_priority_queue.pop(i)
                break
        
        # If g and rhs are inconsistent, add u to U
        if self.cost_to_go[update_state] != self.right_hand_side[update_state]:
            heapq.heappush(self.update_priority_queue, (self.calculate_key(update_state, start), update_state))
        
    def compute_shortest_path(self, start):
        """
        Compute the shortest path from start to goal using D* Lite.
        
        Args:
            start: Start state (x, y)

        """
        while (len(self.update_priority_queue) > 0 and 
            (self.update_priority_queue[0][0] < self.calculate_key(start, start) or 
                self.right_hand_side[start] > self.cost_to_go[start])):
            k_old, u = heapq.heappop(self.update_priority_queue)
            k_new = self.calculate_key(u, start)
            
            if k_old < k_new:
                heapq.heappush(self.update_priority_queue, (k_new, u))
            elif self.cost_to_go[u] > self.right_hand_side[u]:
                self.cost_to_go[u] = self.right_hand_side[u]
                for s_prev in self.get_predecessors(u):
                    self.update_vertex(s_prev, start)
            else:
                self.cost_to_go[u] = float('inf')
                self.update_vertex(u, start)
                for s_prev in self.get_predecessors(u):
                    self.update_vertex(s_prev, start)
        
    def get_neighbors(self, state):
        """
        Get valid neighboring states for state.
        
        Args:
            state: State (x, y)
            
        Returns:
            neighbors: List of neighboring states
        """
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Up, Right, Down, Left
        neighbors = []
        
        for dx, dy in directions:
            nx, ny = state.x + dx, state.y + dy
            
            # Check if within map bounds
            if 0 <= nx < GameConstants.MAP_SIZE and 0 <= ny < GameConstants.MAP_SIZE:
                # Check if not an obstacle in the cost map
                if self.cost_map[ny][nx] < float('inf'):
                    neighbors.append(State(nx, ny))
        
        return neighbors
        
    def get_predecessors(self, state):
        """
        Get valid predecessors of state (same as neighbors for grid-based maps).
        
        Args:
            state: State (x, y)
            
        Returns:
            List of predecessor states
        """
        return self.get_neighbors(state)
    
    def get_cost(self, state1, state2):
        """
        Get cost of moving from state1 to state2.
        Adjacent movement cost based on the cost map.
        
        Args:
            state1: First state (x, y)
            state2: Second state (x, y)
            
        Returns:
            base_cost: Cost of moving from state1 to state2
        """
        # Check if adjacent
        if abs(state1.x - state2.x) + abs(state1.y - state2.y) != 1:
            return float('inf')
        
        # Base cost - unit_move_cost or higher if in nebula
        base_cost = self.cost_map[state2.y][state2.x]
        
        # Check for dynamic obstacles (other units)
        if (state2.x, state2.y) in self.dynamic_obstacles:
            return float('inf')
        
        return base_cost
    
    def update_cost_map(self, obs):
        """
        Update the cost map based on the current observation.
        
        Args:
            obs: Observation from the environment
        """
        
        # Initialize cost map with base move cost
        move_cost = obs["env_cfg_unit_move_cost"][0]
        width, height = GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT
        
        # Start with default move cost for all tiles
        self.cost_map = np.ones((height, width), dtype=np.float32) * move_cost
        
        # Mark asteroid tiles as impassable
        for y in range(height):
            for x in range(width):
                # Check if tile is an asteroid (impassable)
                if obs["map_features_tile_type"][y][x] == 2:  # Assuming 2 is asteroid
                    self.cost_map[y][x] = float('inf')
                
                # Add extra cost for nebula tiles
                elif obs["map_features_tile_type"][y][x] == 1:  # Assuming 1 is nebula
                    # Increase cost based on nebula energy reduction (if known)
                    # This is a heuristic - you might want to discover the actual value
                    nebula_cost_factor = 2.0  # Arbitrary factor
                    self.cost_map[y][x] = move_cost * nebula_cost_factor
        
        # Update dynamic obstacles from other units' positions
        self.dynamic_obstacles = {}
        
        # Track friendly units
        for i in range(obs["units_mask"].shape[1]):  # Iterate through units
            if obs["units_mask"][0][i] > 0:  # If unit exists
                x, y = obs["units_position"][0][i]
                self.unit_positions[i] = (x, y)
        
        # Track enemy units as obstacles
        for i in range(obs["units_mask"].shape[1]):  # Iterate through enemy units
            if obs["units_mask"][1][i] > 0:  # If enemy unit exists
                x, y = obs["units_position"][1][i]
                self.dynamic_obstacles[(x, y)] = True
        
        # Store this as the last known map
        self.last_known_map = obs
        
    def initialize_search(self, start, goal):
        """
        Initialize D* Lite search from goal to start.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
        """
        # Clear previous search
        self.update_priority_queue = []
        self.right_hand_side = defaultdict(lambda: float('inf'))
        self.cost_to_go = defaultdict(lambda: float('inf'))
        
        self.start = State(start[0], start[1])
        self.goal = State(goal[0], goal[1])
        
        # Initialize search from goal (D* Lite works backward)
        self.right_hand_side[self.goal] = 0
        heapq.heappush(self.update_priority_queue, (self.calculate_key(self.goal, self.start), self.goal))
        
        # Compute initial path
        self.compute_shortest_path(self.start)
        
    def get_next_action(self, current_pos):
        """
        Get the next action along the path.
        
        Args:
            current_pos: Current position (x, y)
            
        Returns:
            Action integer (0-4) corresponding to the move direction
        """
        current = State(current_pos[0], current_pos[1])
        
        # If we've reached the goal or no path exists
        if current == self.goal or self.cost_to_go[current] == float('inf'):
            return 0  # Stay in place
        
        # Find the best neighbor with lowest g-value
        best_neighbor = None
        min_cost = float('inf')
        
        for neighbor in self.get_neighbors(current):
            cost = self.get_cost(current, neighbor) + self.cost_to_go[neighbor]
            if cost < min_cost:
                min_cost = cost
                best_neighbor = neighbor
        
        if best_neighbor is None:
            return 0  # No valid move
        
        # Convert to action code
        dx = best_neighbor.x - current.x
        dy = best_neighbor.y - current.y
        
        if dx == 0 and dy == 0:
            return ActionType.STAY.value 
        elif dx == 0 and dy == -1:
            return ActionType.MOVE_UP.value
        elif dx == 1 and dy == 0:
            return ActionType.MOVE_RIGHT.value 
        elif dx == 0 and dy == 1:
            return ActionType.MOVE_DOWN.value 
        elif dx == -1 and dy == 0:
            return ActionType.MOVE_LEFT.value 
            
        return ActionType.STAY.value  # Default 
        
    def find_targets_for_units(self, obs, player_id):
        """
        Find targets for units based on the current observation.
        Similar to your StaticPathPlanner's implementation but with
        consideration for dynamic obstacles.
        
        Args:
            obs: Observation from the environment
            player_id: ID of the player
            
        Returns:
            targets: Dictionary mapping unit indices to target positions
        """
        targets = {}
        
        # Find active units
        active_units = []
        for i in range(obs["units_mask"][player_id].shape[1]):
            if obs["units_mask"][player_id][i] > 0:  # If unit exists
                unit_pos = tuple(obs["units_position"][player_id][i])
                unit_energy = obs["units_energy"][player_id][i][0]
                active_units.append((i, unit_pos, unit_energy))
        
        # Prioritize targets
        for unit_idx, unit_pos, unit_energy in active_units:
            # Prioritize relic nodes if they're visible
            best_target = None
            best_value = -float('inf')
            
            # Check relic nodes
            for i in range(len(obs["relic_nodes_mask"])):
                if obs["relic_nodes_mask"][i] > 0:  # If relic node is visible
                    relic_pos = tuple(obs["relic_nodes"][i])
                    
                    # Calculate distance
                    distance = abs(relic_pos[0] - unit_pos[0]) + abs(relic_pos[1] - unit_pos[1])
                    
                    # Calculate value (higher for relics, especially if we're close)
                    value = 100 - distance  # Higher value for closer relics
                    
                    if value > best_value:
                        best_value = value
                        best_target = relic_pos
            
            # If no relic nodes visible, check energy nodes
            if best_target is None:
                # Find highest energy tiles
                energy_map = obs["map_features_energy"]
                max_energy = np.max(energy_map)
                
                if max_energy > 0:
                    # Find positions with max energy
                    max_positions = np.where(energy_map == max_energy)
                    if len(max_positions[0]) > 0:
                        # Pick the closest max energy position
                        closest_idx = 0
                        closest_dist = float('inf')
                        
                        for idx in range(len(max_positions[0])):
                            y, x = max_positions[0][idx], max_positions[1][idx]
                            dist = abs(x - unit_pos[0]) + abs(y - unit_pos[1])
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_idx = idx
                        
                        y, x = max_positions[0][closest_idx], max_positions[1][closest_idx]
                        best_target = (x, y)
                        
            # If still no target, explore unexplored areas
            if best_target is None:
                # Get sensor mask to find unexplored areas
                sensor_mask = obs["sensor_mask"]
                
                # Find edge of explored area (boundary between seen and unseen)
                edges = []
                for y in range(GameConstants.MAP_HEIGHT):
                    for x in range(GameConstants.MAP_WIDTH):
                        if sensor_mask[y][x] > 0:  # If this tile is visible
                            # Check if any neighbors are not visible
                            has_unseen_neighbor = False
                            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                                nx, ny = x + dx, y + dy
                                if (0 <= nx < GameConstants.MAP_WIDTH and 
                                    0 <= ny < GameConstants.MAP_HEIGHT and 
                                    sensor_mask[ny][nx] == 0):
                                    has_unseen_neighbor = True
                                    break
                            
                            if has_unseen_neighbor:
                                edges.append((x, y))
                
                if edges:
                    # Pick the closest edge
                    closest_edge = min(edges, key=lambda e: abs(e[0] - unit_pos[0]) + abs(e[1] - unit_pos[1]))
                    best_target = closest_edge
            
            # If all else fails, pick a random position that's not an obstacle
            if best_target is None:
                valid_positions = []
                for y in range(GameConstants.MAP_HEIGHT):
                    for x in range(GameConstants.MAP_WIDTH):
                        if self.cost_map[y][x] < float('inf'):
                            valid_positions.append((x, y))
                
                if valid_positions:
                    best_target = valid_positions[np.random.randint(0, len(valid_positions))]
                else:
                    # Just stay in place
                    best_target = unit_pos
            
            targets[unit_idx] = best_target
        
        return targets
        
    def replan_path(self, unit_idx, obs, player_id, current_pos=None, goal_pos=None):
        """
        Replan a path for a unit when obstacles are detected.
        
        Args:
            unit_idx: Index of the unit
            obs: Observation from the environment
            player_id: ID of the player
            current_pos: Current position of the unit (optional)
            goal_pos: Goal position for the unit (optional)

        """
        # Update the cost map
        self.update_cost_map(obs)
        
        # Get current position if not provided
        if current_pos is None:
            if obs["units_mask"][0][unit_idx] > 0:
                current_pos = tuple(obs["units_position"][player_id][unit_idx])
            else:
                return  # Unit doesn't exist
        
        # Get goal position if not provided
        if goal_pos is None:
            if unit_idx in self.targets:
                goal_pos = self.targets[unit_idx]
            else:
                return  # No target for this unit
        
        # Check if replanning is necessary
        if unit_idx not in self.paths:
            # Initialize a new path
            self.initialize_search(current_pos, goal_pos)
            self.paths[unit_idx] = {
                'start': current_pos,
                'goal': goal_pos,
                'last_pos': current_pos,
            }
        else:
            # Update k_m for changed edge costs
            current = State(current_pos[0], current_pos[1])
            self.accumylated_cost_changes += self._heuristic(self.paths[unit_idx]['last_pos'], current)
            
            # Reinitialize search from current position
            self.start = current
            self.compute_shortest_path(self.start)
            
            # Update path info
            self.paths[unit_idx]['start'] = current_pos
            self.paths[unit_idx]['last_pos'] = current_pos
        
    def get_next_actions(self, obs, player_id):
        """
        Get the next action for each unit based on its path.
        
        Args:
            obs: Observation from the environment
            player_id: ID of the player
            
        Returns:
            actions: List of actions for each unit
        """
        actions = []
        player_idx = 0 if player_id == "player_0" else 1
        
        # Update cost map based on current observation
        self.update_cost_map(obs)
        
        # Get next action for each unit
        for i in range(GameConstants.MAX_UNITS):
            if obs["units_mask"][0][i] > 0:  # If unit exists
                current_pos = tuple(obs["units_position"][player_idx][i])
                
                # Check if we need to replan
                if i in self.paths:
                    # Check if we reached the goal
                    if current_pos == self.paths[i]['goal']:
                        actions.append(0)  # Stay at goal
                        continue
                    
                    # Check if the path is invalid due to obstacles
                    if self.cost_map[current_pos[1]][current_pos[0]] >= float('inf'):
                        # We're on an obstacle! (shouldn't happen, but just in case)
                        actions.append(0)
                        continue
                    
                    # Get next action from D* Lite
                    try:
                        action = self.get_next_action(current_pos)
                        actions.append(action)
                        
                        # Update last position
                        self.paths[i]['last_pos'] = current_pos
                    except Exception as e:
                        # In case of errors, stay in place
                        print(f"Error in D* Lite: {e}")
                        actions.append(0)
                else:
                    # No path for this unit yet
                    actions.append(0)
            else:
                # Unit doesn't exist
                actions.append(0)
        
        return actions

    def detect_collision_risk(self, obs, unit_idx, next_pos, look_ahead_steps=1):
        """
        Detect if there's a collision risk with enemy units if the unit moves to next_pos.
        
        Args:
            obs: Observation from the environment
            unit_idx: Index of the unit
            next_pos: Next position (x, y) to check for collision
            look_ahead_steps: Number of steps to look ahead for collision prediction
            
        Returns:
            bool: True if collision risk is detected, False otherwise
        """
        if not obs["units_mask"][0][unit_idx]:
            return False  # Unit doesn't exist
        
        # Get enemy unit positions and predict their movement (simple prediction)
        enemy_positions = []
        for i in range(obs["units_mask"].shape[1]):
            if obs["units_mask"][1][i] > 0:  # If enemy unit exists
                enemy_pos = tuple(obs["units_position"][1][i])
                enemy_positions.append(enemy_pos)
                
                # Simple prediction of where enemy might move
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = enemy_pos[0] + dx, enemy_pos[1] + dy
                    if 0 <= nx < GameConstants.MAP_WIDTH and 0 <= ny < GameConstants.MAP_HEIGHT:
                        enemy_positions.append((nx, ny))
        
        # Check if next_pos or positions look_ahead_steps away collide with enemy
        if next_pos in enemy_positions:
            return True
            
        # Look ahead for more sophisticated collision prediction
        if look_ahead_steps > 1:
            # This would implement a more sophisticated collision prediction
            # based on the trajectory of our unit and predicted enemy movements
            pass
        
        return False