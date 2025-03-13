import heapq
import numpy as np
from environment import NodeType, Space, GameConstants
import sys

class LuxAStar:
    def __init__(self, space, start, goal, ship_energy=100, failed_paths=None):
        self.space = space
        self.start = start
        self.goal = goal
        self.width = space.width
        self.height = space.height
        self.obstacles = space.obstacles
        self.nebula_tiles = space.nebula_tiles
        self.ship_energy = ship_energy
        self.failed_paths = failed_paths if failed_paths else set()
        self.dynamic_obstacles = []
        self.temporary_obstacles = []
        self.relic_nodes = []  # Add this to track relic nodes

    def heuristic_cost_estimate(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def distance_between(self, n1, n2):
        # Return infinity for obstacles
        if n2 in self.obstacles:
            return float("inf")
        
        # Base cost is Manhattan distance
        base_cost = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])

        # Apply dynamic nebula penalty based on ship's energy
        # Higher energy = lower penalty
        nebula_penalty = 0
        if n2 in self.nebula_tiles:
            # Max penalty is 10, decreases as energy increases
            nebula_penalty = max(1, 10 - self.ship_energy // 100)

        # Apply penalty for paths that failed before
        # This helps the ship try new routes if it gets stuck
        failed_path_count = sum(1 for path in self.failed_paths if (n1, n2) in path)
        failed_path_penalty = min(20, failed_path_count * 5)

        # Reduce cost for paths to relics to prioritize them
        relic_discount = 0
        if self.is_path_to_relic(self.goal):
            relic_discount = 0.2  # 20% discount for paths to relics

        # Calculate total cost with all factors
        total_cost = base_cost + nebula_penalty + failed_path_penalty
        total_cost = total_cost * (1 - relic_discount)

        return total_cost

    def neighbors(self, node):
        x, y = node
        results = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

        for dx, dy in directions:
            newx, newy = x + dx, y + dy

            # Skip obstacles, dynamic obstacles, and temporary obstacles
            if (
                (newx, newy) in self.obstacles
                or (newx, newy) in self.dynamic_obstacles
                or (newx, newy) in self.temporary_obstacles
            ):
                continue

            # Skip out-of-bounds moves
            if not (0 <= newx < self.width and 0 <= newy < self.height):
                continue

            results.append((newx, newy))

        return results

    def is_path_to_relic(self, goal):
        """Check if this path is leading to a relic node"""
        return goal in self.relic_nodes

    def is_goal_reached(self, current, goal):
        return np.array_equal(current, goal)

    def astar(self, start, goal):
        if self.is_goal_reached(start, goal):
            return [start]

        close_set = set()
        came_from = {}
        gscore = {start: 0}
        fscore = {start: self.heuristic_cost_estimate(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))

        while oheap:
            current = heapq.heappop(oheap)[1]

            if self.is_goal_reached(current, goal):
                data = []
                while current in came_from:
                    data.append(current)
                    current = came_from[current]
                data.append(start)
                data.reverse()
                return data

            close_set.add(current)

            for neighbor in self.neighbors(current):
                if neighbor in close_set:
                    continue

                tentative_g_score = gscore.get(
                    current, float("inf")
                ) + self.distance_between(current, neighbor)

                if tentative_g_score < gscore.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + self.heuristic_cost_estimate(
                        neighbor, goal
                    )
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))

        return False

    def update_dynamic_obstacles(self, enemy_positions, own_energy):
        """
        Update dynamic obstacles based on enemy positions and energy comparisons.
        
        Key improvement: Considers energy advantage/disadvantage when deciding
        whether to avoid enemies or potentially attack them.
        """
        # Clear previous dynamic obstacles
        self.dynamic_obstacles = []

        for enemy_pos, enemy_energy in enemy_positions:
            # Strong enemy (20% stronger) - avoid widely
            if enemy_energy > own_energy * 1.2:
                self.dynamic_obstacles.append(enemy_pos)
                # Add surrounding tiles as obstacles too
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = enemy_pos[0] + dx, enemy_pos[1] + dy
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        self.dynamic_obstacles.append((new_x, new_y))
            # Significantly weaker enemy (50% weaker) - don't avoid (might want to attack)
            elif own_energy > enemy_energy * 1.5:
                pass  # Don't add to obstacles - we might want to attack
            # Otherwise, just avoid direct collision
            else:
                self.dynamic_obstacles.append(enemy_pos)

def a_star_search(
    ship, goal, map_features, team_vision, enemy_positions=None, step_count=0
):
    """
    Enhanced A* search with adaptive energy management and alternative paths.
    
    Key improvements:
    1. Handles enemy positions intelligently
    2. Provides fallback goals if the primary goal is unreachable
    3. Adapts strategy based on game phase (early/late)
    """
    # Create Space object
    space = Space(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, map_features)
    space.obstacles = ship.detect_obstacles(map_features)
    space.nebula_tiles = ship.detect_nebula(map_features)

    # Process enemy positions
    enemy_positions_with_energy = []
    if enemy_positions:
        # Process and normalize enemy position data
        for enemy_pos in enemy_positions:
            try:
                # Extract position and energy based on data format
                if isinstance(enemy_pos, tuple) and len(enemy_pos) == 2:
                    if isinstance(enemy_pos[0], tuple):
                        # Format: ((x, y), energy)
                        x, y = enemy_pos[0]
                        energy = enemy_pos[1]
                    else:
                        # Format: (x, y)
                        x, y = enemy_pos
                        energy = 100  # Default energy
                elif isinstance(enemy_pos, (list, np.ndarray)):
                    # Format: [x, y]
                    x, y = enemy_pos[0], enemy_pos[1]
                    energy = 100  # Default
                else:
                    continue

                # Ensure coordinates are integers
                x, y = int(x), int(y)

                # Only add valid positions
                if (
                    x >= 0
                    and y >= 0
                    and x < GameConstants.MAP_WIDTH
                    and y < GameConstants.MAP_HEIGHT
                ):
                    enemy_positions_with_energy.append(((x, y), energy))

            except (TypeError, IndexError) as e:
                print(f"Error processing enemy position {enemy_pos}: {e}", file=sys.stderr)
                continue

    # Create pathfinder with ship's starting position
    start = ship.get_position()
    pathfinder = LuxAStar(space, start, goal, ship.energy)
    
    # Add visible relic nodes to pathfinder
    # (This assumes you have access to relic_nodes in the observation)
    # If you don't, you can skip this part
    try:
        for i in range(len(ship.relic_nodes)):
            if ship.relic_nodes_mask[i] > 0:
                pathfinder.relic_nodes.append(tuple(ship.relic_nodes[i]))
    except (AttributeError, NameError):
        pass  # Skip if relic_nodes not available

    # Update dynamic obstacles
    if enemy_positions_with_energy:
        pathfinder.update_dynamic_obstacles(enemy_positions_with_energy, ship.energy)

    # Try to find path to goal
    path = pathfinder.astar(start, goal)

    # If path found, return it
    if path:
        return list(path)

    # If no valid path, try alternative goals in priority order:
    late_game = step_count > 70
    alt_goal = None

    # ALTERNATIVE GOAL 1: Energy Node if Low Energy
    if ship.energy < 100:
        alt_goal = ship.find_closest(NodeType.ENERGY_NODE, map_features, team_vision)
        
    # ALTERNATIVE GOAL 2: Relic Node if Late Game
    elif late_game:
        alt_goal = ship.find_closest(NodeType.RELIC_NODE, map_features, team_vision)
        
    # ALTERNATIVE GOAL 3: Unexplored Area
    else:
        # Find unexplored positions
        unexplored_positions = [
            (x, y)
            for x in range(GameConstants.MAP_WIDTH)
            for y in range(GameConstants.MAP_HEIGHT)
            if not team_vision[x, y]
        ]

        if unexplored_positions:
            # Find closest unexplored position
            alt_goal = min(
                unexplored_positions,
                key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]),
            )

    # Try A* on alternative goal
    if alt_goal:
        path = LuxAStar(space, start, alt_goal, ship.energy).astar(start, alt_goal)

    # FINAL FALLBACK: Move Toward Center
    if not path:
        center = (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)
        path = LuxAStar(space, start, center, ship.energy).astar(start, center)

    return list(path) if path else [start]