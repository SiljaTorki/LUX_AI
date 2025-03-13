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

    def heuristic_cost_estimate(self, current, goal):
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])

    def distance_between(self, n1, n2):
        if n2 in self.obstacles:
            return float("inf")

        base_cost = abs(n1[0] - n2[0]) + abs(n1[1] - n2[1])

        # Dynamic nebula penalty based on ship's energy level
        nebula_penalty = (
            max(1, 10 - self.ship_energy // 100) if n2 in self.nebula_tiles else 0
        )

        # Diminishing penalty for failed paths
        failed_path_count = sum(1 for path in self.failed_paths if (n1, n2) in path)
        failed_path_penalty = min(20, failed_path_count * 5)

        # Reduce cost for paths to relics to prioritize them
        relic_discount = 0
        if self.is_path_to_relic(self.goal):
            relic_discount = 0.2  # 20% discount for paths to relics

        total_cost = base_cost + nebula_penalty + failed_path_penalty
        total_cost = total_cost * (1 - relic_discount)

        return total_cost

    def neighbors(self, node):
        x, y = node
        results = []
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, Right, Down, Left

        for dx, dy in directions:
            newx, newy = x + dx, y + dy

            if (
                (newx, newy) in self.obstacles
                or (newx, newy) in self.dynamic_obstacles
                or (newx, newy) in self.temporary_obstacles
            ):
                continue

            if not (0 <= newx < self.width and 0 <= newy < self.height):
                continue  # Skip out-of-bounds moves

            results.append((newx, newy))

        return results

    def is_path_to_relic(self, goal):
        """Check if this path is leading to a relic node"""
        # This is a simple check - if the goal is a relic node, return true
        return hasattr(self, "relic_nodes") and goal in self.relic_nodes

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

        Args:
            enemy_positions: List of tuples [(pos, energy)] containing enemy positions and energy
            own_energy: Current energy of the unit using this pathfinding
        """
        # Clear previous dynamic obstacles
        self.dynamic_obstacles = []

        for enemy_pos, enemy_energy in enemy_positions:
            # If enemy has more energy, avoid them with a wider margin
            if enemy_energy > own_energy * 1.2:
                # Add the enemy position and surrounding tiles as obstacles
                self.dynamic_obstacles.append(enemy_pos)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    new_x, new_y = enemy_pos[0] + dx, enemy_pos[1] + dy
                    # Make sure the new position is within bounds
                    if 0 <= new_x < self.width and 0 <= new_y < self.height:
                        self.dynamic_obstacles.append((new_x, new_y))
            # If we have energy advantage, don't avoid
            elif own_energy > enemy_energy * 1.5:
                pass  # Don't add to obstacles - we might want to attack
            else:
                # Just avoid direct collision
                self.dynamic_obstacles.append(enemy_pos)


def a_star_search(
    ship, goal, map_features, team_vision, enemy_positions=None, step_count=0
):
    """
    A* search with adaptive energy management and alternative paths.
    """

    space = Space(GameConstants.MAP_WIDTH, GameConstants.MAP_HEIGHT, map_features)
    space.obstacles = ship.detect_obstacles(map_features)
    space.nebula_tiles = ship.detect_nebula(map_features)

    # Add enemy positions as temporary obstacles if they are close
    enemy_positions_with_energy = []
    if enemy_positions:
        # Print structure for debugging
        print(f"Enemy positions type: {type(enemy_positions)}", file=sys.stderr)
        if len(enemy_positions) > 0:
            print(
                f"First enemy position type: {type(enemy_positions[0])}",
                file=sys.stderr,
            )

        # Create a new list to ensure proper formatting
        processed_enemy_positions = []

        for enemy_pos in enemy_positions:
            try:
                # Try to extract x, y depending on data structure
                if isinstance(enemy_pos, tuple) and len(enemy_pos) == 2:
                    # If it's a tuple of (pos, energy) where pos is also a tuple
                    if isinstance(enemy_pos[0], tuple):
                        x, y = enemy_pos[0]
                        energy = enemy_pos[1]
                    else:
                        # Regular (x, y) tuple
                        x, y = enemy_pos
                        energy = 100  # Default energy
                elif isinstance(enemy_pos, (list, np.ndarray)):
                    # If it's a list or array
                    x, y = enemy_pos[0], enemy_pos[1]
                    energy = 100  # Default
                else:
                    # Skip invalid formats
                    continue

                # Convert to integers to be safe
                x, y = int(x), int(y)

                # Only add valid positions
                if (
                    x >= 0
                    and y >= 0
                    and x < GameConstants.MAP_WIDTH
                    and y < GameConstants.MAP_HEIGHT
                ):
                    processed_enemy_positions.append(((x, y), energy))

            except (TypeError, IndexError) as e:
                print(
                    f"Error processing enemy position {enemy_pos}: {e}", file=sys.stderr
                )
                continue

    start = ship.get_position()
    pathfinder = LuxAStar(space, start, goal, ship.energy)

    if enemy_positions_with_energy:
        pathfinder.update_dynamic_obstacles(enemy_positions_with_energy, ship.energy)

    path = pathfinder.astar(start, goal)

    if path:
        return list(path)

    # If no valid path, try alternative goals:
    late_game = step_count > 70
    alt_goal = None

    # Prioritize Energy Nodes if Low Energy
    if ship.energy < 100:
        alt_goal = ship.find_closest(NodeType.ENERGY_NODE, map_features, team_vision)
    elif late_game:
        alt_goal = ship.find_closest(NodeType.RELIC_NODE, map_features, team_vision)
    else:
        # Explore new areas if energy is sufficient
        unexplored_positions = [
            (x, y)
            for x in range(GameConstants.MAP_WIDTH)
            for y in range(GameConstants.MAP_HEIGHT)
            if not team_vision[x, y]
        ]

        if unexplored_positions:
            alt_goal = min(
                unexplored_positions,
                key=lambda p: abs(p[0] - start[0]) + abs(p[1] - start[1]),
            )

    # Try A* on alternative goal
    if alt_goal:
        path = LuxAStar(space, start, alt_goal, ship.energy).astar(start, alt_goal)

    # **Final Fallback: Move Toward Center or Defensive Position**
    if not path:
        center = (GameConstants.MAP_WIDTH // 2, GameConstants.MAP_HEIGHT // 2)
        path = LuxAStar(space, start, center, ship.energy).astar(start, center)

    return list(path) if path else [start]


def find_best_alternative_goal(ship, map_features, team_vision, game_state):
    """Find the best alternative if A* fails."""
    potential_goals = []

    step = game_state.step
    early_game = step < 30
    late_game = step >= 70

    # Exploration Focus (Early Game)
    unexplored_tiles = ship.get_unexplored_tiles(team_vision)
    for tile in unexplored_tiles[:5]:
        potential_goals.append((tile, 10 if early_game else 5))

    # Energy Collection (Mid Game)
    energy_nodes = ship.find_all(NodeType.ENERGY_NODE, map_features, team_vision)
    for node in energy_nodes[:3]:
        potential_goals.append((node, 8 if ship.energy < 150 else 3))

    # Relic Capture (Late Game)
    relic_nodes = ship.find_all(NodeType.RELIC_NODE, map_features, team_vision)
    for node in relic_nodes[:3]:
        potential_goals.append((node, 15 if late_game else 2))

    # Sort by priority/distance
    potential_goals.sort(key=lambda g: ship.distance_to(g[0]) / g[1])

    # Return best valid goal
    for alt_goal, _ in potential_goals:
        return alt_goal
    return ship.get_position()
