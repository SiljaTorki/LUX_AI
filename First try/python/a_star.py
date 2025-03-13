import numpy as np
import sys
from astar import AStar
from environment import NodeType, Space, GameConstants


class LuxAStar(AStar):
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

        return base_cost + nebula_penalty + failed_path_penalty

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

    def is_goal_reached(self, current, goal):
        return np.array_equal(current, goal)


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
    if enemy_positions:
        nearby_enemies = [
            enemy_pos
            for enemy_pos in enemy_positions
            if abs(enemy_pos[0] - ship.pos[0]) + abs(enemy_pos[1] - ship.pos[1]) < 3
        ]
        space.obstacles.extend(nearby_enemies)

    start = ship.get_position()
    path = LuxAStar(space, start, goal, ship.energy).astar(start, goal)

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