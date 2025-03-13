import gymnasium as gym
import numpy as np
from gymnasium import spaces
import numpy as np
from environment import NodeType, ActionType


class LuxEnv2(gym.Env):
    """Custom Environment that follows gym interface"""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, map_size=24, max_units=16, max_steps=100, render_mode=None):
        super(LuxEnv2, self).__init__()

        self.map_size = map_size
        self.max_units = max_units
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.matches_played = 0
        self.match_step = 0
        self.total_matches_in_game = 5
        self.total_steps_taken = 0

        # Internal state
        self.current_step = 0
        self.map = None  # Will be initialized in reset()
        self.units = None  # Will be initialized in reset()
        self.info = None  # Will be initialized in reset()

        # Action space: 0-5 (0=no-op, 1-4=move directions, 5=sap)
        self.action_space = spaces.Discrete(6)

        # Define constants for observation space
        self.num_teams = 2
        self.max_relic_nodes = 6
        self.min_energy_per_tile = -20
        self.max_energy_per_tile = 20
        self.init_unit_energy = 100
        self.max_unit_energy = 400
        self.min_unit_energy = 0
        self.spawn_rate = 3  # Spawn a new unit every N steps
        self.max_steps_per_match = 100
        self.match_wins = np.zeros(2, dtype=np.int32)
        self.fog_of_war = True

        self.prev_team0_energy = 0
        self.prev_team1_energy = 0
        self.energy_gained = 0
        self.energy_lost = 0
        self.successful_saps = 0
        self.failed_saps = 0
        self.relic_points = np.zeros(2, dtype=np.int32)
        self.prev_relic_points = np.zeros(2, dtype=np.int32)
        self.visible_tiles_pct = 0

        self.relic_nodes = []
        self.relic_nodes_mask = np.zeros(self.max_relic_nodes, dtype=np.bool_)

        self.observation_space = spaces.Dict(
            {
                "units_position": spaces.Box(
                    low=-1,
                    high=self.map_size,
                    shape=(self.num_teams, self.max_units, 2),
                    dtype=np.int8,
                ),
                "units_energy": spaces.Box(
                    low=0,
                    high=self.max_unit_energy,
                    shape=(self.num_teams, self.max_units, 1),
                    dtype=np.int32,
                ),
                "units_mask": spaces.MultiBinary((self.num_teams, self.max_units)),
                "sensor_mask": spaces.MultiBinary((self.map_size, self.map_size)),
                "map_features_energy": spaces.Box(
                    low=self.min_energy_per_tile,
                    high=self.max_energy_per_tile,
                    shape=(self.map_size, self.map_size),
                    dtype=np.float64,
                ),
                "map_features_tile_type": spaces.Box(
                    low=-1,
                    high=4,
                    shape=(self.map_size, self.map_size),
                    dtype=np.int8,
                ),
                "relic_nodes_mask": spaces.MultiBinary(self.max_relic_nodes),
                "relic_nodes": spaces.Box(
                    low=-1,
                    high=1,
                    shape=(self.max_relic_nodes, 2),
                    dtype=np.int32,
                ),
                "team_points": spaces.Box(
                    low=0, high=1000, shape=(self.num_teams,), dtype=np.int32
                ),
                "team_wins": spaces.Box(
                    low=0, high=1000, shape=(self.num_teams,), dtype=np.int32
                ),
                "steps": spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32),
                "match_steps": spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
                "remainingOverageTime": spaces.Box(
                    low=0, high=100, shape=(1,), dtype=np.int32
                ),
            }
        )

    def step(self, action):
        """
        Take a step in the environment

        Args:
            action: Int value (0-5) representing the action to take

        Returns:
            observation: Current observation after action
            reward: Reward for the action
            terminated: Whether the episode is terminated
            truncated: Whether the episode is truncated
            info: Additional information
        """
        self.current_step += 1
        self.match_step += 1
        self.total_steps_taken += 1

        active_units_team0 = [
            unit for unit in self.units if unit["team"] == 0 and unit["active"]
        ]
        active_units_team1 = [
            unit for unit in self.units if unit["team"] == 1 and unit["active"]
        ]

        current_team0_energy = sum(unit["energy"] for unit in active_units_team0)
        current_team1_energy = sum(unit["energy"] for unit in active_units_team1)

        # Define movement directions
        directions = [
            (0, 0),  # No-op
            (0, -1),  # Up
            (1, 0),  # Right
            (0, 1),  # Down
            (-1, 0),  # Left
        ]
        # Spawn new units periodically
        if self.current_step % self.spawn_rate == 0 and self.current_step > 0:
            # Count current active units per team
            team0_units = sum(
                1 for unit in self.units if unit["team"] == 0 and unit["active"]
            )
            team1_units = sum(
                1 for unit in self.units if unit["team"] == 1 and unit["active"]
            )

            # Spawn for team 0 if under max limit
            if team0_units < self.max_units:
                # Spawn at origin
                pos_x, pos_y = 0, 0
                if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
                    # If not, search for any open position
                    found_position = False
                    for x in range(self.map_size):
                        for y in range(self.map_size):
                            if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                                pos_x, pos_y = x, y
                                found_position = True
                                break
                        if found_position:
                            break

                    # In the extremely unlikely case no open position exists
                    if not found_position:
                        # Fall back to original position and hope for the best
                        pos_x, pos_y = 0, 0

                self.units.append(
                    {
                        "team": 0,
                        "position": np.array([pos_x, pos_y]),
                        "energy": self.init_unit_energy,
                        "active": True,
                    }
                )

            # Spawn for team 1 if under max limit
            if team1_units < self.max_units:
                # Spawn at opposite corner
                pos_x = self.map_size - 1
                pos_y = self.map_size - 1
                if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
                    # If not, search for any open position
                    found_position = False
                    for x in range(self.map_size):
                        for y in range(self.map_size):
                            if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                                pos_x, pos_y = x, y
                                found_position = True
                                break
                        if found_position:
                            break

                    # In the extremely unlikely case no open position exists
                    if not found_position:
                        # Fall back to original position and hope for the best
                        pos_x, pos_y = self.map_size - 1, self.map_size - 1
                # while self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
                #     pos_x = self.map_size - 1 - ((pos_x + 1) % 3)
                #     pos_y = self.map_size - 1 - ((pos_y + 1) % 3)

                self.units.append(
                    {
                        "team": 1,
                        "position": np.array([pos_x, pos_y]),
                        "energy": self.init_unit_energy,
                        "active": True,
                    }
                )

        active_unit_found = False
        reward = 0
        for unit in self.units:
            if unit["energy"] <= 0:
                unit["active"] = False
            if unit["team"] == 0 and unit["active"] and not active_unit_found:
                active_unit_found = True

                if action == ActionType.STAY.value:
                    # No-op - do nothing
                    pass
                elif ActionType.MOVE_UP.value <= action <= ActionType.MOVE_LEFT.value:
                    # Movement action
                    dx, dy = directions[action]

                    # Calculate new position
                    new_x = unit["position"][0] + dx
                    new_y = unit["position"][1] + dy

                    # Check bounds
                    if 0 <= new_x < self.map_size and 0 <= new_y < self.map_size:
                        # Check if tile is not an asteroid
                        if (
                            self.map["tile_type"][new_x, new_y]
                            != NodeType.ASTEROID.value
                        ):
                            # Check if enough energy
                            if unit["energy"] >= self.info["unit_move_cost"]:
                                # Update position and reduce energy
                                unit["position"] = np.array([new_x, new_y])
                                unit["energy"] -= self.info["unit_move_cost"]

                        if self.map["tile_type"][new_x, new_y] == NodeType.NEBULA.value:
                            unit["energy"] -= self.info["nebula_tile_energy_reduction"]

                elif action == ActionType.SAP.value:
                    # Sap action
                    if unit["energy"] >= int(self.info["unit_sap_cost"]):
                        unit["energy"] -= int(self.info["unit_sap_cost"])

                        # Check for enemy units to sap (within range 1)
                        pos_x, pos_y = unit["position"]
                        successful_sap = False

                        for other_unit in self.units:
                            if (
                                other_unit["team"] != unit["team"]
                                and other_unit["active"]
                            ):
                                other_x, other_y = other_unit["position"]
                                # Manhattan distance <= 1
                                if abs(pos_x - other_x) + abs(pos_y - other_y) <= 1:
                                    other_unit["energy"] -= int(
                                        self.info["unit_sap_cost"]
                                    )
                                    successful_sap = True

                        # Track successful vs failed saps
                        if successful_sap:
                            self.successful_saps += 1
                        else:
                            self.failed_saps += 1

        # Move towards nearest enemy
        self.move_towards_opponent_step()
        # Process environment effects
        self.resolve_collisions()
        self.process_energy_updates()

        # # Update for next step
        self.prev_team0_energy = current_team0_energy
        self.prev_team1_energy = current_team1_energy

        # Check relic nodes and update points
        self.update_environment_objects()
        self.update_relic_nodes()

        # Calculate reward
        reward = self.calculate_reward()

        # Check if episode is done
        terminated = self.matches_played >= self.total_matches_in_game
        truncated = False

        # Return observation, reward, terminated, truncated, info
        info = {
            "team_0_units": sum(
                1 for unit in self.units if unit["team"] == 0 and unit["active"]
            ),
            "team_1_units": sum(
                1 for unit in self.units if unit["team"] == 1 and unit["active"]
            ),
            "match_step": self.current_step % 100,
            "team0_energy": current_team0_energy,
            "team1_energy": current_team1_energy,
        }

        return self.get_observation(), reward, terminated, truncated, info

    def check_match_end(self):
        """Check if the current match has ended based on game rules"""

        # End the match after reaching max steps for this match
        if self.match_step >= self.max_steps_per_match - 1:
            # Determine match winner based on relic points, then energy
            if self.relic_points[0] > self.relic_points[1]:
                winner = 0
            elif self.relic_points[1] > self.relic_points[0]:
                winner = 1
            else:
                # If tied on relic points, check energy
                team0_energy = sum(
                    unit["energy"]
                    for unit in self.units
                    if unit["team"] == 0 and unit["active"]
                )
                team1_energy = sum(
                    unit["energy"]
                    for unit in self.units
                    if unit["team"] == 1 and unit["active"]
                )

                if team0_energy > team1_energy:
                    winner = 0
                elif team1_energy > team0_energy:
                    winner = 1
                else:
                    # Random winner if tied on both
                    winner = np.random.randint(0, 2)
            self.matches_played += 1
            # Reset for next match
            self.reset_match(winner)
            return True

        return False

    def reset_match(self, winner=None):
        """Reset the environment for the next match while preserving match wins"""
        if winner is not None:
            self.match_wins[winner] += 1
        # Update match wins
        self.relic_points = np.zeros(2, dtype=np.int32)
        self.prev_relic_points = np.zeros(2, dtype=np.int32)

        # Reset units with initial energy
        for unit in self.units:
            unit["energy"] = self.init_unit_energy
            unit["active"] = True

            # Reset positions
            if unit["team"] == 0:
                # Player 0 units in top-left
                pos_x, pos_y = 0, 0
                if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
                    # If not, search for any open position
                    found_position = False
                    for x in range(self.map_size):
                        for y in range(self.map_size):
                            if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                                pos_x, pos_y = x, y
                                found_position = True
                                break
                        if found_position:
                            break

                    # In the extremely unlikely case no open position exists
                    if not found_position:
                        # Fall back to original position and hope for the best
                        pos_x, pos_y = 0, 0

                unit["position"] = np.array([pos_x, pos_y])
            else:
                # Player 1 units in bottom-right
                pos_x = self.map_size - 1
                pos_y = self.map_size - 1
                if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
                    # If not, search for any open position
                    found_position = False
                    for x in range(self.map_size):
                        for y in range(self.map_size):
                            if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                                pos_x, pos_y = x, y
                                found_position = True
                                break
                        if found_position:
                            break

                    # In the extremely unlikely case no open position exists
                    if not found_position:
                        # Fall back to original position and hope for the best
                        pos_x, pos_y = self.map_size - 1, self.map_size - 1

                unit["position"] = np.array([pos_x, pos_y])

        # Reset energy tracking variables
        self.prev_team0_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 0 and unit["active"]
        )
        self.prev_team1_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 1 and unit["active"]
        )

        # Reset counters
        self.successful_saps = 0
        self.match_step = 0
        self.failed_saps = 0

    def update_environment_objects(self):
        """Move environment objects like asteroids, nebula tiles, and energy nodes"""

        # Only update environment objects periodically to reduce computation
        if self.current_step % 5 != 0:  # Update every 5 steps
            return

        # Nebula movement (slow drift)
        if (
            hasattr(self.info, "nebula_tile_drift_speed")
            and self.info["nebula_tile_drift_speed"] != 0
        ):
            # Create a copy of the tile map to avoid modifying while iterating
            new_tile_map = self.map["tile_type"].copy()

            for x in range(self.map_size):
                for y in range(self.map_size):
                    if self.map["tile_type"][x, y] == NodeType.NEBULA.value:
                        # Random drift with probability based on drift speed
                        if np.random.random() < abs(
                            self.info["nebula_tile_drift_speed"]
                        ):
                            # Determine drift direction
                            dx, dy = 0, 0
                            if self.info["nebula_tile_drift_speed"] > 0:
                                # Drift toward one of the corners based on position
                                dx = 1 if x < self.map_size // 2 else -1
                                dy = 1 if y < self.map_size // 2 else -1
                            else:
                                # Drift toward center
                                dx = 1 if x < self.map_size // 2 else -1
                                dy = 1 if y < self.map_size // 2 else -1

                            # Check if target position is available (not an asteroid)
                            new_x, new_y = x + dx, y + dy
                            if (
                                0 <= new_x < self.map_size
                                and 0 <= new_y < self.map_size
                            ):
                                if (
                                    new_tile_map[new_x, new_y]
                                    != NodeType.ASTEROID.value
                                ):
                                    # Move the nebula
                                    new_tile_map[new_x, new_y] = NodeType.NEBULA.value
                                    new_tile_map[x, y] = 0  # Clear original position

            # Update the tile map
            self.map["tile_type"] = new_tile_map

        # Energy node movement (create gradients around energy hotspots)
        if (
            hasattr(self.info, "energy_node_drift_speed")
            and self.info["energy_node_drift_speed"] > 0
        ):
            # Find energy hotspots (tiles with high energy values)
            hotspots = []
            threshold = 5  # Tiles with energy above this are considered hotspots

            for x in range(self.map_size):
                for y in range(self.map_size):
                    if self.map["energy"][x, y] > threshold:
                        hotspots.append((x, y))

            # Move energy outward from hotspots
            if hotspots and np.random.random() < self.info["energy_node_drift_speed"]:
                for x, y in hotspots:
                    # Create energy gradient around hotspot
                    magnitude = self.info.get("energy_node_drift_magnitude", 3)
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            if dx == 0 and dy == 0:
                                continue

                            # Apply energy transfer with distance falloff
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
                                if (
                                    self.map["tile_type"][nx, ny]
                                    != NodeType.ASTEROID.value
                                ):
                                    # Move some energy to adjacent tile
                                    energy_transfer = min(
                                        1, self.map["energy"][x, y] * 0.1
                                    )
                                    if energy_transfer > 0:
                                        self.map["energy"][x, y] -= energy_transfer
                                        self.map["energy"][nx, ny] += energy_transfer

    def get_observation(self):
        """Convert internal state to structured observation dict"""
        # Initialize observation components
        units_position = np.full((self.num_teams, self.max_units, 2), -1, dtype=np.int8)
        units_energy = np.zeros((self.num_teams, self.max_units, 1), dtype=np.int32)
        units_mask = np.zeros((self.num_teams, self.max_units), dtype=np.int8)

        # Fill unit information
        for i, unit in enumerate(self.units):
            team = unit["team"]
            unit_idx = i % (self.max_units // 2)  # Index within the team

            if unit["active"]:
                units_position[team, unit_idx] = unit["position"]
                units_energy[team, unit_idx, 0] = np.int32(
                    max(0, min(unit["energy"], self.max_unit_energy))
                )
                units_mask[team, unit_idx] = 1

        # Create sensor mask (simplified - all tiles visible)
        sensor_mask = np.ones((self.map_size, self.map_size), dtype=np.int8)

        # Placeholder for relic nodes - not fully implemented in this simplified version
        relic_nodes_mask = np.zeros(self.max_relic_nodes, dtype=np.int8)
        relic_nodes = np.full((self.max_relic_nodes, 2), -1, dtype=np.int32)

        # Team statistics
        team0_units = sum(
            1 for unit in self.units if unit["team"] == 0 and unit["active"]
        )
        team1_units = sum(
            1 for unit in self.units if unit["team"] == 1 and unit["active"]
        )
        team_points = np.array([team0_units, team1_units], dtype=np.int32)
        team_wins = np.zeros(self.num_teams, dtype=np.int32)

        map_energy = self.map["energy"]
        if map_energy is None:
            map_energy = np.zeros((self.map_size, self.map_size), dtype=np.float64)
        else:
            # Min is -20, max is 20
            map_energy = np.clip(map_energy, -20, 20)

        # Create the observation dictionary
        observation = {
            "units_position": units_position,
            "units_energy": units_energy,
            "units_mask": units_mask,
            "sensor_mask": sensor_mask,
            "map_features_energy": map_energy,
            "map_features_tile_type": self.map["tile_type"],
            "relic_nodes_mask": relic_nodes_mask,
            "relic_nodes": relic_nodes,
            "team_points": team_points,
            "team_wins": team_wins,
            "steps": np.array([self.current_step], dtype=np.int32),
            "match_steps": np.array([self.current_step % 100], dtype=np.int32),
            "remainingOverageTime": np.array(
                [self.remaining_overage_time], dtype=np.int32
            ),
        }

        return observation

    def move_towards_opponent_step(self):
        # For each active opponent unit
        for unit in self.units:
            if unit["team"] == 1 and unit["active"]:
                # Find the nearest player unit
                nearest_dist = float("inf")
                nearest_pos = None

                for other_unit in self.units:
                    if other_unit["team"] == 0 and other_unit["active"]:
                        dist = np.sum(np.abs(unit["position"] - other_unit["position"]))
                        if dist < nearest_dist:
                            nearest_dist = dist
                            nearest_pos = other_unit["position"]

                # If a player unit was found and we have enough energy to move
                if (
                    nearest_pos is not None
                    and unit["energy"] >= self.info["unit_move_cost"]
                ):
                    # If adjacent, try to sap
                    if nearest_dist <= 1 and unit["energy"] >= int(
                        self.info["unit_sap_cost"]
                    ):
                        # Sap the player unit
                        unit["energy"] -= int(self.info["unit_sap_cost"])
                        for other_unit in self.units:
                            if other_unit["team"] == 0 and other_unit["active"]:
                                other_dist = np.sum(
                                    np.abs(unit["position"] - other_unit["position"])
                                )
                                if other_dist <= 1:
                                    other_unit["energy"] -= int(
                                        self.info["unit_sap_cost"]
                                    )

                    # Otherwise move toward the player unit
                    else:
                        # Determine direction to move (simplified)
                        dx = np.sign(nearest_pos[0] - unit["position"][0])
                        dy = np.sign(nearest_pos[1] - unit["position"][1])

                        # Prioritize movement in one direction
                        if abs(dx) > 0 and np.random.random() > 0.5:
                            move_x = unit["position"][0] + dx
                            move_y = unit["position"][1]
                        else:
                            move_x = unit["position"][0]
                            move_y = unit["position"][1] + dy

                        # Check bounds and asteroid tiles
                        if (
                            0 <= move_x < self.map_size
                            and 0 <= move_y < self.map_size
                            and self.map["tile_type"][move_x, move_y]
                            != NodeType.ASTEROID.value
                        ):
                            unit["position"] = np.array([move_x, move_y])
                            unit["energy"] -= self.info["unit_move_cost"]

                # Otherwise, move to a high-energy tile if available
                elif unit["energy"] >= self.info["unit_move_cost"]:
                    # Check energy in adjacent tiles
                    best_energy = self.map["energy"][
                        unit["position"][0], unit["position"][1]
                    ]
                    best_dir = (0, 0)

                    for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                        new_x, new_y = (
                            unit["position"][0] + dx,
                            unit["position"][1] + dy,
                        )
                        if (
                            0 <= new_x < self.map_size
                            and 0 <= new_y < self.map_size
                            and self.map["tile_type"][new_x, new_y]
                            != NodeType.ASTEROID.value
                        ):
                            tile_energy = self.map["energy"][new_x, new_y]
                            if tile_energy > best_energy:
                                best_energy = tile_energy
                                best_dir = (dx, dy)

                    # Move to best adjacent tile
                    if best_dir != (0, 0):
                        unit["position"] = np.array(
                            [
                                unit["position"][0] + best_dir[0],
                                unit["position"][1] + best_dir[1],
                            ]
                        )
                        unit["energy"] -= self.info["unit_move_cost"]

    def resolve_collisions(self):
        """Handle unit collisions"""
        # Check for units in the same position
        for i, unit1 in enumerate(self.units):
            if not unit1["active"]:
                continue

            for j, unit2 in enumerate(self.units[i + 1 :], i + 1):
                if not unit2["active"]:
                    continue

                # If units are in the same position and from different teams
                if (
                    np.array_equal(unit1["position"], unit2["position"])
                    and unit1["team"] != unit2["team"]
                ):

                    # Higher energy unit survives
                    if unit1["energy"] > unit2["energy"]:
                        unit2["active"] = False
                    elif unit2["energy"] > unit1["energy"]:
                        unit1["active"] = False
                    else:
                        # If equal energy, both survive (or could implement random winner)
                        pass

    def calculate_reward(self):
        """Calculate a more comprehensive reward for PPO training"""
        # Count units and calculate total energy
        team0_units = sum(
            1 for unit in self.units if unit["team"] == 0 and unit["active"]
        )
        team1_units = sum(
            1 for unit in self.units if unit["team"] == 1 and unit["active"]
        )

        team0_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 0 and unit["active"]
        )
        team1_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 1 and unit["active"]
        )

        # Base reward components
        reward = 0.0

        # Unit advantage (scaled by maximum possible units)
        unit_diff = (team0_units - team1_units) / max(1, self.max_units // 2)
        reward += 0.3 * unit_diff

        # Energy advantage (more important, scaled by maximum possible energy)
        max_possible_energy = self.max_unit_energy * (self.max_units // 2)
        energy_diff = (team0_energy - team1_energy) / max(1, max_possible_energy)
        reward += 0.5 * energy_diff

        # Relic points (most important win condition)
        relic_diff = (self.relic_points[0] - self.relic_points[1]) / max(
            1, 10
        )  # Assuming max 10 relic points per match
        reward += 1.0 * relic_diff

        # Reward for gaining relic points
        relic_points_gained = self.relic_points[0] - self.prev_relic_points[0]
        if relic_points_gained > 0:
            reward += 0.5 * relic_points_gained  # Bonus for capturing relics

        # Winning a match (terminal reward)
        if self.check_match_end():
            if self.relic_points[0] > self.relic_points[1]:
                reward += 10.0  # Large reward for winning a match
            elif self.relic_points[0] == self.relic_points[1]:
                team0_energy = sum(
                    unit["energy"]
                    for unit in self.units
                    if unit["team"] == 0 and unit["active"]
                )
                team1_energy = sum(
                    unit["energy"]
                    for unit in self.units
                    if unit["team"] == 1 and unit["active"]
                )
                if team0_energy > team1_energy:
                    reward += 10.0

        # Proximity reward - encourage units to be close to active relic nodes
        if team0_units > 0:
            # Calculate average distance to nearest active relic node for each unit
            total_distance = 0
            count = 0
            active_relic_positions = [
                node["position"]
                for i, node in enumerate(self.relic_nodes)
                if node["active"] and self.relic_nodes_mask[i]
            ]

            if active_relic_positions:  # Only if there are active relic nodes
                for unit in self.units:
                    if unit["team"] == 0 and unit["active"]:
                        # Find minimum distance to any active relic node
                        min_distance = float("inf")
                        for relic_pos in active_relic_positions:
                            dist = np.sum(np.abs(unit["position"] - relic_pos))
                            min_distance = min(min_distance, dist)

                        if min_distance != float("inf"):
                            total_distance += min_distance
                            count += 1

                if count > 0:
                    avg_distance = total_distance / count
                    # Convert to a reward (closer is better)
                    proximity_reward = max(0, 1 - (avg_distance / (self.map_size * 2)))
                    reward += 0.3 * proximity_reward

        # Strategic sapping - if we can track successful saps
        if hasattr(self, "successful_saps") and self.successful_saps > 0:
            reward += 0.2 * self.successful_saps
            self.successful_saps = 0  # Reset for next step

        # Penalize failed saps
        if hasattr(self, "failed_saps") and self.failed_saps > 0:
            reward -= 0.1 * self.failed_saps
            self.failed_saps = 0  # Reset for next step

        return reward

    def initialize_relic_nodes(self):
        """Initialize relic nodes across the map"""
        self.relic_nodes = []
        self.relic_nodes_mask = np.zeros(self.max_relic_nodes, dtype=np.bool_)

        # Create some relic nodes in symmetrical positions
        for i in range(self.max_relic_nodes // 2):
            # Random position in first quadrant
            x = np.random.randint(0, self.map_size // 2)
            y = np.random.randint(0, self.map_size // 2)

            # Original position
            self.relic_nodes.append(
                {
                    "position": np.array([x, y]),
                    "spawn_step": np.random.randint(0, self.max_steps_per_match),
                    "active": False,
                }
            )

            # Symmetric position
            self.relic_nodes.append(
                {
                    "position": np.array(
                        [self.map_size - 1 - x, self.map_size - 1 - y]
                    ),
                    "spawn_step": np.random.randint(0, self.max_steps_per_match),
                    "active": False,
                }
            )

        # Set initial visibility for relic nodes
        for i, node in enumerate(self.relic_nodes):
            if node["spawn_step"] == 0:
                node["active"] = True
                self.relic_nodes_mask[i] = True

    def update_relic_nodes(self):
        """Update relic nodes state and check for captures"""
        # Save previous points for reward calculation
        self.prev_relic_points = self.relic_points.copy()

        # Activate nodes based on current step
        for i, node in enumerate(self.relic_nodes):
            if self.current_step >= node["spawn_step"] and not node["active"]:
                node["active"] = True
                self.relic_nodes_mask[i] = True

        # Check for captures
        for i, node in enumerate(self.relic_nodes):
            if not node["active"]:
                continue

            # Check for units from each team at this position
            for team in range(2):
                team_units_at_pos = sum(
                    1
                    for unit in self.units
                    if unit["team"] == team
                    and unit["active"]
                    and np.array_equal(unit["position"], node["position"])
                )

                # If team has units at this relic node position, award points
                if team_units_at_pos > 0:
                    self.relic_points[team] += 1

    def process_energy_updates(self):
        """Update unit energy based on map energy values"""
        for unit in self.units:
            if not unit["active"]:
                continue

            x, y = unit["position"]
            # Add energy from tile
            unit["energy"] += self.map["energy"][x, y]

            # Apply nebula energy reduction if on nebula tile
            if self.map["tile_type"][x, y] == NodeType.NEBULA.value:
                unit["energy"] -= 1

            # Ensure energy is within bounds
            unit["energy"] = max(0, min(unit["energy"], self.max_unit_energy))

            # Deactivate unit if energy reaches 0
            if unit["energy"] <= 0:
                unit["active"] = False

    def reset(self, seed=None, options=None):
        if not hasattr(self, "total_steps_taken") or self.total_steps_taken is None:
            self.total_steps_taken = 0

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.matches_played = 0
        self.match_step = 0
        self.team_wins = np.zeros(2, dtype=np.int32)
        self.total_matches_in_game = 5

        self.energy_gained = 0
        self.energy_lost = 0
        self.successful_saps = 0
        self.failed_saps = 0
        self.relic_points = np.zeros(2, dtype=np.int32)
        self.prev_relic_points = np.zeros(2, dtype=np.int32)
        self.remaining_overage_time = 60

        # Initialize relic nodes (simple implementation)
        self.initialize_relic_nodes()

        # Create a symmetric map with random energy distribution
        # Start with a random noise field
        energy_noise = np.random.uniform(
            -3, 8, (self.map_size // 2, self.map_size // 2)
        )

        # Create a symmetric energy map by mirroring the noise
        energy_map = np.zeros((self.map_size, self.map_size))
        energy_map[: self.map_size // 2, : self.map_size // 2] = energy_noise
        energy_map[: self.map_size // 2, self.map_size // 2 :] = np.fliplr(energy_noise)
        energy_map[self.map_size // 2 :, :] = np.flipud(
            energy_map[: self.map_size // 2, :]
        )

        # Round to integers
        energy_map = np.round(energy_map).astype(np.int32)

        # Create symmetric tile types (nebula and asteroid)
        tile_type_map = np.zeros((self.map_size, self.map_size), dtype=np.int8)

        # Generate nebula and asteroid tiles for one quadrant
        quarter_size = self.map_size // 2
        quarter_rng = np.random.random((quarter_size, quarter_size))

        # Create patterns in the first quadrant

        nebula_quarter = np.zeros_like(quarter_rng, dtype=np.int8)
        nebula_quarter[quarter_rng < 0.12] = NodeType.NEBULA.value

        asteroid_quarter = np.zeros_like(quarter_rng, dtype=np.int8)
        asteroid_quarter[(quarter_rng > 0.88) & (quarter_rng < 0.93)] = (
            NodeType.ASTEROID.value
        )
        quarter_tiles = nebula_quarter + asteroid_quarter

        # Mirror to create symmetry
        tile_type_map[:quarter_size, :quarter_size] = quarter_tiles
        tile_type_map[:quarter_size, quarter_size:] = np.fliplr(quarter_tiles)
        tile_type_map[quarter_size:, :quarter_size] = np.flipud(quarter_tiles)
        tile_type_map[quarter_size:, quarter_size:] = np.flipud(
            np.fliplr(quarter_tiles)
        )

        self.map = {"energy": energy_map, "tile_type": tile_type_map}

        # Initialize units - start at opposite corners
        self.units = []

        # Player 0 units in top-left
        # Find a position without an asteroid
        pos_x, pos_y = 0, 0
        if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
            # If not, search for any open position
            found_position = False
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                        pos_x, pos_y = x, y
                        found_position = True
                        break
                if found_position:
                    break

            # In the extremely unlikely case no open position exists
            if not found_position:
                # Fall back to original position and hope for the best
                pos_x, pos_y = 0, 0

        self.units.append(
            {
                "team": 0,
                "position": np.array([pos_x, pos_y]),
                "energy": self.init_unit_energy,
                "active": True,
            }
        )

        # Player 1 units in bottom-right
        pos_x = self.map_size - 1
        pos_y = self.map_size - 1
        if self.map["tile_type"][pos_x, pos_y] == NodeType.ASTEROID.value:
            # If not, search for any open position
            found_position = False
            for x in range(self.map_size):
                for y in range(self.map_size):
                    if self.map["tile_type"][x, y] != NodeType.ASTEROID.value:
                        pos_x, pos_y = x, y
                        found_position = True
                        break
                if found_position:
                    break

            # In the extremely unlikely case no open position exists
            if not found_position:
                # Fall back to original position and hope for the best
                pos_x, pos_y = self.map_size - 1, self.map_size - 1

        self.units.append(
            {
                "team": 1,
                "position": np.array([pos_x, pos_y]),
                "energy": self.init_unit_energy,
                "active": True,
            }
        )

        # Add some "energy nodes" - areas with higher energy values
        for _ in range(4):
            # Choose a random location in the first quadrant
            x = np.random.randint(0, self.map_size // 2)
            y = np.random.randint(0, self.map_size // 2)

            # Create a small energy hotspot
            radius = np.random.randint(2, 4)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        # Apply to all symmetrical positions
                        self.set_symmetric_energy(
                            x, y, dx, dy, np.random.randint(5, 12)
                        )
        self.prev_team0_energy = np.int32(
            sum(
                unit["energy"]
                for unit in self.units
                if unit["team"] == 0 and unit["active"]
            )
        )
        self.prev_team1_energy = np.int32(
            sum(
                unit["energy"]
                for unit in self.units
                if unit["team"] == 1 and unit["active"]
            )
        )
        # Get observation and info
        observation = self.get_observation()
        self.info = {
            "team_0_units": self.max_units // 2,
            "team_1_units": self.max_units // 2,
            "map_size": self.map_size,
            "unit_move_cost": np.random.choice(list(range(1, 6))),
            "unit_sensor_range": np.random.choice([1, 2, 3, 4]),
            "nebula_tile_vision_reduction": np.random.choice(list(range(0, 8))),
            "nebula_tile_energy_reduction": np.random.choice([0, 1, 2, 3, 5, 25]),
            "unit_sap_cost": np.random.choice(list(range(30, 51))),
            "unit_sap_range": np.random.choice(list(range(3, 8))),
            "unit_sap_dropoff_factor": np.random.choice([0.25, 0.5, 1]),
            "unit_energy_void_factor": np.random.choice([0.0625, 0.125, 0.25, 0.375]),
            "nebula_tile_drift_speed": np.random.choice(
                [-0.15, -0.1, -0.05, -0.025, 0.025, 0.05, 0.1, 0.15]
            ),
            "energy_node_drift_speed": np.random.choice([0.01, 0.02, 0.03, 0.04, 0.05]),
            "energy_node_drift_magnitude": np.random.choice(list(range(3, 6))),
        }
        return observation, self.info

    def set_symmetric_energy(self, center_x, center_y, dx, dy, value):
        """Helper method to set energy values symmetrically across the map"""
        positions = [
            (center_x + dx, center_y + dy),
            (self.map_size - 1 - center_x - dx, center_y + dy),
            (center_x + dx, self.map_size - 1 - center_y - dy),
            (self.map_size - 1 - center_x - dx, self.map_size - 1 - center_y - dy),
        ]

        for x, y in positions:
            if 0 <= x < self.map_size and 0 <= y < self.map_size:
                self.map["energy"][x, y] = np.int32(
                    max(0, min(value, self.max_unit_energy))
                )

    def render(self):
        # Render the environment to the screen
        if self.render_mode != "human":
            return

        # Print game info
        print(f"\nStep: {self.current_step} (Match step: {self.current_step % 100})")
        print(
            f"Team 0 units: {sum(1 for unit in self.units if unit['team'] == 0 and unit['active'])}"
        )
        print(
            f"Team 1 units: {sum(1 for unit in self.units if unit['team'] == 1 and unit['active'])}"
        )

        # Get total energy for each team
        team0_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 0 and unit["active"]
        )
        team1_energy = sum(
            unit["energy"]
            for unit in self.units
            if unit["team"] == 1 and unit["active"]
        )
        print(f"Team 0 energy: {team0_energy}")
        print(f"Team 1 energy: {team1_energy}")

        # Print the map with a border for clarity
        print("+" + "-" * (self.map_size * 5) + "+")

        for y in range(self.map_size):
            row = "|"
            for x in range(self.map_size):
                # Check for units first
                unit_found = False
                for unit in self.units:
                    if unit["active"] and np.array_equal(unit["position"], [x, y]):
                        team_char = (
                            "🔵" if unit["team"] == 0 else "🔴"
                        )  # Unicode circles for units
                        energy = str(unit["energy"]).rjust(3)
                        row += f"{team_char}{energy}"
                        unit_found = True
                        break

                if not unit_found:
                    # Then show tile information
                    if self.map["tile_type"][x, y] == NodeType.NEBULA.value:
                        row += " 🌫️  "  # Fog emoji for nebula
                    elif self.map["tile_type"][x, y] == NodeType.ASTEROID.value:
                        row += " 🪨  "  # Rock emoji for asteroid
                    else:
                        # Show energy value
                        energy_val = str(self.map["energy"][x, y]).rjust(4)
                        row += energy_val

            row += "|"
            print(row)

    def close(self):
        pass
