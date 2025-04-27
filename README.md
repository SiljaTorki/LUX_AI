# Lux AI Season 3 Reinforcement Learning Project

This project was built for the [Lux AI Season 3 Competition on Kaggle](https://www.kaggle.com/competitions/lux-ai-season-3).

**About the Competition:**
With the help of over 600 space organizations, Mars has been successfully terraformed and colonized. As colonists look beyond Mars, deep-space telescopes reveal ancient relics hidden among asteroids and nebula gas. To uncover the secrets of a lost civilization, expeditions are launched to explore these relics. In Lux AI Season 3, participants design AI agents that control fleets of ships competing 1v1 to explore the unknown, gather energy, secure relics, and outmaneuver rivals in dynamic, fog-of-war environments.

Agents must balance multi-variable optimization, resource gathering, path planning, and strategic decision-making to succeed. Reinforcement learning is optional; participants can use any method to design effective agents.

In this project, we implement three different approaches to tackle the competition challenge:

- A Base RL Wrapper for training fundamental behaviors.

- A Static Planner Wrapper integrating A\* pathfinding for efficient navigation.

- A MAPPOHR Wrapper combining dynamic replanning (D\* Lite) and multi-agent coordination through reinforcement learning.

The design is inspired by strong baselines and cutting-edge research:

- [Yizhe Wang's PPO Starter Notebook](https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3/notebook#ppo_game_env.py)

- [MAPPOHR Research Paper (Multi-Agent Path Planning)](https://arxiv.org/abs/2306.01270)

## Table of Contents

- [Project Structure](#project-structure)
- [About the Game](#about-the-game)
  - [Observation Space](#observation-space)
  - [Action Space](#action-space)
  - [Win Conditions](#win-conditions)
- [Agent Implementations](#agent-implementations)
  - [Base Wrapper](#1-base-wrapper)
  - [Static Planner Wrapper](#2-static-planner-wrapper)
  - [MAPPOHR Wrapper](#3-mappohr-wrapper)
- [Reward Structures](#reward-structures)
  - [Base Wrapper Rewards](#base-wrapper-rewards)
  - [Static Planner Wrapper Rewards](#static-planner-wrapper-rewards)
  - [MAPPOHR Wrapper Rewards](#mappohr-wrapper-rewards)
- [Pathfinding Algorithms](#pathfinding-algorithms)
- [PPO Network Architecture](#ppo-network-architecture)
- [Training Setup](#training-setup)
- [Acknowledgments](#acknowledgments)

## Project Structure

```
LUX_AI/
├── wrappers/
│   ├── base_wrapper.py            # Base wrapper with core RL functionality
│   ├── static_planner_wrapper.py  # Adds static path planning (A*)
│   ├── mappo_wrapper.py           # Adds dynamic planning (D* Lite) + MAPPO coordination
│   └── utils/
│       └── path_finding.py        # A* and D* Lite pathfinding algorithms
├── common/
│   ├── environment.py             # Game constants and enums
│   └── helper.py                  # Helper functions (env setup, callbacks)
├── training/
│   ├── train_base.ipynb           # Train base wrapper
│   ├── train_static_planner.ipynb # Train static planner wrapper
│   └── train_MAPPOHR.ipynb        # Train MAPPOHR wrapper
├── lux/
│   ├── kit.py                     # Competition kit
│   ├── utils.py                   # Competition utils
├── agent.py                       # Core agent code
├── main.py                        # Local evaluation script
├── replay.html                    # Match replays
├── README.md                      # (This document)
└── requirements.txt               # Python dependencies
```

## About the Game

In Lux AI Season 3, two teams compete over a sequence of 5 matches. Each team controls units to explore the map, gather energy, and control relic nodes while managing resources and vision constraints.

### Observation Space

The original observation space from the competition provides structured information about the game state. It includes:

- **Units:** Their position, energy, and visibility (mask).

- **Map Features:** Tile energy levels and types (empty, asteroid, or nebula).

- **Relic Nodes:** Positions and visibility masks.

- **Team Scores and Game State:** Team points, wins, current steps.

- **Environment Settings:** Such as map size, move cost, and sap range.

The Base Wrapper defines this structure as a `spaces.Dict` in the observation space, splitting and normalizing information as needed. The Static Planner and MAPPOHR Wrappers build on top of it.

### Action Space

Each unit selects from 6 discrete actions:

- 0: Stay

- 1-4: Move (Up, Right, Down, Left)

- 5: Sap (with target coordinates)

### Win Conditions

To win the overall 5-match game:

- Win more matches than your opponent.

- Win a match by earning more relic points.

- If tied on points, the team with more total energy wins.

- If energy is tied, a random winner is chosen.

## Agent Implementations

### 1. Base Wrapper

The Base Wrapper sets up a clean and flexible starting point for training agents in Lux AI Season 3 using Stable Baselines3. The original competition environment provides raw observations that are not directly usable for the Stable Baselines3. So the wrapper reorganizes key information such as unit positions, energy levels, map visibility, relic node locations, and team scores into a structured, normalized Gym-style observation space that Stable Baselines3 can work with. It also defines a custom reward function designed for the Lux environment, guiding the PPO agent's learning by encouraging behaviors like exploring the map, collecting energy, controlling relics, and avoiding unnecessary energy losses. With these improvements, the Base Wrapper makes it possible to train agents easily while also laying the foundation for the static path planning and multi-agent wrappers. See [Reward Structures](#base-wrapper-rewards) for more about the rewards.

- **Inspired by:** [PPO SB3 Kaggle Starter](https://www.kaggle.com/code/yizhewang3/ppo-stable-baselines3/notebook#ppo_game_env.py).

### 2. Static Planner Wrapper

The Static Planner Wrapper builds on top of the Base Wrapper by adding global path planning using the A\* algorithm. At the start of each match, a static path is planned once for each unit toward important targets like relics or energy nodes. These paths do not consider moving obstacles or enemies and are not updated during the match. By giving units a path to follow, the Static Planner encourages exploration. See [Reward Structures](#static-planner-wrapper-rewards) for more about the rewards used in this wrapper.

### 3. MAPPO Wrapper

The MAPPOHR Wrapper builds on top of the Static Planner Wrapper and implements the steps needed for MAPPOHR as described in the original research paper. It introduces dynamic replanning with D\* Lite, allowing units to adjust their paths in real-time when obstacles or enemies appear. It also uses a multi-agent reinforcement learning approach (MAPPO) where agents share information during training but act independently during matches. On top of that, it adds heuristic rules to guide agents toward safer and smarter actions. The goal of the MAPPOHR Wrapper is to make the agents much more adaptive and able to work together in dynamic and challenging environments. See [Reward Structures](#mappohr-wrapper-rewards) for details about the rewards used here.

## Reward Structures

### Base Wrapper Rewards

### Exploration Rewards

- **New Tile Discovery**: +15.0 for discovering a unvisited tile.
- **Map Coverage**: Graduated rewards based on total map coverage:
  - +10.0 bonus when reaching 50% explored for the first time.
  - +5.0 bonus when reaching 25% explored for the first time.
  - +1.0 for exploring a new tile (base reward).
- **Global Exploration**: +0.2 for each newly discovered tile in the current step.
- **Frontier Direction**: +5.0 for moving toward unexplored frontiers.

### Relic and Point Rewards

- **Point Generation**: +8.0 reward for earning points by controlling relic nodes.
- **Standing on a Point Tile**: +5.0 reward each step for staying on a known relic point tile.
- **Long-Term Point Holding**: +2.0 bonus for maintaining control of a relic point tile over multiple turns.
- **Relic Proximity**:
  - +2.0 for being within 2 tiles of a relic node.
  - +0.5 for being within 4 tiles of a relic node.
- **Relic Area Exploration**: +1.5 reward for exploring a new tile near a relic node (+1.0 extra if very close).

### Movement and Positioning Rewards

- **Distance from Spawn**: +0.5 × normalized distance from spawn (encourages wider exploration).
- **Unit Dispersion**: +0.5 × normalized dispersion (rewards units for spreading out).
- **Direction to Enemy**: In early game, reward for moving toward enemy spawn.
- **Energy Field Collection**: +0.3 × energy value × energy factor (more reward if unit has lower energy).

### Sap Action Rewards

- **Successful Sap**: +15.0 base reward for hitting enemy units.
- **Multiple Enemies**: +6.0 for each extra enemy hit beyond the first.
- **Energy Management during Sap**: +2.0 × energy ratio (bonus for using sap effectively).

### Penalties

- **Sap Misses**:
  - -1.0 if enemies were in range but missed.
  - -0.5 if no enemies were in range (smaller penalty).
- **Low Energy Sap**: -1.0 for using sap when unit energy is too low.
- **Invalid Movement**: -0.3 for trying to move into an obstacle or outside the map.
- **Nebula Tiles**: -0.2 base penalty (scaled by the nebula’s energy reduction effect).
- **Staying in Same Area**: -10.0 for staying within 3 unique positions over 10 steps (unless sitting on a point tile).
- **Revisiting Tiles**: -5.0 for revisiting the same tile in the same step (prevents oscillation behavior).

### State-Based Adjustments

- **Energy Maintenance**: +0.1 × energy ratio (reward for keeping unit energy levels high).
- **Dynamic Weighting**: Gradually shifts reward focus from exploration to relic control as the match progresses and the map becomes explored.

### Static Planner Wrapper Rewards

The Static Planner Wrapper uses the same reward structure as the Base Wrapper, with two important additions:

- **Strategic Sap near Relics**: +2.0 extra reward for using sap when close to a relic node (to encourage territory control).
- **Strategic Sap in Key Areas**: +2.0 extra reward for using sap in strategic locations on the map (important control zones).

These extra rewards encourage units to use sap not just randomly, but in ways that help secure important parts of the map and gain an advantage over opponents.

### MAPPOHR Wrapper Rewards

The MAPPOHR Wrapper builds on top of the reward structures from both the Base Wrapper and the Static Planner Wrapper, and introduces additional rewards and penalties focused on team coordination:

- **Penalty for All Waiting**: -2.0 if all agents execute a \"wait\" action at the same time (discourages passivity).
- **Penalty for All Replanning**: -2.0 if all agents execute a \"replan\" action at the same time (prevents unnecessary replanning).
- **Collision Penalty**: -1.5 if units at risk of collision choose contradictory actions (poor movement coordination).
- **Collision Avoidance Bonus**: +2.0 if agents successfully avoid previously detected collision risks.

These additional rewards and penalties help agents learn better team strategies, avoid traffic jams, and coordinate movement and actions in a dynamic environment.

## Pathfinding Algorithms

Implemented in wrappers/utils/path_finding.py:

- A\* (StaticPlanner): Preplans optimal paths to relics and energy nodes.

- D Lite\* (DynamicPlanner): On-the-fly path replanning for dynamic environments.

## PPO Network Architecture

### Base Wrapper and Static Planner Wrapper

- **Policy:** MultiInputPolicy

- **Network Architecture:**
  - Hidden layers: [512, 256, 128]
  - Activation: ReLU

### MAPPOHR Wrapper

- **Features Extractor:** LuxFeaturesExtractor

  - Encodes unit info, map features, relics, and game state separately
  - Combines into a single 512-dimensional feature vector

- **Policy:** RNNPolicy

  - Adds LSTM after feature extraction for temporal memory

- **Network Architecture:**
  - Policy head (pi) and value head (vf): [128, 64]
  - Activation: ReLU

## Training Setup

The same general training setup is used for all wrappers:

- **Algorithm**: PPO (Proximal Policy Optimization)

- **Learning Rate**:

  - Base/Static: linearly scheduled from 5e-4 to 1e-4
  - MAPPOHR: linearly scheduled from 1e-3 to 5e-5

- **Steps per Rollout (n_steps)**: 2048

- **Batch Size**: 128

- **Number of Epochs**: 5

- **Discount Factor (gamma)**: 0.99

- **GAE Lambda**: 0.95

- **Clip Range**: 0.2

- **Value Function Coefficient (vf_coef)**: 0.25

- **Entropy Coefficient (ent_coef)**: 3.0

- **Clip Range VF**: 0.2

- **Parallel Environments**: 8 (`SubprocVecEnv`)

- **Normalization**: Observations and rewards (`VecNormalize`)

- **Callbacks**:

  - Enhanced Tensorboard logging
  - Checkpoint saving every 1000 steps

- **Device**: MPS (Apple Silicon GPU) if available, otherwise CPU

- **Training Length**: 1,000,000 total timesteps

## Acknowledgments

- [Lux AI Challenge](https://www.kaggle.com/competitions/lux-ai-season-3)

- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/index.html)

- [MAPPOHR Paper](https://arxiv.org/abs/2306.01270)

- [python-astar](https://pypi.org/project/astar/)
