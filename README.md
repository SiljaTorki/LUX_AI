# Lux AI Season 3

**Our Paper**:  
📄 [Exploring PPO Extensions for Lux AI Season 3: Integrating Planning Heuristics and Multi-Policy Training (Lux_AI_season_3.pdf)](./Lux_AI_season_3.pdf)

This repository contains two separate implementations based on different research papers, as well as our own research report that combines and extends the ideas.

## Implementations

### 1. MAPPOHR: Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning

- **Location**: [`mappohr/`](./mappohr/)
- **Description**:  
  Implementation of a multi-robot path planning system using a hybrid method: a real-time multi-agent reinforcement learning agent (MAPPO) combined with heuristic path planning (e.g., D\* Lite). Robots dynamically replan their paths while avoiding collisions and coordinating efficiently.

---

### 2. multi-agent-rllib

- **Location**: [`multi-agent-rllib/`](./multi-agent-rllib/)
- **Description**:  
  This directory contains the code for the multi -agent and -policy implementation mentioned in the report. It includes necessary environment and agent definitions, + training code to train a two-policy multi-agent PPO model with LSTM blocks. Everything is orchestrated through the main.ipynb notebook.

## Group Work Note

The implementation found in [`multi-agent-rllib/`](./multi-agent-rllib/) represents the outcome of our group collaboration.  
During the group phase, we discussed and explored common strategies, reinforcement learning methods, and reward structures.

However, due to significant challenges with training stability, environment setup, and framework integration, the final solutions diverged.  
As a result, each member's work evolved into separate base agents built from the shared ideas.

- [`multi-agent-rllib/`](./multi-agent-rllib/) reflects the main group work and initial concepts.
- [`mappohr/`](./mappohr/) evolved independently to address technical challenges with different frameworks and training configurations.

Thus, while one implementation traces back to the group work, the two solutions differ substantially due to the technical direction each member pursued.
