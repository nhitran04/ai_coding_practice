"""Passive ADP on configurable MiniGrids with position-only states and cardinal moves."""

import argparse
import csv
import random
import shutil
from dataclasses import dataclass, field
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar

import imageio.v2 as imageio
import matplotlib
import numpy as np
from gymnasium.spaces import Discrete
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

matplotlib.use("Agg")
import matplotlib.pyplot as plt


State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


class Move(IntEnum):
    """Absolute movement directions, using MiniGrid's direction numbering."""

    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


def random_policy(
    state: tuple[int, int],
    *,
    goal: tuple[int, int] = (4, 1),
) -> dict[Move, float]:
    """Return equal probabilities for cardinal moves; no actions at the goal."""
    if state == goal:
        return {}
    return {move: 0.25 for move in Move}


@dataclass
class MDP(Generic[State, Action]):
    """
    Tabular MDP representation used by the passive ADP agent.

    P stores estimated transition probabilities P(s' | s, a), R stores observed
    transition rewards R(s, a, s'), and A records actions observed in each
    state. The agent learns these tables; discount factor gamma is supplied in
    advance.
    """

    gamma: float
    P: dict[tuple[State, Action], dict[State, float]] = field(default_factory=dict)
    R: dict[tuple[State, Action, State], float] = field(default_factory=dict)
    A: dict[State, set[Action]] = field(default_factory=dict)


class PassiveADPAgent(Generic[State, Action]):
    """AIMA 4e, Figure 22.2: PASSIVE-ADP-LEARNER.

    Source: https://aima.cs.berkeley.edu/algorithms.pdf Method signatures
    include terminal-state handling for episodic MiniGrid runs.
    """

    def __init__(
        self,
        policy: Callable[[State], dict[Action, float]],
        discount_factor: float = 0.99,
    ) -> None:
        """
        Store the stationary stochastic policy and initialize the MDP
        representation and persistent tables.
        """
        # A discount below one guarantees a unique discounted Bellman solution.
        if not 0.0 <= discount_factor < 1.0:
            raise ValueError("discount_factor must be at least 0 and less than 1")

        self.pi = policy  # maps each state to action probabilities
        self.mdp: MDP[State, Action] = MDP(discount_factor)  # MDP
        self.U: dict[State, float] = {}  # table of utilities for states

        self.N: dict[
            tuple[State, Action], dict[State, int]
        ] = {}  # table of outcome count vecctors

        self.s: State | None = None  # previous state
        self.a: Action | None = None  # previous action

    def passive_adp_learning(
        self,
        percept: tuple[State, float],
        *,
        terminal: bool = False,
    ) -> Action | None:
        """
        Learn from (state, reward), evaluate the policy, and return its next
        action.

        The reward belongs to the previous transition. A terminal percept ends
        the episode after that transition is learned and returns no action.
        """
        s_prime, reward = percept
        if s_prime not in self.U:
            self.U[s_prime] = 0

        if self.s is not None:
            assert self.a is not None
            state_action = (self.s, self.a)
            outcomes = self.N.setdefault(state_action, {})
            outcomes[s_prime] = outcomes.get(s_prime, 0) + 1
            self.mdp.R[(self.s, self.a, s_prime)] = reward
            self.mdp.A.setdefault(self.s, set()).add(self.a)
            self.mdp.P[state_action] = self.normalize(outcomes)
            self.U = self.policy_evaluation(self.pi, self.U, self.mdp)

        if terminal:
            self.reset_episode()
            return None

        action_probabilities = self.pi(s_prime)
        self.s = s_prime
        self.a = random.choices(
            list(action_probabilities),
            weights=list(action_probabilities.values()),
        )[0]
        return self.a

    def act(self, percept: tuple[State, float, bool]) -> Action | None:
        """
        Adapt a (state, reward, terminal) percept to passive_adp_learning.
        """
        state, reward, terminal = percept
        return self.passive_adp_learning((state, reward), terminal=terminal)

    @staticmethod
    def normalize(outcome_counts: dict[State, int]) -> dict[State, float]:
        """
        Convert outcome counts into probabilities summing to one.
        """
        total = sum(outcome_counts.values())
        return {state: count / total for state, count in outcome_counts.items()}

    @staticmethod
    def policy_evaluation(
        policy: Callable[[State], dict[Action, float]],
        utilities: dict[State, float],
        mdp: MDP[State, Action],
    ) -> dict[State, float]:
        """
        Return utilities satisfying the fixed-policy Bellman equations.
        """
        threshold = 0.001
        V = utilities.copy()

        while True:
            delta = 0.0
            for state in V:
                v = V[state]

                state_value = 0.0
                for action, action_probability in policy(state).items():
                    action_value = 0.0
                    for successor, probability in mdp.P.get(
                        (state, action), {}
                    ).items():
                        action_value += probability * (
                            mdp.R[(state, action, successor)] + mdp.gamma * V[successor]
                        )
                    state_value += action_probability * action_value

                V[state] = state_value

                delta = max(delta, abs(v - V[state]))

            if delta < threshold:
                return V

    def reset_episode(self) -> None:
        """
        Clear the previous state and action while retaining the learned model
        and utilities.
        """
        self.s = None
        self.a = None


class SimpleEnv(MiniGridEnv):
    """
    Reach the goal in a configurable interior using absolute cardinal movement.

    MiniGrid's orientation is used for rendering and executing each move; the
    agent's state consists only of its (x, y) position.
    """

    def __init__(self, width: int = 4, height: int = 3, max_steps: int = 256, **kwargs):
        """
        Configure playable dimensions (excluding walls) and the episode step limit.
        """
        if width < 1 or height < 1 or width * height == 1:
            raise ValueError("Playable dimensions must be positive with at least two cells")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.playable_width = width
        self.playable_height = height
        self.goal_pos = (width, 1)
        mission_space = MissionSpace(mission_func=self._gen_mission)
        # MiniGrid dimensions include the surrounding walls.
        super().__init__(
            mission_space=mission_space,
            width=width + 2,
            height=height + 2,
            max_steps=max_steps,
            **kwargs,
        )

        self.action_space = Discrete(len(Move))

    def step(self, action: Move | int):
        """Move one cell in an absolute direction, staying put against a wall.

        Set MiniGrid's facing direction before moving, so the previous
        orientation never affects the transition. Each move costs one step.
        """
        self.agent_dir = int(Move(action))
        return super().step(Actions.forward)

    @staticmethod
    def _gen_mission():
        """
        Return the instruction describing the environment's objective
        """
        return "reach the goal"

    def _gen_grid(self, width, height):
        """
        Build the boundary walls and place the agent and goal at their starting
        positions.
        """
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.agent_pos = (1, height - 2)
        self.agent_dir = 0  # Face right.
        self.put_obj(Goal(), width - 2, 1)
        self.mission = self._gen_mission()


def save_heatmap(
    agent: PassiveADPAgent,
    episode: int,
    destination: Path,
    width: int = 4,
    height: int = 3,
) -> None:
    """Save one utility per (x, y) cell; gray cells are unobserved."""
    fig, ax = plt.subplots(figsize=(max(7, width * 0.8), max(5, height * 0.7)), constrained_layout=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("lightgray")
    values = np.full((height, width), np.nan)
    for y in range(1, height + 1):
        for x in range(1, width + 1):
            if (x, y) in agent.U:
                values[y - 1, x - 1] = agent.U[x, y]
    plot = ax.imshow(values, cmap=cmap, vmin=0, vmax=1)
    ax.set(xlabel="x", ylabel="y")
    ax.set_xticks(range(width), labels=range(1, width + 1))
    ax.set_yticks(range(height), labels=range(1, height + 1))
    for y in range(height):
        for x in range(width):
            value = values[y, x]
            text = "unseen" if np.isnan(value) else f"{value:.3f}"
            if (x, y) == (width - 1, 0):
                text = "Goal\n" + text
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                color="white" if not np.isnan(value) and value < 0.5 else "black",
            )
    fig.colorbar(plot, ax=ax, label="Estimated discounted return")
    ax.set_title(f"{width}×{height} utilities after episode {episode} (gray = unseen)")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def train(
    episodes: int = 100,
    checkpoint_every: int = 10,
    output_dir: Path | None = None,
    seed: int = 42,
    width: int = 4,
    height: int = 3,
    max_steps: int = 256,
) -> Path:
    """Reset this size's outputs, recording episodes 1–10 and later checkpoints."""
    if episodes < 1 or checkpoint_every < 1:
        raise ValueError("episodes and checkpoint_every must be positive")
    # Validate the environment before replacing any previous outputs.
    env = SimpleEnv(width=width, height=height, max_steps=max_steps, render_mode="rgb_array")
    output_root = Path(output_dir) if output_dir is not None else Path(__file__).parent / "outputs"
    output_dir = output_root / f"{width}x{height}"
    videos = output_dir / "videos"
    heatmaps = output_dir / "heatmaps"
    for directory in (videos, heatmaps):
        if directory.exists():
            shutil.rmtree(directory)
    videos.mkdir(parents=True, exist_ok=True)
    heatmaps.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    agent = PassiveADPAgent(partial(random_policy, goal=env.goal_pos))
    try:
        with (output_dir / "episodes.csv").open("w", newline="") as metrics:
            log = csv.writer(metrics)
            log.writerow(
                [
                    "episode",
                    "steps",
                    "terminated",
                    "truncated",
                    "environment_return",
                    "learning_return",
                ]
            )
            for episode in range(1, episodes + 1):
                env.reset(seed=seed + episode - 1)
                agent.reset_episode()
                state = tuple(env.agent_pos)
                action = agent.act((state, 0.0, False))
                checkpoint = episode <= 10 or episode % checkpoint_every == 0
                writer = None
                steps = 0
                environment_return = 0.0
                learning_return = 0.0
                try:
                    if checkpoint:
                        writer = imageio.get_writer(
                            videos / f"episode_{episode:04d}.mp4",
                            fps=10,
                            codec="libx264",
                            macro_block_size=1,
                        )
                        writer.append_data(env.render())
                    while True:
                        _, reward, terminated, truncated, _ = env.step(action)
                        steps += 1
                        environment_return += reward
                        # The native reward depends on elapsed time. Use
                        # stationary transition rewards for the (x, y) state
                        # model.
                        learning_reward = float(terminated)
                        learning_return += learning_reward
                        state = tuple(env.agent_pos)
                        # Process the final transition even on time-limit
                        # truncation. Only true termination gives the state
                        # terminal semantics.
                        action = agent.act((state, learning_reward, terminated))
                        if writer is not None:
                            writer.append_data(env.render())
                        if terminated or truncated:
                            agent.reset_episode()
                            break
                finally:
                    if writer is not None:
                        writer.close()
                log.writerow(
                    [
                        episode,
                        steps,
                        terminated,
                        truncated,
                        environment_return,
                        learning_return,
                    ]
                )
                if checkpoint:
                    save_heatmap(
                        agent, episode, heatmaps / f"episode_{episode:04d}.png", width, height
                    )
                    metrics.flush()
                    print(
                        f"Episode {episode}: {steps} steps; goal={terminated}; saved video and heatmap",
                        flush=True,
                    )
    finally:
        env.close()
    return output_dir


def main():
    """Run configurable passive ADP training and report the artifact
    directory."""
    parser = argparse.ArgumentParser(
        description="Train passive ADP with position-only states and uniform cardinal moves."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, help="Output root; runs are stored in WIDTHxHEIGHT subfolders")
    parser.add_argument("--width", type=int, default=4, help="Playable columns, excluding walls")
    parser.add_argument("--height", type=int, default=3, help="Playable rows, excluding walls")
    parser.add_argument("--max-steps", type=int, default=256, help="Step limit per episode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = train(
        episodes=args.episodes, checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir, seed=args.seed,
        width=args.width, height=args.height, max_steps=args.max_steps,
    )
    print(f"Artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
