"""Passive temporal-difference (TD(0)) learning on configurable MiniGrids."""

import argparse
import csv
import random
import shutil

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


def fixed_policy(
    state: tuple[int, int],
    *,
    goal: tuple[int, int] = (4, 1),
) -> dict[Move, float]:
    """Move horizontally toward the goal, then vertically; stop at the goal.

    Return a probability of one for the selected action. The policy stays
    unchanged while the TD agent learns its utilities.
    """
    x, y = state
    goal_x, goal_y = goal
    if x < goal_x:
        action = Move.EAST
    elif x > goal_x:
        action = Move.WEST
    elif y > goal_y:
        action = Move.NORTH
    elif y < goal_y:
        action = Move.SOUTH
    else:
        return {}
    return {action: 1.0}


class PassiveTDAgent(Generic[State, Action]):
    """Skeleton for AIMA Chapter 23, Figure 23.4: PASSIVE-TD-LEARNER.

    pi corresponds to the fixed policy; s is the previous state; U stores
    utilities; N corresponds to the textbook's state-frequency table N_s.
    MiniGrid supplies arrival rewards and explicit episode termination.
    """

    def __init__(
        self,
        policy: Callable[[State], dict[Action, float]],
        discount_factor: float = 0.99,
    ) -> None:
        # TODO: Validate discount_factor and store it as self.gamma.
        self.pi = policy
        self.s: State | None = None
        self.U: dict[State, float] = {}
        # N_s in the textbook: each value is an integer count, initially zero.
        self.N: dict[State, int] = {}
        self.gamma = discount_factor

    def passive_td_learning(
        self,
        percept: tuple[State, float],
        *,
        terminal: bool = False,
    ) -> Action | None:
        """Process (s_prime, reward) and return an action under the fixed
        policy."""
        # TODO 1: Unpack percept into s_prime and reward (the textbook's r).
        s_prime, reward = percept

        # TODO 2: If s_prime is new, initialize U[s_prime] to zero.
        if s_prime not in self.U:
            self.U[s_prime] = 0

        # TODO 3: If the previous state s is not None: Initialize its integer
        #   count N[s] to zero if absent, then increment. Choose and document a
        #   step-size schedule alpha(n), evaluated at the updated N[s]. The
        #   textbook leaves the schedule unspecified; for convergence, choose
        #   one whose step sizes sum to infinity and whose squares have a finite
        #   sum. Apply the textbook update to U[s]: U[s] <- U[s] + alpha(N[s]) *
        #   (reward + gamma * U[s_prime] - U[s]). For true termination, use zero
        #   for the successor utility.
        if self.s is not None:
            if self.s not in self.N:
                self.N[self.s] = 0
            self.N[self.s] += 1
            self.alpha = 1.0 / self.N[self.s]
            self.U[self.s] = self.U[self.s] + self.alpha * (
                reward + (self.gamma * self.U[s_prime] - self.U[self.s])
            )

        # TODO 4: MiniGrid episode handling: after learning the final
        # transition, reset the previous state and return None if terminal is
        # True.
        if terminal:
            self.reset_episode()
            return None

        # TODO 5: Remember s_prime as the previous state s.
        self.s = s_prime

        # TODO 6: Return an action under pi(s_prime). Here pi returns an action
        # probability dictionary, so sample one action rather than returning it.
        action_probabilities = self.pi(s_prime)
        self.a = random.choices(
            list(action_probabilities), weights=list(action_probabilities.values())
        )[0]
        return self.a

    def act(self, percept: tuple[State, float, bool]) -> Action | None:
        # TODO: Adapt (state, reward, terminal) to passive_td_learning.
        state, reward, terminal = percept
        return self.passive_td_learning((state, reward), terminal=terminal)

    def reset_episode(self) -> None:
        # TODO: Clear the previous state while retaining utilities and counts.
        self.s = None


class SimpleEnv(MiniGridEnv):
    """
    Reach the goal in a configurable interior using absolute cardinal movement.

    MiniGrid's orientation is used for rendering and executing each move; the
    agent's state consists only of its (x, y) position.
    """

    def __init__(self, width: int = 4, height: int = 3, max_steps: int = 256, **kwargs):
        """
        Configure playable dimensions (excluding walls) and the episode step
        limit.
        """
        if width < 1 or height < 1 or width * height == 1:
            raise ValueError(
                "Playable dimensions must be positive with at least two cells"
            )
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
    agent: PassiveTDAgent,
    episode: int,
    destination: Path,
    width: int = 4,
    height: int = 3,
) -> None:
    """Save one utility per (x, y) cell; gray cells are unobserved."""
    fig, ax = plt.subplots(
        figsize=(max(7, width * 0.8), max(5, height * 0.7)), constrained_layout=True
    )
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
    """Reset this size's outputs, recording episodes 1–10 and later
    checkpoints."""
    if episodes < 1 or checkpoint_every < 1:
        raise ValueError("episodes and checkpoint_every must be positive")
    # Validate the environment before replacing any previous outputs.
    env = SimpleEnv(
        width=width, height=height, max_steps=max_steps, render_mode="rgb_array"
    )
    output_root = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).parent / "outputs" / "passive-td"
    )
    output_dir = output_root / f"{width}x{height}"
    videos = output_dir / "videos"
    heatmaps = output_dir / "heatmaps"
    for directory in (videos, heatmaps):
        if directory.exists():
            shutil.rmtree(directory)
    videos.mkdir(parents=True, exist_ok=True)
    heatmaps.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    agent = PassiveTDAgent(partial(fixed_policy, goal=env.goal_pos))
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
                        if action is None:
                            raise RuntimeError(
                                "TD agent returned no action before episode end. "
                                "Complete act() and passive_td_learning() first."
                            )
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
                        agent,
                        episode,
                        heatmaps / f"episode_{episode:04d}.png",
                        width,
                        height,
                    )
                    metrics.flush()
                    print(
                        f"Episode {episode}: {steps} steps; goal={terminated}; saved video and heatmap",
                        flush=True,
                    )
    finally:
        env.close()
    return output_dir


def main() -> None:
    """Run configurable passive TD(0) training and report the artifact
    directory."""
    parser = argparse.ArgumentParser(
        description="Train passive TD(0) with position-only states and a fixed horizontal-then-vertical policy."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output root; runs are stored in WIDTHxHEIGHT subfolders",
    )
    parser.add_argument(
        "--width", type=int, default=4, help="Playable columns, excluding walls"
    )
    parser.add_argument(
        "--height", type=int, default=3, help="Playable rows, excluding walls"
    )
    parser.add_argument(
        "--max-steps", type=int, default=256, help="Step limit per episode"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output_dir = train(
        episodes=args.episodes,
        checkpoint_every=args.checkpoint_every,
        output_dir=args.output_dir,
        seed=args.seed,
        width=args.width,
        height=args.height,
        max_steps=args.max_steps,
    )
    print(f"Artifacts saved to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
