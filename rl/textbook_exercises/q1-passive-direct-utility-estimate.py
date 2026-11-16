"""MiniGrid setup for passive direct utility estimation."""

from enum import IntEnum
from pathlib import Path
from typing import Callable, Generic, Hashable, TypeVar

from gymnasium.spaces import Discrete
from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv


State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


class Move(IntEnum):
    """Absolute movement directions, using MiniGrid's direction numbering."""

    EAST = 0
    SOUTH = 1
    WEST = 2
    NORTH = 3


def fixed_policy(
    state: tuple[int, int],
    *,
    goal: tuple[int, int] = (4, 1),
) -> dict[Move, float]:
    """Move horizontally toward the goal, then vertically; stop at the goal.

    Return action probabilities using the same interface as random_policy. This
    deterministic policy stays unchanged during utility estimation.
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


class PassiveDirectUtilityAgent(Generic[State, Action]):
    """Estimate fixed-policy utilities from observed episode returns."""

    def __init__(
        self,
        policy: Callable[[State], dict[Action, float]],
        discount_factor: float = 0.99,
    ) -> None:
        """Store the fixed policy and initialize persistent and episode data."""
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1 inclusive")

        self.pi = policy
        self.gamma = discount_factor
        self.U: dict[State, float] = {}  # Mean observed return for each state.
        self.N: dict[State, int] = {}  # Number of returns averaged per state.

        # Each reward belongs to the transition arriving at its paired state.
        self.episode_history: list[tuple[State, float]] = []

    def passive_direct_utility_learning(
        self,
        percept: tuple[State, float],
        *,
        terminal: bool = False,
    ) -> Action | None:
        # TODO: Record the state and reward (reward belongs to the prior
        # transition). TODO: On termination, learn from the completed episode
        # and return None. TODO: Otherwise select an action using the fixed
        # policy.
        pass

    def act(self, percept: tuple[State, float, bool]) -> Action | None:
        # TODO: Adapt (state, reward, terminal) to the learning method.
        state, reward, terminal = percept
        return self.passive_direct_utility_learning((state, reward), terminal=terminal)

    def update_utilities(self) -> None:
        # TODO: Estimate utilities from discounted returns in the episode
        # history. TODO: Choose and document first-visit or every-visit
        # estimation.
        pass

    def reset_episode(self) -> None:
        # TODO: Clear episode history while retaining learned utilities and
        # counts.
        pass


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
    agent: PassiveDirectUtilityAgent,
    episode: int,
    destination: Path,
    width: int = 4,
    height: int = 3,
) -> None:
    # TODO: Plot position utilities, mark unseen cells, and save the figure.
    pass


def train(
    episodes: int = 100,
    checkpoint_every: int = 10,
    output_dir: Path | None = None,
    seed: int = 42,
    width: int = 4,
    height: int = 3,
    max_steps: int = 256,
) -> Path:
    # TODO: Create the environment and agent with fixed_policy bound to
    # env.goal_pos. TODO: Run episodes, passing position, reward, and terminal
    # status to the agent. TODO: Choose consistent learning rewards and document
    # how truncation is handled. TODO: Record metrics, videos, and utility
    # heatmaps at checkpoints. TODO: Close resources and return the output
    # directory.
    pass


def main() -> None:
    # TODO: Parse command-line options and invoke train with those options.
    pass
