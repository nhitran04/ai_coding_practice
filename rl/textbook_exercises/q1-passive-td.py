"""Passive temporal-difference (TD(0)) learning on configurable MiniGrids."""

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
    """Learn the utilities of a fixed policy with one-step TD(0) updates."""

    def __init__(
        self,
        policy: Callable[[State], dict[Action, float]],
        discount_factor: float = 0.99,
    ) -> None:
        # TODO: Validate and store the discount factor and fixed policy.
        # TODO: Initialize utility estimates U and state update counts N.
        # TODO: Initialize the previous state s to None.
        pass

    def passive_td_learning(
        self,
        percept: tuple[State, float],
        *,
        terminal: bool = False,
    ) -> Action | None:
        # TODO: Unpack the current state and reward from the prior transition.
        # TODO: Initialize the utility of a newly observed state.
        # TODO: If a previous state exists, increment its update count and
        # update its utility using the reward and successor utility (TD(0)).
        # TODO: Use zero successor value on true termination; then reset
        # episode state and return None after processing the final transition.
        # TODO: Otherwise remember this state and sample an action from pi.
        pass

    def act(self, percept: tuple[State, float, bool]) -> Action | None:
        # TODO: Adapt (state, reward, terminal) to passive_td_learning.
        pass

    @staticmethod
    def learning_rate(update_count: int) -> float:
        # TODO: Return a decreasing step size for a positive state update count.
        pass

    def reset_episode(self) -> None:
        # TODO: Clear the previous state while retaining utilities and counts.
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
    agent: PassiveTDAgent,
    episode: int,
    destination: Path,
    width: int = 4,
    height: int = 3,
) -> None:
    # TODO: Plot utilities by position, mark unseen cells and the goal, and save.
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
    # TODO: Validate options, create SimpleEnv, and seed action sampling.
    # TODO: Create PassiveTDAgent with fixed_policy bound to env.goal_pos.
    # TODO: Default outputs to outputs/passive-td/WIDTHxHEIGHT.
    # TODO: Reset the environment and agent episode state for each episode.
    # TODO: Pass position, reward, and termination status to agent.act each step.
    # TODO: Use stationary learning rewards, as in the passive ADP script.
    # TODO: On truncation, learn the last transition with its successor value,
    # then reset episode state without marking that successor as terminal.
    # TODO: Save videos and heatmaps for episodes 1-10 and later checkpoints.
    # TODO: Write episode metrics to CSV, close resources, and return the path.
    pass


def main() -> None:
    # TODO: Parse the same command-line options as the passive ADP script.
    # TODO: Call train and print the resulting artifact directory.
    pass
