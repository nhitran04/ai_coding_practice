"""A MiniGrid environment with a 4-column by 3-row playable area."""

from dataclasses import dataclass, field
from typing import Callable, Generic, Hashable, TypeVar

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


def fixed_policy(state: tuple[int, int, int]) -> Actions:
    """
    Map (x, y, direction) to an action toward the goal at (4, 1).
    """
    x, y, direction = state
    if (x, y) == (4, 1):
        return Actions.done

    # MiniGrid directions: 0 = right, 1 = down, 2 = left, 3 = up.
    desired_direction = 0 if x < 4 else 3
    if direction == desired_direction:
        return Actions.forward
    if (desired_direction - direction) % 4 == 1:
        return Actions.right
    return Actions.left


@dataclass
class LearnedMDP(Generic[State, Action]):
    """The textbook's learned model P, transition rewards R, actions A, and
    γ."""

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
        policy: Callable[[State], Action],
        discount_factor: float = 0.99,
    ) -> None:
        """
        Store the fixed policy and initialize the learned MDP and persistent
        tables.
        """
        # A discount below one guarantees a unique discounted Bellman solution.
        if not 0.0 <= discount_factor < 1.0:
            raise ValueError("discount_factor must be at least 0 and less than 1")

        self.pi = policy
        self.mdp: LearnedMDP[State, Action] = LearnedMDP(discount_factor)
        self.U: dict[State, float] = {}
        self.N: dict[tuple[State, Action], dict[State, int]] = {}
        self.s: State | None = None
        self.a: Action | None = None
        self.terminal_states: set[State] = set()

    def passive_adp_learning(
        self,
        percept: tuple[State, float],
        *,
        terminal: bool = False,
    ) -> Action | None:
        """Learn from (state, reward), evaluate the policy, and return its next
        action.

        The reward belongs to the previous transition. A terminal percept ends
        the episode after that transition is learned and returns no action.
        """
        ...

    def act(self, percept: tuple[State, float, bool]) -> Action | None:
        """
        Adapt a (state, reward, terminal) percept to passive_adp_learning.
        """
        ...

    @staticmethod
    def normalize(outcome_counts: dict[State, int]) -> dict[State, float]:
        """
        Convert outcome counts into probabilities summing to one.
        """
        ...

    @staticmethod
    def policy_evaluation(
        policy: Callable[[State], Action],
        utilities: dict[State, float],
        mdp: LearnedMDP[State, Action],
    ) -> dict[State, float]:
        """
        Return utilities satisfying the fixed-policy Bellman equations.
        """
        ...

    def reset_episode(self) -> None:
        """
        Clear the previous state and action while retaining the learned model
        and utilities.
        """
        ...


class SimpleEnv(MiniGridEnv):
    """
    Reach the goal in a 4x3 interior surrounded by boundary walls.
    """

    def __init__(self, **kwargs):
        """
        Configure the mission, grid dimensions, and episode step limit.
        """
        mission_space = MissionSpace(mission_func=self._gen_mission)
        # MiniGrid dimensions include the surrounding walls.
        super().__init__(
            mission_space=mission_space,
            width=6,
            height=5,
            max_steps=256,
            **kwargs,
        )

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

        self.agent_pos = (1, 3)
        self.agent_dir = 0  # Face right.
        self.put_obj(Goal(), width - 2, 1)
        self.mission = self._gen_mission()


def main():
    env = SimpleEnv(render_mode="human")
    try:
        ManualControl(env, seed=42).start()
    finally:
        env.close()


if __name__ == "__main__":
    main()
