"""A MiniGrid environment with a 4-column by 3-row playable area."""

from dataclasses import dataclass, field
import random
from typing import Callable, Generic, Hashable, TypeVar

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


State = TypeVar("State", bound=Hashable)
Action = TypeVar("Action", bound=Hashable)


def random_policy(state: tuple[int, int, int]) -> dict[Actions, float]:
    """Return a uniform distribution over movement actions; no actions at the
    goal."""
    x, y, _ = state
    if (x, y) == (4, 1):
        return {}
    return {Actions.left: 1 / 3, Actions.right: 1 / 3, Actions.forward: 1 / 3}


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

                for action, action_probability in policy(state).items():
                    for successor, probability in mdp.P.get(
                        (state, action), {}
                    ).items():
                        V[state] = sum(
                            action_probability
                            * sum(
                                probability
                                * (
                                    mdp.R[(state, action, successor)]
                                    * mdp.gamma
                                    * V[successor]
                                )
                            )
                        )

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
