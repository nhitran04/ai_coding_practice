"""A MiniGrid environment with a 4-column by 3-row playable area."""

from typing import Callable, Generic, TypeVar

from minigrid.core.actions import Actions
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


State = TypeVar("State")
Action = TypeVar("Action")


def fixed_policy(state: tuple[int, int, int]) -> Actions:
    """Map (x, y, direction) to an action toward the goal at (4, 1)."""
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


class PassiveADPAgent(Generic[State, Action]):
    """Follow a fixed policy while learning the model and state utilities."""

    def __init__(
        self,
        policy: Callable[[State], Action],
        discount_factor: float = 0.99,
    ) -> None:
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1")

        self.policy = policy
        self.discount_factor = discount_factor

        # Learned state rewards and utilities.
        self.rewards: dict[State, float] = {}
        self.reward_counts: dict[State, int] = {}
        self.utilities: dict[State, float] = {}
        self.terminal_states: set[State] = set()

        # Estimate P(next_state | state, action) from observed transitions.
        self.state_action_counts: dict[tuple[State, Action], int] = {}
        self.transition_counts: dict[tuple[State, Action, State], int] = {}
        self.transition_model: dict[tuple[State, Action], dict[State, float]] = {}

        # The first percept of an episode has no preceding transition.
        self.previous_state: State | None = None
        self.previous_action: Action | None = None

    def act(
        self,
        percept: tuple[State, float, bool],  # state, reward, terminal
    ) -> Action | None: ...

    def update_reward(self, state: State, reward: float) -> None: ...

    def update_transition_model(
        self,
        previous_state: State,
        previous_action: Action,
        current_state: State,
    ) -> None: ...

    def evaluate_policy(self) -> None: ...

    def reset_episode(self) -> None: ...


class SimpleEnv(MiniGridEnv):
    """Reach the goal in a 4x3 interior surrounded by boundary walls."""

    def __init__(self, **kwargs):
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
        return "reach the goal"

    def _gen_grid(self, width, height):
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
