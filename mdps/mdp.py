from __future__ import annotations
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):
    """
    SimpleEnv class inherits from MiniGridEnv
    """

    def __init__(self, agent_start_pos=(1, 3), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space, width=6, height=5, max_steps=256, **kwargs
        )

    @staticmethod
    def _gen_mission():
        """
        Defines the mission space and returns a string that corresponds to the mission
        """
        return "grand mission"

    def _gen_grid(self, width, height):
        """
        Generates the grid-world and creates the environment
        """
        # creates and empty grid
        self.grid = Grid(width, height)

        # create the walls that surrounds the grid
        self.grid.wall_rect(0, 0, width, height)

        # place the agent in the environment
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # place the goal in the environment
        self.put_obj(Goal(), width - 2, 1)


def main():
    env = SimpleEnv(render_mode="human")

    # enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
