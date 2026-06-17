import numpy as np

from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.manual_control import ManualControl


class SimpleEnv(MiniGridEnv):
    def __init__(self, agent_start_pos=(1, 3), agent_start_dir=0, **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space, width=6, height=5, max_steps=256, **kwargs
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        self.grid.set(2, 2, Wall())

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.put_obj(Goal(), width - 2, height - 2)

    @staticmethod
    def _gen_mission():
        return "reach the goal"


def get_states(env):
    states = []
    directions = ["up", "down", "right", "left"]
    for i in range(1, env.width - 1):
        for j in range(1, env.height - 1):
            cell = env.grid.get(i, j)

            if cell is not None and cell.type == "wall":
                continue

            for d in directions:
                states.append((i, j, d))
    return states


def get_actions(env):
    available_actions = []
    for i in env.actions:
        available_actions.append({i.name})
    return available_actions


def value_iteration():
    pass


if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    print(get_states(env))
    # print(get_actions(env))
