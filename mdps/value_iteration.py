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


class ValueIterationClass:
    def __init__(self, env, reward=None):
        if reward is None:
            reward = {0: -0.04, 1: 1.0, 2: -1.0, 3: np.nan}

        self.env = env
        self.num_rows = env.height
        self.num_cols = env.width
        self.num_states = len(self.get_states())
        self.num_actions = 3
        self.reward = reward
        # self.reward_function = self.get_reward_function(env)

    def get_reward_function(self):
        reward_table = np.zeros(self.num_states)
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                for action in range(self.num_actions):
                    s = self.get_state_from_pos((row, col, action), self.env)
                    reward_table[s] = self.reward[self.map[row, col, action]]
        return reward_table

    def get_state_from_pos(self, pos):
        indexed_states = self.get_indexed_states(self.env)
        for i in indexed_states.keys():
            if i[0] == pos[0] and i[1] == pos[1] and i[2] == pos[2]:
                return indexed_states[i]

    def get_states(self):
        """
        A state is represented by the location (x, y) and the agent's orientation.
        """
        indexed_states = {}
        directions = [0, 1, 2, 3]

        for i in range(1, self.env.width - 1):
            for j in range(1, self.env.height - 1):
                cell = self.env.grid.get(i, j)
                if cell is not None and cell.type == "wall":
                    indexed_states[(i, j)] = 3
                elif cell is not None and cell.type == "goal":
                    for d in directions:
                        indexed_states[(i, j, d)] = 1
                else:
                    for d in directions:
                        indexed_states[(i, j, d)] = 0

        return indexed_states

    def get_actions(self):
        """
        Actions:
        - 0: left
        - 1: right
        - 2: forward
        """
        available_actions = []
        for i in self.env.actions:
            if i.value >= 0 and i.value <= 2:
                available_actions.append(i.value)
        return available_actions


if __name__ == "__main__":
    env = SimpleEnv(render_mode="human")
    env.reset()
    value_iter = ValueIterationClass(env)
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    print(value_iter.get_states())
    # print(get_actions(env))
    # print(get_transition_model(env, random_rate=0.2))
    # print(value_iter.get_state_from_pos((3, 1, 0), env))
