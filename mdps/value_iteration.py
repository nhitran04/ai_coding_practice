import numpy as np
import matplotlib.pyplot as plt

from minigrid.core.grid import Grid
from minigrid.core.world_object import Wall, Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.manual_control import ManualControl
from matplotlib import patches


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
        self.state_rewards = self.get_states()
        self.states = list(self.state_rewards.keys())
        self.state_to_index = {state: index for index, state in enumerate(self.states)}
        self.num_states = len(self.get_states())
        self.num_actions = 3
        self.num_directions = 4
        self.reward = reward
        self.reward_function = self.get_reward_function()
        self.transition_model = self.get_transition_model()

    def get_reward_function(self):
        """
        Assigns a reward to each state and returns a reward table.
        """
        reward_table = np.zeros(self.num_states)

        for state, state_index in self.state_to_index.items():
            reward_type = self.state_rewards[state]
            reward_table[state_index] = self.reward[reward_type]

        return reward_table

    def get_states(self):
        """
        A state is represented by the location (x, y) and the agent's orientation.
        """
        state_rewards = {}
        directions = [0, 1, 2, 3]

        for i in range(1, self.env.width - 1):
            for j in range(1, self.env.height - 1):
                cell = self.env.grid.get(i, j)
                if cell is not None and cell.type == "wall":
                    state_rewards[(i, j, None)] = 3
                elif cell is not None and cell.type == "goal":
                    for d in directions:
                        state_rewards[(i, j, d)] = 1
                else:
                    for d in directions:
                        state_rewards[(i, j, d)] = 0

        return state_rewards

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

    def get_transition_model(self, random_rate=0.2):
        transition_model = np.zeros(
            (self.num_states, self.num_actions, self.num_states)
        )

        for k, v in self.get_states().items():
            neighbor_s = np.zeros(self.num_actions)

            if v == 0:
                for action in range(self.num_actions):
                    new_row, new_col, new_dir = self.get_next_state(k, action)
                    s_prime = self.get_state_from_pos((new_row, new_col, new_dir))
                    neighbor_s[action] = s_prime

                for action in range(self.num_actions):
                    main = int(neighbor_s[action])
                    transition_model[v, action, main] += 1 - random_rate

                    if action != 0:
                        left = int(neighbor_s[action - 1])
                        transition_model[v, action, left % self.num_actions] += (
                            random_rate / 2.0
                        )

                    if action != self.num_actions - 1:
                        right = int(neighbor_s[action + 1])
                        transition_model[v, action, right % self.num_actions] += (
                            random_rate / 2.0
                        )

        return transition_model

    def move_forward(self, x, y, direction):
        """
        Moves the agent forward depending on its orientation.
        """
        new_x, new_y = x, y

        if direction == 0:  # agent faces fight
            new_x += 1
        elif direction == 1:  # agent faces down
            new_y += 1
        elif direction == 2:  # agent faces left
            new_x -= 1
        elif direction == 3:  # agent faces up
            new_y -= 1

        positions = {(state[0], state[1]) for state in self.states}

        if (new_x, new_y) not in positions:
            return x, y

        cell = self.env.grid.get(new_x, new_y)
        if cell is not None and cell.type == "wall":
            return x, y

        return new_x, new_y

    def get_next_state(self, state, action):
        row, col, direction = state

        if direction is not None:
            if action == 0:  # turn left
                return (row, col, (direction - 1) % 4)
            elif action == 1:  # turn right
                return (row, col, (direction + 1) % 4)
            elif action == 2:  # move forward
                new_row, new_col = self.move_forward(row, col, direction)
                return (new_row, new_col, direction)

        return state

    def get_state_from_pos(self, pos):
        """
        Returns the state index based on position.
        """
        return self.state_to_index.get(pos)


if __name__ == "__main__":
    env = SimpleEnv()
    env.reset()
    value_iter = ValueIterationClass(env)
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    # print(value_iter.get_states())
    # print(value_iter.get_state_from_pos((2, 2, None)))
    print(value_iter.get_reward_function())
    # print(value_iter.get_transition_model())
