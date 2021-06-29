# ------------------------------------------------------------------------------
#  Numenta Platform for Intelligent Computing (NuPIC)
#  Copyright (C) 2021, Numenta, Inc.  Unless you have an agreement
#  with Numenta, Inc., for a separate license for this software code, the
#  following terms and conditions apply:
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero Public License version 3 as
#  published by the Free Software Foundation.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#  See the GNU Affero Public License for more details.
#
#  You should have received a copy of the GNU Affero Public License
#  along with this program.  If not, see http://www.gnu.org/licenses.
#
#  http://numenta.org/licenses/
#
# ------------------------------------------------------------------------------

import itertools
from collections import deque
from copy import copy

import gym
from gym import spaces
import numpy as np
from PIL import Image
import pybullet


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class LazyFrames(object):
    def __init__(self, frames):
        """
        from https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974
        143998/baselines/common/atari_wrappers.py#L229

        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

    def count(self):
        frames = self._force()
        return frames.shape[frames.ndim - 1]

    def frame(self, i):
        return self._force()[..., i]


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        See Also
        --------
        baselines.common.atari_wrappers.LazyFrames

        TODO: can we replace this for VecFrameStack from stable_baselines3?
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
            dtype=env.observation_space.dtype
        )

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, acc_info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(
            Image.fromarray(img).resize(size, resample=Image.BILINEAR), dtype=np.uint8
        )
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()


class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset"""
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info["random_state"] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)


class MontezumaInfoWrapper(gym.Wrapper):
    ram_map = {
        "room": dict(
            index=3,
        ),
        "x": dict(
            index=42,
        ),
        "y": dict(
            index=43,
        ),
    }

    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.visited = set()
        self.visited_rooms = set()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        ram_state = unwrap(self.env).ale.getRAM()
        for name, properties in MontezumaInfoWrapper.ram_map.items():
            info[name] = ram_state[properties["index"]]
        pos = (info["x"], info["y"], info["room"])
        self.visited.add(pos)
        self.visited_rooms.add(info["room"])
        if done:
            info["mz_episode"] = dict(
                pos_count=len(self.visited), visited_rooms=copy(self.visited_rooms)
            )
            self.visited.clear()
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()


class MarioXReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.0

    def reset(self):
        ob = self.env.reset()
        self.current_level = [0, 0]
        self.visited_levels = set()
        self.visited_levels.add(tuple(self.current_level))
        self.current_max_x = 0.0
        return ob

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        levellow, levelhigh, xscrollHi, xscrollLo = (
            info["levelLo"],
            info["levelHi"],
            info["xscrollHi"],
            info["xscrollLo"],
        )
        currentx = xscrollHi * 256 + xscrollLo
        new_level = [levellow, levelhigh]
        if new_level != self.current_level:
            self.current_level = new_level
            self.current_max_x = 0.0
            reward = 0.0
            self.visited_levels.add(tuple(self.current_level))
        else:
            if currentx > self.current_max_x:
                delta = currentx - self.current_max_x
                self.current_max_x = currentx
                reward = delta
            else:
                reward = 0.0
        if done:
            info["levels"] = copy(self.visited_levels)
            info["retro_episode"] = dict(levels=copy(self.visited_levels))
        return ob, reward, done, info


class LimitedDiscreteActions(gym.ActionWrapper):
    KNOWN_BUTTONS = {"A", "B"}
    KNOWN_SHOULDERS = {"L", "R"}

    """
    Reproduces the action space from curiosity paper.
    """

    def __init__(self, env, all_buttons, whitelist=KNOWN_BUTTONS | KNOWN_SHOULDERS):
        gym.ActionWrapper.__init__(self, env)

        self._num_buttons = len(all_buttons)
        button_keys = {
            i
            for i in range(len(all_buttons))
            if all_buttons[i] in whitelist & self.KNOWN_BUTTONS
        }
        buttons = [(), *zip(button_keys), *itertools.combinations(button_keys, 2)]
        # shoulder_keys = {
        #     i
        #     for i in range(len(all_buttons))
        #     if all_buttons[i] in whitelist & self.KNOWN_SHOULDERS
        # }
        # shoulders = [(), *zip(shoulder_keys),
        #               *itertools.permutations(shoulder_keys, 2)]
        arrows = [(), (4,), (5,), (6,), (7,)]  # (), up, down, left, right
        acts = []
        acts += arrows
        acts += buttons[1:]
        acts += [a + b for a in arrows[-2:] for b in buttons[1:]]
        self._actions = acts
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class FrameSkip(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n = n

    def step(self, action):
        done = False
        totrew = 0
        for _ in range(self.n):
            ob, rew, done, info = self.env.step(action)
            totrew += rew
            if done:
                break
        return ob, totrew, done, info


def make_mario_env(crop=True, frame_stack=True, clip_rewards=False):
    assert clip_rewards is False
    import retro

    # gym.undo_logger_setup()
    env = retro.make("SuperMarioBros-Nes", "Level1-1")
    buttons = env.buttons
    env = MarioXReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=crop)
    if frame_stack:
        env = FrameStack(env, 4)
    env = LimitedDiscreteActions(env, buttons)
    return env


class OneChannel(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(OneChannel, self).__init__(env)
        assert env.observation_space.dtype == np.uint8
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return obs[:, :, 2:3]


class RetroALEActions(gym.ActionWrapper):
    def __init__(self, env, all_buttons, n_players=1):
        gym.ActionWrapper.__init__(self, env)
        self.n_players = n_players
        self._num_buttons = len(all_buttons)
        bs = [-1, 0, 4, 5, 6, 7]

        def update_actions(old_actions, offset=0):
            actions = []
            for b in old_actions:
                for button in bs:
                    action = []
                    action.extend(b)
                    if button != -1:
                        action.append(button + offset)
                    actions.append(action)
            return actions

        current_actions = [[]]
        for i in range(self.n_players):
            current_actions = update_actions(current_actions, i * self._num_buttons)
        self._actions = current_actions
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):
        mask = np.zeros(self._num_buttons * self.n_players)
        for i in self._actions[a]:
            mask[i] = 1
        return mask


class NoReward(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, rew, done, info = self.env.step(action)
        return ob, 0.0, done, info


def make_multi_pong(frame_stack=True):
    import gym
    import retro

    gym.undo_logger_setup()
    game_env = env = retro.make("Pong-Atari2600", players=2)
    env = RetroALEActions(env, game_env.BUTTONS, n_players=2)
    env = NoReward(env)
    env = FrameSkip(env, 4)
    env = ProcessFrame84(env, crop=False)
    if frame_stack:
        env = FrameStack(env, 4)

    return env


class StickyActionEnv(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionEnv, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class WrapRobot(object):
    """
    Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
    Returns only Box of retina image as observation instead of
    Dictionary of (joint_positions, touch_sensors, retina, depth,
    mask, object_positions, goal, goal_mask, goal_positions).
    This is required for the openAI stable-baselines algorithms
    as they don't support Dict observations.
    Also the action space is reformated into the Box format, omitting
    the render parameter.

    crop_obs: Specifies whether the retina image should be cropped from
             240x320x3 to 180x180x3 (mainly cuts off white space)
    """

    def __init__(self, env, crop_obs=False):
        self._env = env
        self.crop_obs = crop_obs

        self._env.action_space = self._env.action_space["joint_command"]
        if self.crop_obs:
            self._env.observation_space = gym.spaces.Box(
                low=0, high=255.0, shape=(180, 180, 3), dtype="float32"
            )
        else:
            self._env.observation_space = self._env.observation_space["retina"]

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        action = {
            "joint_command": action,
            "render": True,
        }
        observation, reward, done, info = self._env.step_joints(action)
        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation, reward, done, info

    def reset(self):
        observation = self._env.reset()
        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation


class CartesianControlDiscrete(object):
    """
    Wrapper for the REALRobotEnv (based on MJCFBaseBulletEnv).
    Changes action space to discrete using cartesian control.
    Actions: (forward, backward, left, right, up, down, grip, release)
    """

    def __init__(
        self, env, crop_obs=False, repeat=1, touch_reward=False, random_force=False
    ):
        self._env = env
        self.crop_obs = crop_obs
        self.increment_size = 0.5
        self.repeat = repeat
        self.touch_reward = touch_reward
        self.random_force = random_force
        self.act_dict = {
            0: "forward",
            1: "backward",
            2: "left",
            3: "right",
            4: "up",
            5: "down",
            6: "grip",
            7: "release",
        }

        self._env.action_space = gym.spaces.Discrete(8)

        if self.crop_obs:
            self._env.observation_space = gym.spaces.Box(
                low=0, high=255.0, shape=(180, 180, 3), dtype="float32"
            )
        else:
            self._env.observation_space = self._env.observation_space["retina"]

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        cartesian = np.ones(7)
        gripper = np.zeros(2)
        loc = self._env.robot.parts["lbr_iiwa_link_7"].get_position()
        # print('location: '+str(loc))
        grip = np.array(
            [
                self._env.robot.jdict["base_to_finger00_joint"].get_position(),
                self._env.robot.jdict["finger10_to_finger11_joint"].get_position(),
            ]
        )
        # Currently orientation always makes gripper point down -> could be
        # changed later and added into action space. (TODO)
        desiredOrientation = pybullet.getQuaternionFromEuler([0, 3.14, -1.57])
        cartesian[:3] = loc
        cartesian[3:] = desiredOrientation

        if action == 0:  # forward
            cartesian[0] += self.increment_size
        elif action == 1:  # backward
            cartesian[0] -= self.increment_size
        elif action == 2:  # left
            cartesian[1] += self.increment_size
        elif action == 3:  # right
            cartesian[1] -= self.increment_size
        elif action == 4:  # up
            cartesian[2] += self.increment_size
        elif action == 5:  # down
            cartesian[2] -= self.increment_size
        elif action == 6:  # grip
            gripper = grip - self.increment_size
        else:  # release
            gripper = grip + self.increment_size

        if cartesian[0] < -0.15:
            cartesian[0] = -0.15
        if cartesian[0] > 0.15:
            cartesian[0] = 0.15

        if cartesian[1] < -0.3:
            cartesian[0] = -0.3
        if cartesian[1] > 0.3:
            cartesian[0] = 0.3

        action = {
            "cartesian_command": cartesian,
            "gripper_command": gripper,
            "render": True,
        }

        for n in range(self.repeat):
            observation, reward, done, info = self._env.step_cartesian(action)

        if self.random_force:
            if np.random.sample(1)[0] < 0.01:
                loc = self._env.robot.parts["lbr_iiwa_link_7"].get_position()
                cartesian[:3] = loc
                cartesian[2] += self.increment_size
                action["cartesian_command"] = cartesian
                for n in range(self.repeat):
                    # going up:
                    observation, reward, done, info = self._env.step_cartesian(action)
                print("random upward movement applied")
        # print('stepped - '+str(loc))
        # TODO: Make this its own wrapper.
        if self.touch_reward:
            reward = np.sum(observation["touch_sensors"]) / 4
            if reward > 0:
                print("touch! - " + str(reward))
            loc = self._env.robot.parts["lbr_iiwa_link_7"].get_position()
            # reward = reward + (0.5 - loc[2])  # *0.01 # distance to table reward
            # print('reward: '+str(reward))

        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation, reward, done, info

    def reset(self):
        observation = self._env.reset()
        if self.crop_obs:
            observation = observation["retina"][0:180, 70:250, :]
        else:
            observation = observation["retina"]
        return observation
