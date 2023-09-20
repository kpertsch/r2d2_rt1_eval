import tensorflow as tf
import numpy as np
from tf_agents.policies import py_tf_eager_policy
import tf_agents
from tf_agents.trajectories import time_step as ts

from r2d2.controllers.oculus_controller import VRPolicy
from r2d2.robot_env import RobotEnv
from r2d2.user_interface.data_collector import DataCollecter
from r2d2.user_interface.gui import RobotGUI


CHECKPOINT_PATH = 'PATHTOCKPT/001008560/'
GOAL_PATH = 'PATHTOGOAL'


def resize(image):
    image = tf.image.resize_with_pad(image, target_width=320, target_height=256)
    image = tf.cast(image, tf.uint8)
    return image


def load_goals():
    return (resize(tf.convert_to_tensor(np.random.rand(128, 128, 3))),
            resize(tf.convert_to_tensor(np.random.rand(128, 128, 3))))


class RT1Policy:
    def __init__(self, checkpoint_path, goal_images):
        """goal_images is a tuple of two goal images."""
        self._policy = py_tf_eager_policy.SavedModelPyTFEagerPolicy(
            model_path=checkpoint_path,
            load_specs_from_pbtxt=True,
            use_tf_function=True)
        self._run_dummy_inference()
        self._goal_images = goal_images
        self._policy_state = self._policy.get_initial_state(batch_size=1)

    def _run_dummy_inference(self):
        observation = tf_agents.specs.zero_spec_nest(
            tf_agents.specs.from_spec(self._policy.time_step_spec.observation))
        tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))
        policy_state = self._policy.get_initial_state(batch_size=1)
        action = self._policy.action(tfa_time_step, policy_state)

    def forward(self, observation):
        # construct observation
        observation['goal_image'] = self._goal_images[0]
        observation['goal_image1'] = self._goal_images[1]

        observation['image'] = resize(tf.convert_to_tensor(
            observation['image']['16291792_left'][:, :, :3].copy()[..., ::-1]))
        observation['image1'] = resize(tf.convert_to_tensor(
            observation['image']['16291792_right'][:, :, :3].copy()[..., ::-1]))

        tfa_time_step = ts.transition(observation, reward=np.zeros((), dtype=np.float32))

        policy_step = self._policy.action(tfa_time_step, self._policy_state)
        action = policy_step.action.numpy()
        self._policy_state = policy_step.state
        return np.concatenate([action['world_vector'],
                               action['rotation_delta'],
                               action['gripper_closedness_action']])


policy = RT1Policy(CHECKPOINT_PATH, load_goals())

env = RobotEnv()
controller = VRPolicy()

data_col = DataCollecter(env = env, controller=controller, policy=policy, save_data=False)
RobotGUI(robot=data_col)