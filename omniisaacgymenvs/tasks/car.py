import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.car import Car

class CarTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)
        self._max_episode_length = 1000  # Adjust episode length as needed

        # Update number of observations and actions based on car's state
        self._num_observations = 8  # Example: position, velocity, orientation, etc.
        self._num_actions = 2  # Example: steering angle, acceleration

        RLTask.__init__(self, name, env)
        return

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._car_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_steering_angle = self._task_cfg["env"]["maxSteeringAngle"]
        self._max_acceleration = self._task_cfg["env"]["maxAcceleration"]

    def set_up_scene(self, scene) -> None:
        self.get_car()
        super().set_up_scene(scene)
        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/car", name="car_view", reset_xform_properties=False
        )

        print("ArticulationView:", self._cars)

        scene.add(self._cars)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("car_view"):
            scene.remove_object("car_view", registry_only=True)
        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/car", name="car_view", reset_xform_properties=False
        )
        scene.add(self._cars)

    def get_car(self):
        prim_path = self.default_zero_env_path + "/car"
        print("get_car() prim_path", prim_path)

        car = Car(
            usd_path="/opt/localdata/VirtualBoxVMs/ov/tim/SelfDrivingCar/assets/car.usd",
            prim_path=prim_path, name="Car",
            translation=self._car_positions
        )
        self._sim_config.apply_articulation_settings(
            "Car", get_prim_at_path(car.prim_path), self._sim_config.parse_actor_config("Car")
        )

    def get_observations(self) -> dict:
        pass

    def pre_physics_step(self, actions) -> None:
        pass
    
    def reset_idx(self, env_ids):
        pass

    def post_reset(self):
        pass

    def calculate_metrics(self) -> None:
        pass

    def is_done(self) -> None:
        pass