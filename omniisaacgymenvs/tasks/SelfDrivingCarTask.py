class SelfDrivingCarTask(RLTask):
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
        self._car_positions = torch.tensor([0.0, 0.0, 0.0])  # Adjust initial positions

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_steering_angle = self._task_cfg["env"]["maxSteeringAngle"]
        self._max_acceleration = self._task_cfg["env"]["maxAcceleration"]

    def set_up_scene(self, scene) -> None:
        self.get_car()
        super().set_up_scene(scene)
        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/Car", name="car_view", reset_xform_properties=False
        )
        scene.add(self._cars)
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)
        if scene.object_exists("car_view"):
            scene.remove_object("car_view", registry_only=True)
        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/Car", name="car_view", reset_xform_properties=False
        )
        scene.add(self._cars)

    def get_car(self):
        car = Car(
            prim_path=self.default_zero_env_path + "/Car", name="Car", translation=self._car_positions
        )
        self._sim_config.apply_articulation_settings(
            "Car", get_prim_at_path(car.prim_path), self._sim_config.parse_actor_config("Car")
        )

    def get_observations(self) -> dict:
        pos = self._cars.get_joint_positions(clone=False)
        vel = self._cars.get_joint_velocities(clone=False)

        self.car_pos = pos[:, self._car_pos_idx]
        self.car_vel = vel[:, self._car_vel_idx]
        self.car_orientation = pos[:, self._car_orientation_idx]
        self.car_steering_angle = pos[:, self._car_steering_idx]

        self.obs_buf[:, 0] = self.car_pos
        self.obs_buf[:, 1] = self.car_vel
        self.obs_buf[:, 2] = self.car_orientation
        self.obs_buf[:, 3] = self.car_steering_angle

        observations = {self._cars.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        controls = torch.zeros((self._cars.count, self._cars.num_dof), dtype=torch.float32, device=self._device)
        controls[:, self._steering_dof_idx] = self._max_steering_angle * actions[:, 0]
        controls[:, self._acceleration_dof_idx] = self._max_acceleration * actions[:, 1]

        indices = torch.arange(self._cars.count, dtype=torch.int32, device=self._device)
        self._cars.set_joint_controls(controls, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        pos = torch.zeros((num_resets, self._cars.num_dof), device=self._device)
        pos[:, self._car_pos_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        pos[:, self._car_orientation_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        vel = torch.zeros((num_resets, self._cars.num_dof), device=self._device)
        vel[:, self._car_vel_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        indices = env_ids.to(dtype=torch.int32)
        self._cars.set_joint_positions(pos, indices=indices)
        self._cars.set_joint_velocities(vel, indices=indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        self._car_pos_idx = self._cars.get_dof_index("carPosition")
        self._car_vel_idx = self._cars.get_dof_index("carVelocity")
        self._car_orientation_idx = self._cars.get_dof_index("carOrientation")
        self._car_steering_idx = self._cars.get_dof_index("steeringJoint")
        self._acceleration_dof_idx = self._cars.get_dof_index("accelerationJoint")

        indices = torch.arange(self._cars.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        # Example reward function for a self-driving car
        reward = 1.0 - self.car_vel * self.car_vel - 0.01 * torch.abs(self.car_steering_angle)
        reward = torch.where(torch.abs(self.car_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.where(torch.abs(self.car_pos) > self._reset_dist, 1, 0)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
        self.reset_buf[:] = resets