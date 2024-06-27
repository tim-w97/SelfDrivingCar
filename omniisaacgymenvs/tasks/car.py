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
        self._initial_position = self._task_cfg["env"]["initial_position"]
        self.aimed_position = self._task_cfg["env"]["aimed_position"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_acceleration = self._task_cfg["env"]["maxAcceleration"]
        self.left_wheel_joint = self._task_cfg["sim"]["car"]["left_wheel_joint"]
        self.right_wheel_joint = self._task_cfg["sim"]["car"]["right_wheel_joint"]

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
        prim_path = self.default_zero_env_path + "/Car"

        car = Car(
            usd_path="assets/car_with_wheels.usd", #/home/HOF-UNIVERSITY.DE/ndemel/SelfDrivingCar/assets/car.usd
            prim_path=prim_path, name="Car",
            translation=torch.tensor(self._initial_position, device=self._device)
        )
        self._sim_config.apply_articulation_settings(
            "Car", get_prim_at_path(car.prim_path), self._sim_config.parse_actor_config("Car")
        )

    def get_observations(self) -> dict:
        # TODO: Beobachtungen für Position und Geschwindigkeit
        position,_= self._cars.get_world_poses(clone=False)
        velocity = self._cars.get_joint_velocities()

        aimed_position = torch.tensor(self.aimed_position, device=self._device)

        position = position.sub(aimed_position)
        self.car_position = torch.norm(position)
        self.car_velocity = velocity
        # TODO: How to connect these two observations with the buffer?

        observations = {self._cars.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # TODO: Apply the action
        actions = torch.tensor(actions)
        # print("Actions: ", len(actions), actions.shape)
        indices = torch.arange(self._cars.count, dtype=torch.int32, device=self._device)

        forces = torch.zeros((self._cars.count, self._cars.num_dof), dtype=torch.float32, device=self._device)
        forces[:, 0] = self._max_acceleration * actions[:,0]
        forces[:, 1] = self._max_acceleration * actions[:,1]

        self._cars.set_joint_efforts(forces, indices=indices)

        # Ist das wirklich notwenig?
        controls = torch.zeros((self._cars.count, self._cars.num_dof), dtype=torch.float32, device=self._device)
        self._cars.set_joint_positions(controls, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        velocity = torch.zeros((num_resets, self._cars.num_dof), device=self._device)
        
        # TODO: Müssen alle Autos zurückgesetzt werden oder nur ein paar

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):
        # Reset Poses
        positions, orientations = self._cars.get_world_poses() # poses returns tuple array
        print(type(positions))
        self._cars.set_world_poses(positions=positions, orientations=orientations)

        # Reset Joint Velocities
        velocity = self._cars.get_joint_velocities()
        velocity[:] = torch.tensor([0.0, 0.0])
        self._cars.set_joint_velocities(velocities=velocity)

        indices = torch.arange(self._cars.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        
        # TODO: Belohnung, wenn das Model eine hohe Geschwindigkeit hat
        reward = 1.0 * self.car_velocity * 0.01
        print("reward", type(self.car_velocity))
        # TODO: Bestrafung, wenn das Model sich immer weiter vom Punkt entfernt
        reward = torch.where(torch.abs(self.car_position) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)

        self.rew_buf[:] = torch.rand(512, device=self._device)

    def is_done(self) -> None:
        resets = torch.where(torch.abs(self.car_position) > self._reset_dist, 1, 0)
        self.reset_buf[:] = resets