import torch
import random as rn
from omni.isaac.core.articulations import ArticulationView, Articulation
from omni.isaac.core.prims.rigid_prim import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.objects.cylinder import FixedCylinder
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.car import Car, Flag

# Test
import carb
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage

class CarTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.position_old = None
        self.update_config(sim_config)
        self._max_episode_length = 1000  # Adjust episode length as needed

        # Update number of observations and actions based on car's state
        self._num_observations = 9  # Example: position, velocity, orientation, etc.
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
        #self.aimed_position = [rn.randint(5,20),rn.randint(5,20),0]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_acceleration = self._task_cfg["env"]["maxAcceleration"]
        self.left_wheel_joint = self._task_cfg["sim"]["car"]["left_wheel_joint"]
        self.right_wheel_joint = self._task_cfg["sim"]["car"]["right_wheel_joint"]

        self.epsilon = self._task_cfg["env"]["epsilon"]

    def set_up_scene(self, scene) -> None:
        self.get_car()
        # self.get_flags()
        super().set_up_scene(scene)
        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/Car", name="car_view", reset_xform_properties=False
        )
        # self._flags = RigidPrimView(
        #     prim_path="/World/envs/.*/flag", name="flag_view"
        # )

        #self._cylinder = FixedCylinder(position=torch.tensor(self.aimed_position, device=self._device), name="rüdiger", prim_path="/World/Xform/Cylinder")
        #scene.add(self._cylinder)

        scene.add(self._cars)
        # scene.add(self._flags)
        return

    def initialize_views(self, scene):
        print("initialize_views wird aufgerufen")
        super().initialize_views(scene)

        if scene.object_exists("car_view"):
            scene.remove_object("car_view", registry_only=True)
        # if scene.object_exists("flag_view"):
        #     scene.remove_object("flag_view", registry_only=True)
        if scene.object_exists("rüdiger"):
            scene.remove_object("rüdiger")

        self._cars = ArticulationView(
            prim_paths_expr="/World/envs/.*/Car", name="car_view", reset_xform_properties=False
        )
        # self._flags = Rigid(
        #     prim_path="/World/envs/.*/flag", name="flag_view"
        # )
        #self._cylinder = FixedCylinder(position=torch.tensor(self.aimed_position, device=self._device), name="rüdiger", prim_path="/World/Xform/Cylinder")
        #scene.add(self._cylinder)
        scene.add(self._cars)
        # scene.add(self._flags)

    def get_car(self):
        prim_path = self.default_zero_env_path + "/Car"

        car = Car(
            usd_path="assets/car_with_wheels.usd",
            prim_path=prim_path, name="Car",
            translation=torch.tensor(self._initial_position, device=self._device)
        )
        self._sim_config.apply_articulation_settings(
            "Car", get_prim_at_path(car.prim_path), self._sim_config.parse_actor_config("Car")
        )

    def get_flags(self):
        prim_path = self.default_zero_env_path + "/flag"
        flag = Flag(
            usd_path="assets/flag.usd",
            prim_path=prim_path, name="Flag",
            translation=torch.tensor(self._initial_position, device=self._device)
        )

    def get_observations(self) -> dict:
        # TODO: Beobachtungen für Position und Geschwindigkeit
            
        position, orientation = self._cars.get_world_poses(clone=False)
        velocity = self._cars.get_joint_velocities()
        #print(self.aimed_position)
        aimed_position = torch.tensor(self.aimed_position, device=self._device)
        position = position - self._env_pos

        if self.position_old == None:
            distance = torch.zeros(self._num_envs, device=self._device)
        else:
            distance = torch.norm(self.position_old.sub(aimed_position), dim=1)

        self.position_old = position
        self.car_position = torch.norm(position.sub(aimed_position), dim=1)
        self.distance_change = distance - self.car_position
        self.car_velocity = velocity

        # TODO: How to connect these two observations with the buffer?
        self.obs_buf[:, 0] = position[:, 0]
        self.obs_buf[:, 1] = position[:, 1]
        self.obs_buf[:, 2] = position[:, 2]
        self.obs_buf[:, 3] = velocity[:, 0]
        self.obs_buf[:, 4] = velocity[:, 1]
        self.obs_buf[:, 5] = orientation[:, 0]
        self.obs_buf[:, 6] = orientation[:, 1]
        self.obs_buf[:, 7] = orientation[:, 2]
        self.obs_buf[:, 8] = orientation[:, 3]

        observations = {self._cars.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions) -> None:
        if not self.world.is_playing():
            return

        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._cars.count, self._cars.num_dof), dtype=torch.float32, device=self._device)
        
        forces[:, 0] = self._max_acceleration * actions[:, 0]
        forces[:, 1] = self._max_acceleration * actions[:, 1]

        indices = torch.arange(self._cars.count, dtype=torch.int32, device=self._device)
        self._cars.set_joint_velocities(forces, indices=indices)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        
        # Es müssen nur die auf 0 gesetzt werden, die auch fertig sind
        prim_position = self._env_pos[env_ids].add(torch.tensor(self._initial_position, device=self._device))
        prim_velocity = torch.zeros((num_resets, 6), device=self._device)
        orientation = torch.zeros((num_resets, 4), dtype=torch.float32, device=self._device)
        joint_velocity = torch.zeros((num_resets, self._cars.num_dof), device=self._device)
        joint_positions = torch.zeros((num_resets, 2), device=self._device)

        orientation[:, 3] = 1.0

        indices = env_ids.to(dtype=torch.int32)
        
        self._cars.set_joint_velocities(joint_velocity, indices=indices)
        self._cars.set_joint_positions(positions=joint_positions, indices=indices) 
        self._cars.set_world_poses(positions=prim_position, orientations=orientation, indices=indices)
        self._cars.set_velocities(velocities=prim_velocity, indices=indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        #self._cars[0].o = [rn.randint(5,20),rn.randint(5,20),0]
        #print("RESET",env_ids)


    def post_reset(self):
        indices = torch.arange(self._cars.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        
        reward = torch.zeros(512, device=self._device)
        # Belohnung
        reward = torch.where(torch.abs(self.car_position) < self.epsilon, reward + torch.ones_like(reward) * 800.0, reward) #1000
        alpha = 2
        impact = 0.4 #0.4
        reward += impact * (self.distance_change - alpha) ** 3
        # reward = torch.where(abs(self.distance_change) < 4.1237e-02, reward + 2.5,reward)
        # reward = torch.where(abs(self.distance_change) > 1.8023e-02, reward - 10.0,reward)
        # Bestrafung
        reward = torch.where(torch.abs(self.car_position) > self._reset_dist, reward + torch.ones_like(reward) * -200.0, reward)
        self.rew_buf[:] = reward

    def is_done(self) -> None:
        resets = torch.where(torch.abs(self.car_position) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(self.car_position) < self.epsilon, 1, resets)
        self.reset_buf[:] = resets