def import_tasks():
    from omniisaacgymenvs.tasks.allegro_hand import AllegroHandTask
    from omniisaacgymenvs.tasks.ant import AntLocomotionTask
    from omniisaacgymenvs.tasks.anymal import AnymalTask
    from omniisaacgymenvs.tasks.anymal_terrain import AnymalTerrainTask
    from omniisaacgymenvs.tasks.ball_balance import BallBalanceTask
    from omniisaacgymenvs.tasks.cartpole import CartpoleTask
    from omniisaacgymenvs.tasks.cartpole_camera import CartpoleCameraTask
    from omniisaacgymenvs.tasks.crazyflie import CrazyflieTask
    from omniisaacgymenvs.tasks.factory.factory_task_nut_bolt_pick import FactoryTaskNutBoltPick
    from omniisaacgymenvs.tasks.factory.factory_task_nut_bolt_place import FactoryTaskNutBoltPlace
    from omniisaacgymenvs.tasks.factory.factory_task_nut_bolt_screw import FactoryTaskNutBoltScrew
    from omniisaacgymenvs.tasks.franka_cabinet import FrankaCabinetTask
    from omniisaacgymenvs.tasks.franka_deformable import FrankaDeformableTask
    from omniisaacgymenvs.tasks.humanoid import HumanoidLocomotionTask
    from omniisaacgymenvs.tasks.ingenuity import IngenuityTask
    from omniisaacgymenvs.tasks.quadcopter import QuadcopterTask
    from omniisaacgymenvs.tasks.shadow_hand import ShadowHandTask

    from omniisaacgymenvs.tasks.warp.ant import AntLocomotionTask as AntLocomotionTaskWarp
    from omniisaacgymenvs.tasks.warp.cartpole import CartpoleTask as CartpoleTaskWarp
    from omniisaacgymenvs.tasks.warp.humanoid import HumanoidLocomotionTask as HumanoidLocomotionTaskWarp

    # TODO: maybe adjust this path later
    from omniisaacgymenvs.tasks.self_driving_car import SelfDrivingCarTask

    # Mappings from strings to environments
    task_map = {
        "AllegroHand": AllegroHandTask,
        "Ant": AntLocomotionTask,
        "Anymal": AnymalTask,
        "AnymalTerrain": AnymalTerrainTask,
        "BallBalance": BallBalanceTask,
        "Cartpole": CartpoleTask,
        "CartpoleCamera": CartpoleCameraTask,
        "FactoryTaskNutBoltPick": FactoryTaskNutBoltPick,
        "FactoryTaskNutBoltPlace": FactoryTaskNutBoltPlace,
        "FactoryTaskNutBoltScrew": FactoryTaskNutBoltScrew,
        "FrankaCabinet": FrankaCabinetTask,
        "FrankaDeformable": FrankaDeformableTask,
        "Humanoid": HumanoidLocomotionTask,
        "Ingenuity": IngenuityTask,
        "Quadcopter": QuadcopterTask,
        "Crazyflie": CrazyflieTask,
        "ShadowHand": ShadowHandTask,
        "ShadowHandOpenAI_FF": ShadowHandTask,
        "ShadowHandOpenAI_LSTM": ShadowHandTask,
        "SelfDrivingCar": SelfDrivingCarTask,
    }

    task_map_warp = {
        "Cartpole": CartpoleTaskWarp,
        "Ant":AntLocomotionTaskWarp,
        "Humanoid": HumanoidLocomotionTaskWarp
    }

    return task_map, task_map_warp


def initialize_task(config, env, init_sim=True):
    from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig

    sim_config = SimConfig(config)
    task_map, task_map_warp = import_tasks()

    cfg = sim_config.config
    if cfg["warp"]:
        task_map = task_map_warp

    task = task_map[cfg["task_name"]](
        name=cfg["task_name"], sim_config=sim_config, env=env
    )

    backend = "warp" if cfg["warp"] else "torch"

    rendering_dt = sim_config.get_physics_params()["rendering_dt"]

    env.set_task(
        task=task,
        sim_params=sim_config.get_physics_params(),
        backend=backend,
        init_sim=init_sim,
        rendering_dt=rendering_dt,
    )

    return task