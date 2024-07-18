from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

class CarView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "AnymalView",
        track_contact_forces=False,
        prepare_contact_sensors=False,
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        self._knees = RigidPrimView(
            prim_paths_expr="/World/envs/.*/anymal/.*_THIGH",
            name="knees_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self._base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/anymal/base",
            name="base_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )