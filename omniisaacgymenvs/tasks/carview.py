from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from typing import Optional

class CarView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "CarView",
        track_contact_forces=False,
        prepare_contact_sensors=False,
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        