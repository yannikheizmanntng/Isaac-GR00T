from __future__ import annotations

from ._base import UR5CfgBase

# naming convention: <robot>_<state_modality>_<action_modality>_<number_of_cameras>_Cfg
# e.g., UR5_Abs_Abs_2_Cfg for UR5 robot with absolute joint state and absolute joint action using 2 cameras

class UR5_Abs_Delta_4_Cfg(UR5CfgBase):
    video_keys = ["video.camera_front", "video.camera_right", "video.camera_left", "video.camera_wrist"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.delta_gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  


class UR5_Abs_Delta_2_Cfg(UR5CfgBase):
    video_keys = ["video.camera_front", "video.camera_wrist"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.delta_gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  