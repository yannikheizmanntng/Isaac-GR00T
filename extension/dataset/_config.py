from __future__ import annotations

from __future__ import annotations

from gr00t.data.dataset import ModalityConfig
from gr00t.data.transform.base import ComposedModalityTransform, ModalityTransform
from gr00t.data.transform.concat import ConcatTransform
from gr00t.data.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from gr00t.data.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)
from gr00t.experiment.data_config import BaseDataConfig
from gr00t.model.transforms import GR00TTransform

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
    video_keys = ["video.camera_wrist", "video.camera_front"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.delta_gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  

class TNGUR5_AbsoluteJointState_DeltaJointAction_2Cams(UR5CfgBase):
    video_keys = ["video.camera_wrist", "video.camera_global_main"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.delta_gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  

class UR5_Abs_Delta_Abs_Grp_4_Cfg(UR5CfgBase):
    video_keys = ["video.camera_front", "video.camera_wrist"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  


class UR5_Abs_Delta_Bin_Grp_4_Cfg(UR5CfgBase):
    video_keys = ["video.camera_front", "video.camera_right", "video.camera_left", "video.camera_wrist"]
    state_keys = ["state.robot_arm", "state.gripper"]
    action_keys = ["action.delta_robot_arm", "action.delta_gripper"]
    language_keys = ["annotation.human.task_description"]
    state_slices = [(0,6), (6,7)]  
    
    def transform(self) -> ModalityTransform:
        transforms = [
            # video transforms
            VideoToTensor(apply_to=self.video_keys),
            VideoCrop(apply_to=self.video_keys, scale=0.95),
            VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(
                apply_to=self.video_keys,
                brightness=0.3,
                contrast=0.4,
                saturation=0.5,
                hue=0.08,
            ),
            VideoToNumpy(apply_to=self.video_keys),
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={key: "min_max" for key in self.state_keys},
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    key: ("binary" if key == "action.delta_gripper" else "min_max")
                    for key in self.action_keys
                },
            ),
            # concat transforms
            ConcatTransform(
                video_concat_order=self.video_keys,
                state_concat_order=self.state_keys,
                action_concat_order=self.action_keys,
            ),
            # model-specific transform
            GR00TTransform(
                state_horizon=len(self.observation_indices),
                action_horizon=len(self.action_indices),
                max_state_dim=64,
                max_action_dim=32,
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)