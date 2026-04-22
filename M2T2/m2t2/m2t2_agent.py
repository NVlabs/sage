from torchvision import transforms
from typing import List
import numpy as np
import pickle
import torch
import trimesh.transformations as tra

from m2t2.dataset_utils import normalize_rgb
from m2t2.m2t2 import M2T2
from m2t2.pointnet2_utils import furthest_point_sample
from m2t2.meshcat_utils import (
    create_visualizer, visualize_pointcloud, visualize_grasp
)
from m2t2.rlbench_utils import (
    pcd_rgb_within_bound, gripper_pose_from_rlbench,
    gripper_pose_to_rlbench, rotation_to_rlbench
)
from m2t2.train_utils import to_gpu, to_cpu
from yarr.agents.agent import Agent, ActResult, Summary


class M2T2Agent(Agent):
    def __init__(self, cfg):
        super(M2T2Agent, self).__init__()
        self.cfg = cfg
        self.model = M2T2(**M2T2.from_config(self.cfg.m2t2))
        self.before = True
        self.after = False
        self.retract = False

    def build(self, training: bool, device=None) -> None:
        self.lang_emb = pickle.load(
            open(self.cfg.eval.lang_emb_path, 'rb')
        )
        self.model = self.model.to(device)
        self.model.eval()
        # self.vis = create_visualizer()

    def update(self, step: int, replay_sample: dict) -> dict:
        return {}

    def reset(self):
        self.before = True
        self.after = False
        self.retract = False

    def act(self, step: int, obs: dict, deterministic=False) -> ActResult:
        np.set_printoptions(suppress=True)
        # print(step, obs['gripper_open'], obs['gripper_joint_positions'])
        if self.before:
            # print(step, 'before')
            # print(obs['lang_goal'])
            pcd_raw, rgb_raw, mask = pcd_rgb_within_bound(
                obs, self.cfg.rlbench.cameras,
                self.cfg.rlbench.scene_bounds, channel_first=True
            )
            # print(pcd.shape, rgb.shape, mask.shape)
            # self.vis.delete()
            # visualize_pointcloud(self.vis, 'scene', pcd_raw, rgb_raw, size=0.01)

            pcd = torch.from_numpy(pcd_raw).float()
            rgb = normalize_rgb(
                torch.from_numpy(rgb_raw / 255).float().T.unsqueeze(-1)
            ).squeeze(-1).T
            # pt_idx = torch.randperm(pcd.shape[0])[:self.cfg.eval.num_points]
            pt_idx = furthest_point_sample(
                pcd.unsqueeze(0).cuda(), self.cfg.eval.num_points
            ).cpu().long()[0]
            pcd, rgb = pcd[pt_idx], rgb[pt_idx]
            data = {
                'inputs': torch.cat([pcd - pcd.mean(axis=0), rgb], dim=1),
                'points': pcd.unsqueeze(1).float(),
                'lang_tokens': torch.from_numpy(
                    self.lang_emb[obs['lang_goal']]
                ).float()
            }
            self.place = False
            if 'take' in obs['lang_goal'] or 'put' in obs['lang_goal']:
                if obs['gripper_open'] == 0:
                    obj_in_hand_id = 92 if 'steak' in obs['lang_goal'] else 83
                    if (mask == obj_in_hand_id).sum() > 0:
                        self.place = True
            if self.place:
                obj_pcd = pcd_raw[mask == obj_in_hand_id]
                obj_rgb = rgb_raw[mask == obj_in_hand_id]
                # visualize_pointcloud(self.vis, 'object', obj_pcd, obj_rgb, size=0.02)
                obj_pcd = torch.from_numpy(obj_pcd).float()
                obj_rgb = self.normalize_rgb(
                    torch.from_numpy(obj_rgb / 255).float().T.unsqueeze(-1)
                ).squeeze(-1).T
                pt_idx = furthest_point_sample(
                    obj_pcd.unsqueeze(0).cuda(), self.cfg.eval.num_obj_points
                ).cpu().long()[0]
                obj_pcd, obj_rgb = obj_pcd[pt_idx], obj_rgb[pt_idx]
                data['ee_pose'] = torch.from_numpy(
                    gripper_pose_from_rlbench(obs['gripper_matrix'][0])
                ).float()
                # make_frame(self.vis, 'end_effector', T=data['ee_pose'].double().numpy())
                inv_ee_pose = data['ee_pose'].inverse()
                obj_pcd_ee = obj_pcd @ inv_ee_pose[:3, :3].T + inv_ee_pose[:3, 3]
                obj_pcd_ee = obj_pcd_ee - obj_pcd_ee.mean(axis=0)
                data['object_points'] = obj_pcd
                data['object_inputs'] = torch.cat([obj_pcd_ee, obj_rgb], dim=1)
                data['task_is_pick'] = torch.tensor(False)
                data['task_is_place'] = torch.tensor(True)
            else:
                data['object_points'] = torch.rand(100, 3)
                data['object_inputs'] = torch.rand(100, 6)
                data['ee_pose'] = torch.eye(4)
                data['task_is_pick'] = torch.tensor(True)
                data['task_is_place'] = torch.tensor(False)
            to_gpu(data)
            for key in data:
                data[key] = data[key].unsqueeze(0)
                # print(key, data[key].shape)
            with torch.no_grad():
                outputs = self.model.infer(data, self.cfg)
            to_cpu(outputs)
            # for key in outputs:
            #     print(key, outputs[key][0].shape)

            self.params = outputs['params'][0].numpy()
            self.pose = gripper_pose_to_rlbench(outputs['actions'][0][0].numpy())
            trans = self.pose[:3, 3] - self.params[0] * self.pose[:3, 2]
            rot = rotation_to_rlbench(self.pose)
            gripper_open = not self.place
            self.before = False

            # mask = outputs['contact_masks'][0]
            # contacts = data['points'][0].cpu()[mask].numpy()
            # confidence = outputs['confidence'][0][mask].numpy()
            # colors = (confidence * np.array([[0, 255, 0]])).astype('uint8')
            # visualize_pointcloud(self.vis, 'contacts', contacts, colors, size=0.02)

            # action = outputs['actions'][0][0].numpy()
            # visualize_grasp(
            #     self.vis, f'action/at', action, colors[0], linewidth=5
            # )
            # before = action.copy()
            # before[:3, 3] -= self.params[0] * before[:3, 2]
            # visualize_grasp(
            #     self.vis, f'action/before', before,
            #     np.roll(colors[0], -1), linewidth=2
            # )
            # after = action.copy()
            # after[:3, 3] -= self.params[1] * after[:3, 2]
            # after[:3, :3] = after[:3, :3] @ tra.euler_matrix(
            #     0, 0, self.params[2]
            # )[:3, :3]
            # visualize_grasp(
            #     self.vis, f'action/after', after,
            #     np.roll(colors[0], -2), linewidth=2
            # )
            # retract = after.copy()
            # retract[:3, 3] -= retract[:3, 2] * self.params[3]
            # visualize_grasp(
            #     self.vis, f'action/retract', retract, colors[0], linewidth=2
            # )
            # input()
        elif self.after:
            # print(step, 'after')
            trans = self.pose[:3, 3] - self.params[1] * self.pose[:3, 2]
            rot = rotation_to_rlbench(
                self.pose @ tra.euler_matrix(0, 0, self.params[2])
            )
            gripper_open = self.params[4] > 0
            self.after = False
            self.retract = True
        elif self.retract:
            # print(step, 'retract')
            trans = self.pose[:3, 3] - (
                self.params[1] + self.params[3]
            ) * self.pose[:3, 2]
            rot = rotation_to_rlbench(self.pose)
            gripper_open = self.params[4] > 0
            self.retract = False
            self.before = True
        else:
            # print(step, 'act')
            trans = self.pose[:3, 3]
            rot = rotation_to_rlbench(self.pose)
            if self.place:
                gripper_open = 1
                trans = trans - 0.02 * self.pose[:3, 2]
                self.before = True
            else:
                gripper_open = 0
                self.after = True
        ignore_collision = 0
        action = np.concatenate([trans, rot, [gripper_open, ignore_collision]])
        return ActResult(action, {}, {})

    def act_summaries(self) -> List[Summary]:
        return []

    def update_summaries(self) -> List[Summary]:
        return []

    def load_weights(self, savedir: str) -> None:
        ckpt = torch.load(
            f'{savedir}/epoch_{self.cfg.eval.epoch}.pth',
            map_location='cpu'
        )
        self.model.load_state_dict(ckpt["model"])

    def save_weights(self, savedir: str) -> None:
        pass