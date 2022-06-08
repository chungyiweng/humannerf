import os
import pickle

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.camera_util import \
    get_camrot, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg


class Dataset(torch.utils.data.Dataset):
    RENDER_SIZE=512
    CAM_PARAMS = {
        'radius': 6.0, 'focal': 1250.
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            bgcolor=None,
            src_type="zju_mocap",
            **_):
        print('[Dataset Path]', dataset_path) 

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()
            
        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints, 
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')

        self.total_frames = cfg.render_frames
        print(f' -- Total Frames: {self.total_frames}')

        self.img_size = self.RENDER_SIZE

        K, E = self.setup_camera(img_size = self.img_size, 
                                 **self.CAM_PARAMS)
        self.camera = {
            'K': K,
            'E': E
        }

        self.bgcolor = bgcolor if bgcolor is not None else [255., 255., 255.]
        self.keyfilter = keyfilter

    @staticmethod
    def setup_camera(img_size, radius, focal):
        x = 0.
        y = -0.25
        z = radius
        campos = np.array([x, y, z], dtype='float32')
        camrot = get_camrot(campos, 
                            lookat=np.array([0, y, 0.]),
                            inv_camera=True)

        E = np.eye(4, dtype='float32')
        E[:3, :3] = camrot
        E[:3, 3] = -camrot.dot(campos)

        K = np.eye(3, dtype='float32')
        K[0, 0] = focal
        K[1, 1] = focal
        K[:2, 2] = img_size / 2.

        return K, E

    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    @staticmethod
    def rotate_bbox(bbox, rmtx):
        min_x, min_y, min_z = bbox['min_xyz']
        max_x, max_y, max_z = bbox['max_xyz']

        bbox_pts = np.array(
            [[min_x, min_y, min_z],
             [min_x, min_y, max_z],
             [min_x, max_y, min_z],
             [min_x, max_y, max_z],
             [max_x, min_y, min_z],
             [max_x, min_y, max_z],
             [max_x, max_y, min_z],
             [max_x, max_y, max_z],])

        rotated_bbox_pts = bbox_pts.dot(rmtx)
        rotated_bbox = {
            'min_xyz': np.min(rotated_bbox_pts, axis=0),
            'max_xyz': np.max(rotated_bbox_pts, axis=0)
        }

        return rotated_bbox

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        results = {}

        bgcolor = np.array(self.bgcolor, dtype='float32')

        H = W = self.img_size

        # load t-pose
        dst_bbox = self.canonical_bbox.copy()
        dst_poses = np.zeros(72, dtype='float32')
        dst_skel_joints = self.canonical_joints.copy()

        # rotate body
        angle = 2 * np.pi / self.total_frames * idx
        add_rmtx = cv2.Rodrigues(np.array([0, -angle, 0], dtype='float32'))[0]
        root_rmtx = cv2.Rodrigues(dst_poses[:3])[0]
        new_root_rmtx = add_rmtx@root_rmtx
        dst_poses[:3] = cv2.Rodrigues(new_root_rmtx)[0][:, 0]

        # rotate boundinig box
        dst_bbox = self.rotate_bbox(dst_bbox, add_rmtx)

        K = self.camera['K'].copy()
        E = self.camera['E'].copy()
        R = E[:3, :3]
        T = E[:3, 3]

        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T) 
        rays_o = rays_o.reshape(-1, 3)# (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
    
        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor})

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_skel_joints
                )
            cnl_gtfms = get_canonical_global_tfms(self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })                                    

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })
        
        return results
