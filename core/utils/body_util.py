from math import cos, sin

import numpy as np

SMPL_JOINT_IDX = {
    'pelvis_root': 0,
    'left_hip': 1,
    'right_hip': 2,
    'belly_button': 3,
    'left_knee': 4,
    'right_knee': 5,
    'lower_chest': 6,
    'left_ankle': 7,
    'right_ankle': 8,
    'upper_chest': 9,
    'left_toe': 10,
    'right_toe': 11,
    'neck': 12,
    'left_clavicle': 13,
    'right_clavicle': 14,
    'head': 15,
    'left_shoulder': 16,
    'right_shoulder': 17,
    'left_elbow': 18,
    'right_elbow': 19,
    'left_wrist': 20,
    'right_wrist': 21,
    'left_thumb': 22,
    'right_thumb': 23
}

SMPL_PARENT = {
    1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 
    11: 8, 12: 9, 13: 9, 14: 9, 15: 12, 16: 13, 17: 14, 18: 16, 19: 17, 20: 18, 
    21: 19, 22: 20, 23: 21}

TORSO_JOINTS_NAME = [
    'pelvis_root', 'belly_button', 'lower_chest', 'upper_chest', 'left_clavicle', 'right_clavicle'
]
TORSO_JOINTS = [
    SMPL_JOINT_IDX[joint_name] for joint_name in TORSO_JOINTS_NAME
]
BONE_STDS = np.array([0.03, 0.06, 0.03])
HEAD_STDS = np.array([0.06, 0.06, 0.06])
JOINT_STDS = np.array([0.02, 0.02, 0.02])


def _to_skew_matrix(v):
    r""" Compute the skew matrix given a 3D vectors.

    Args:
        - v: Array (3, )

    Returns:
        - Array (3, 3)

    """
    vx, vy, vz = v.ravel()
    return np.array([[0, -vz, vy],
                    [vz, 0, -vx],
                    [-vy, vx, 0]])


def _to_skew_matrices(batch_v):
    r""" Compute the skew matrix given 3D vectors. (batch version)

    Args:
        - batch_v: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    """
    batch_size = batch_v.shape[0]
    skew_matrices = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)

    for i in range(batch_size):
        skew_matrices[i] = _to_skew_matrix(batch_v[i])

    return skew_matrices


def _get_rotation_mtx(v1, v2):
    r""" Compute the rotation matrices between two 3D vector. (batch version)
    
    Args:
        - v1: Array (N, 3)
        - v2: Array (N, 3)

    Returns:
        - Array (N, 3, 3)

    Reference:
        https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """

    batch_size = v1.shape[0]
    
    v1 = v1 / np.clip(np.linalg.norm(v1, axis=-1, keepdims=True), 1e-5, None)
    v2 = v2 / np.clip(np.linalg.norm(v2, axis=-1, keepdims=True), 1e-5, None)
    
    normal_vec = np.cross(v1, v2, axis=-1)
    cos_v = np.zeros(shape=(batch_size, 1))
    for i in range(batch_size):
        cos_v[i] = v1[i].dot(v2[i])

    skew_mtxs = _to_skew_matrices(normal_vec)
    
    Rs = np.zeros(shape=(batch_size, 3, 3), dtype=np.float32)
    for i in range(batch_size):
        Rs[i] = np.eye(3) + skew_mtxs[i] + \
                    (skew_mtxs[i].dot(skew_mtxs[i])) * (1./(1. + cos_v[i]))
    
    return Rs


def _construct_G(R_mtx, T):
    r""" Build 4x4 [R|T] matrix from rotation matrix, and translation vector
    
    Args:
        - R_mtx: Array (3, 3)
        - T: Array (3,)

    Returns:
        - Array (4, 4)
    """

    G = np.array(
        [[R_mtx[0, 0], R_mtx[0, 1], R_mtx[0, 2], T[0]],
         [R_mtx[1, 0], R_mtx[1, 1], R_mtx[1, 2], T[1]],
         [R_mtx[2, 0], R_mtx[2, 1], R_mtx[2, 2], T[2]],
         [0.,          0.,          0.,          1.]],
        dtype='float32')

    return G
    

def _deform_gaussian_volume(
        grid_size, 
        bbox_min_xyz,
        bbox_max_xyz,
        center, 
        scale_mtx, 
        rotation_mtx):
    r""" Deform a standard Gaussian volume.
    
    Args:
        - grid_size:    Integer
        - bbox_min_xyz: Array (3, )
        - bbox_max_xyz: Array (3, )
        - center:       Array (3, )   - center of Gaussain to be deformed
        - scale_mtx:    Array (3, 3)  - scale of Gaussain to be deformed
        - rotation_mtx: Array (3, 3)  - rotation matrix of Gaussain to be deformed

    Returns:
        - Array (grid_size, grid_size, grid_size)
    """

    R = rotation_mtx
    S = scale_mtx

    # covariance matrix after scaling and rotation
    SIGMA = R.dot(S).dot(S).dot(R.T)

    min_x, min_y, min_z = bbox_min_xyz
    max_x, max_y, max_z = bbox_max_xyz
    zgrid, ygrid, xgrid = np.meshgrid(
        np.linspace(min_z, max_z, grid_size),
        np.linspace(min_y, max_y, grid_size),
        np.linspace(min_x, max_x, grid_size),
        indexing='ij')
    grid = np.stack([xgrid - center[0], 
                     ygrid - center[1], 
                     zgrid - center[2]],
                    axis=-1)

    dist = np.einsum('abci, abci->abc', np.einsum('abci, ij->abcj', grid, SIGMA), grid)

    return np.exp(-1 * dist)


def _std_to_scale_mtx(stds):
    r""" Build scale matrix from standard deviations
    
    Args:
        - stds: Array(3,)

    Returns:
        - Array (3, 3)
    """

    scale_mtx = np.eye(3, dtype=np.float32)
    scale_mtx[0][0] = 1.0/stds[0]
    scale_mtx[1][1] = 1.0/stds[1]
    scale_mtx[2][2] = 1.0/stds[2]

    return scale_mtx


def _rvec_to_rmtx(rvec):
    r''' apply Rodriguez Formula on rotate vector (3,)

    Args:
        - rvec: Array (3,)

    Returns:
        - Array (3, 3)
    '''
    rvec = rvec.reshape(3, 1)

    norm = np.linalg.norm(rvec)
    theta = norm
    r = rvec / (norm + 1e-5)

    skew_mtx = _to_skew_matrix(r)

    return cos(theta)*np.eye(3) + \
           sin(theta)*skew_mtx + \
           (1-cos(theta))*r.dot(r.T)


def body_pose_to_body_RTs(jangles, tpose_joints):
    r""" Convert body pose to global rotation matrix R and translation T.
    
    Args:
        - jangles (joint angles): Array (Total_Joints x 3, )
        - tpose_joints:           Array (Total_Joints, 3)

    Returns:
        - Rs: Array (Total_Joints, 3, 3)
        - Ts: Array (Total_Joints, 3)
    """

    jangles = jangles.reshape(-1, 3)
    total_joints = jangles.shape[0]
    assert tpose_joints.shape[0] == total_joints

    Rs = np.zeros(shape=[total_joints, 3, 3], dtype='float32')
    Rs[0] = _rvec_to_rmtx(jangles[0,:])

    Ts = np.zeros(shape=[total_joints, 3], dtype='float32')
    Ts[0] = tpose_joints[0,:]

    for i in range(1, total_joints):
        Rs[i] = _rvec_to_rmtx(jangles[i,:])
        Ts[i] = tpose_joints[i,:] - tpose_joints[SMPL_PARENT[i], :]
    
    return Rs, Ts


def get_canonical_global_tfms(canonical_joints):
    r""" Convert canonical joints to 4x4 global transformation matrix.
    
    Args:
        - canonical_joints: Array (Total_Joints, 3)

    Returns:
        - Array (Total_Joints, 4, 4)
    """

    total_bones = canonical_joints.shape[0]

    gtfms = np.zeros(shape=(total_bones, 4, 4), dtype='float32')
    gtfms[0] = _construct_G(np.eye(3), canonical_joints[0,:])

    for i in range(1, total_bones):
        translate = canonical_joints[i,:] - canonical_joints[SMPL_PARENT[i],:]
        gtfms[i] = gtfms[SMPL_PARENT[i]].dot(
                            _construct_G(np.eye(3), translate))

    return gtfms


def approx_gaussian_bone_volumes(
    tpose_joints, 
    bbox_min_xyz, bbox_max_xyz,
    grid_size=32):
    r""" Compute approximated Gaussian bone volume.
    
    Args:
        - tpose_joints:  Array (Total_Joints, 3)
        - bbox_min_xyz:  Array (3, )
        - bbox_max_xyz:  Array (3, )
        - grid_size:     Integer
        - has_bg_volume: boolean

    Returns:
        - Array (Total_Joints + 1, 3, 3, 3)
    """

    total_joints = tpose_joints.shape[0]

    grid_shape = [grid_size] * 3
    tpose_joints = tpose_joints.astype(np.float32)

    calibrated_bone = np.array([0.0, 1.0, 0.0], dtype=np.float32)[None, :]
    g_volumes = []
    for joint_idx in range(0, total_joints):
        gaussian_volume = np.zeros(shape=grid_shape, dtype='float32')

        is_parent_joint = False
        for bone_idx, parent_idx in SMPL_PARENT.items():
            if joint_idx != parent_idx:
                continue

            S = _std_to_scale_mtx(BONE_STDS * 2.)
            if joint_idx in TORSO_JOINTS:
                S[0][0] *= 1/1.5
                S[2][2] *= 1/1.5

            start_joint = tpose_joints[SMPL_PARENT[bone_idx]]
            end_joint = tpose_joints[bone_idx]
            target_bone = (end_joint - start_joint)[None, :]

            R = _get_rotation_mtx(calibrated_bone, target_bone)[0].astype(np.float32)

            center = (start_joint + end_joint) / 2.0

            bone_volume = _deform_gaussian_volume(
                            grid_size, 
                            bbox_min_xyz,
                            bbox_max_xyz,
                            center, S, R)
            gaussian_volume = gaussian_volume + bone_volume

            is_parent_joint = True

        if not is_parent_joint:
            # The joint is not other joints' parent, meaning it is an end joint
            joint_stds = HEAD_STDS if joint_idx == SMPL_JOINT_IDX['head'] else JOINT_STDS
            S = _std_to_scale_mtx(joint_stds * 2.)

            center = tpose_joints[joint_idx]
            gaussian_volume = _deform_gaussian_volume(
                                grid_size, 
                                bbox_min_xyz,
                                bbox_max_xyz,
                                center, 
                                S, 
                                np.eye(3, dtype='float32'))
            
        g_volumes.append(gaussian_volume)
    g_volumes = np.stack(g_volumes, axis=0)

    # concatenate background weights
    bg_volume = 1.0 - np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.0, max=1.0)
    g_volumes = np.concatenate([g_volumes, bg_volume], axis=0)
    g_volumes = g_volumes / np.sum(g_volumes, axis=0, keepdims=True).clip(min=0.001)
    
    return g_volumes
