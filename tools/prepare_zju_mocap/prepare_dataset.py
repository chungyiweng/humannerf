import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    '387.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'


def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")

    select_view = cfg['training_view']

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()
    
    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[select_view].astype('float32')
    cam_Rs = np.array(cams['R'])[select_view].astype('float32')
    cam_Ts = np.array(cams['T'])[select_view].astype('float32') / 1000.
    cam_Ds = np.array(cams['D'])[select_view].astype('float32')

    K = cam_Ks     #(3, 3)
    D = cam_Ds[:, 0]
    E = np.eye(4)  #(4, 4)
    cam_T = cam_Ts[:3, 0]
    E[:3, :3] = cam_Rs
    E[:3, 3]= cam_T
    
    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[select_view] \
            for multi_view_paths in img_path_frames_views
    ])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    output_path = os.path.join(cfg['output']['dir'], 
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path, exist_ok=True)
    out_img_dir  = prepare_dir(output_path, 'images')
    out_mask_dir = prepare_dir(output_path, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}
    mesh_infos = {}
    all_betas = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        out_name = 'frame_{:06d}'.format(idx)

        img_path = os.path.join(subject_dir, ipath)
    
        # load image
        img = np.array(load_image(img_path))

        if subject in ['313', '315']:
            _, image_basename, _ = split_path(img_path)
            start = image_basename.find(')_')
            smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = idx

        # load smpl parameters
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()

        betas = smpl_params['shapes'][0] #(10,)
        poses = smpl_params['poses'][0]  #(72,)
        Rh = smpl_params['Rh'][0]  #(3,)
        Th = smpl_params['Th'][0]  #(3,)
        
        all_betas.append(betas)

        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
        }

        # write mesh info
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses,
            'joints': joints, 
            'tpose_joints': tpose_joints
        }

        # load and write mask
        mask = get_mask(subject_dir, ipath)
        save_image(to_3ch_image(mask), 
                   os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)
        
    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)

if __name__ == '__main__':
    app.run(main)
