import copy
import os, sys
import pickle

import chainer
import numpy as np
import scipy.io
import time
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from . import pose_dataset_base
import math

# Joints in H3.6M -- data has 32 joints,
# but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'



def project_point_radial(P, R, T, f, c, k, p):
    """
    Project points from 3d to 2d using camera parameters
    including radial and tangential distortion
    Args
    P: Nx3 points in world coordinates
    R: 3x3 Camera rotation matrix
    T: 3x1 Camera translation parameters
    f: (scalar) Camera focal length
    c: 2x1 Camera center
    k: 3x1 Camera radial distortion coefficients
    p: 2x1 Camera tangential distortion coefficients
    Returns
    Proj: Nx2 points in pixel space
    D: 1xN depth of each point in camera space
    radial: 1xN radial distortion per point
    tan: 1xN tangential distortion per point
    r2: 1xN squared radius of the projected points before distortion
    """

    # P is a matrix of 3-dimensional points
    assert len(P.shape) == 2
    assert P.shape[1] == 3

    N = P.shape[0]
    X = R.dot(P.T - T)  # rotate and translate
    XX = X[:2, :] / X[2, :]
    r2 = XX[0, :] ** 2 + XX[1, :] ** 2

    radial = 1 + np.einsum(
        'ij,ij->j', np.tile(k, (1, N)), np.array([r2, r2 ** 2, r2 ** 3]))
    tan = p[0] * XX[1, :] + p[1] * XX[0, :]

    XXX = XX * np.tile(radial + tan, (2, 1)) + \
        np.outer(np.array([p[1], p[0]]).reshape(-1), r2)

    Proj = (f * XXX) + c
    Proj = Proj.T

    D = X[2,]

    return Proj, D, radial, tan, r2


class H36M(pose_dataset_base.PoseDatasetBase):

    def __init__(self, action='all',
                 train=True, use_sh_detection=False, frame_interval=1, frame_per=1, c=10):
        self.c = c
        length = frame_interval * frame_per
        if train:
            subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
        else:
            subjects = ['S9', 'S11']
        if not os.path.exists('data/h36m'):
            os.mkdir('data/h36m')

        # 3D joints
        if not os.path.exists('data/h36m/points_3d.pkl'):
            print('Downloading 3D points in Human3.6M dataset.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118616&authkey=AFIfEB6VYEZnhlE" -O ' + \
                'data/h36m/points_3d.pkl')
        with open('data/h36m/points_3d.pkl', 'rb') as f:
            p3d = pickle.load(f)
        # camera parameters
        if not os.path.exists('data/h36m/cameras.pkl'):
            print('Downloading camera parameters.')
            os.system('wget --no-check-certificate "https://onedriv' + \
                'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                'D60FE71FF90FD%2118615&authkey=AEUoi3s16rBTFRA" -O ' + \
                'data/h36m/cameras.pkl')
        with open('data/h36m/cameras.pkl', 'rb') as f:
            cams = pickle.load(f)
        # StackedHourglass
        if use_sh_detection:
            if not os.path.exists('data/h36m/sh_detect_2d.pkl'):
                print('Downloading detected 2D points by Stacked Hourglass.')
                os.system('wget --no-check-certificate "https://onedriv' + \
                    'e.live.com/download?cid=B08D60FE71FF90FD&resid=B08' + \
                    'D60FE71FF90FD%2118619&authkey=AMBf6RPcWQgjsh0" -O ' + \
                    'data/h36m/sh_detect_2d.pkl')
            with open('data/h36m/sh_detect_2d.pkl', 'rb') as f:
                p2d_sh = pickle.load(f)

        with open('data/actions.txt') as f:
            actions_all = f.read().split('\n')[:-1]
        if action == 'all':
            actions = actions_all
        elif action in actions_all:
            actions = [action]
        else:
            raise Exception('Invalid action.')

        # 17 joints
        dim_to_use_x = np.where(np.array([x != '' for x in H36M_NAMES]))[0] * 3
        dim_to_use_y = dim_to_use_x + 1
        dim_to_use_z = dim_to_use_x + 2
        dim_to_use = np.array(
            [dim_to_use_x, dim_to_use_y, dim_to_use_z]).T.flatten()
        self.N = len(dim_to_use_x)

        p3d = copy.deepcopy(p3d)
        self.data_list = []
        for s in subjects:
            for action_name in actions:
                def search(a):
                    fs = list(filter(
                        lambda x: x.split()[0] == a, p3d[s].keys()))
                    return fs

                files = []
                files += search(action_name)
                # 'Photo' is 'TakingPhoto' in S1
                if action_name == 'Photo':
                    files += search('TakingPhoto')
                # 'WalkDog' is 'WalkingDog' in S1
                if action_name == 'WalkDog':
                    files += search('WalkingDog')
                for file_name in files:
                    p3d[s][file_name] = p3d[s][file_name][:, dim_to_use]
                    L = p3d[s][file_name].shape[0]
                    for cam_name in cams[s].keys():
                        if not (cam_name == '54138969' and s == 'S11' \
                                and action_name == 'Directions'):
                            # 50Hz -> 10Hz
                            if cam_name == '54138969':
                                for start_pos in range(0, L - length + 1, length if train else 5):
                                    info = {'subject': s,
                                            'action_name': action_name,
                                            'start_pos': start_pos,
                                            'length': length,
                                            'cam_name': cam_name,
                                            'file_name': file_name}
                                    self.data_list.append(info)
        self.p3d = p3d
        self.cams = cams
        self.train = train
        self.use_sh_detection = use_sh_detection
        if use_sh_detection:
            self.p2d_sh = p2d_sh

    def __len__(self):
        return len(self.data_list)

    def get_example(self, i):
        info = self.data_list[i]
        subject = info['subject']
        start_pos = info['start_pos']
        length = info['length']
        cam_name = info['cam_name']
        file_name = info['file_name']

        poses_xyz = self.p3d[subject][file_name][start_pos:start_pos + length]
        params = self.cams[subject][cam_name]
        if self.use_sh_detection:
            if 'TakingPhoto' in file_name:
                file_name = file_name.replace('TakingPhoto', 'Photo')
            if 'WalkingDog' in file_name:
                file_name = file_name.replace('WalkingDog', 'WalkDog')
            sh_detect_xy = self.p2d_sh[subject][file_name]
            sh_detect_xy = sh_detect_xy[cam_name][start_pos:start_pos+length] # load adata

        if 'TakingPhoto' in file_name:
            file_name = file_name.replace('TakingPhoto', 'Photo')
        if 'WalkingDog' in file_name:
            file_name = file_name.replace('WalkingDog', 'WalkDog')

        # world->camera
        P = poses_xyz.reshape(length, 17, 3) # 34, 3
        X = params['R'].dot(P[0].T).T
        for j in range(1, length):
            X1 = params['R'].dot(P[j].T).T
            X = np.concatenate((X, X1), axis=0)
        X = X.reshape(length, self.N * 3)  # shape=(length, 3*n_joints)

        # Normalization of 3d points.
        scale = np.empty((length), np.float32)
        A = np.empty((length, self.N * 3), np.float32)

        for j in range(length):
            A[j], scale[j] = self._normalize_3d(X[j], self.c)

        A = A.astype(np.float32)
        scale = scale.astype(np.float32)
        
        if self.use_sh_detection:
            # normalization for 2D poses from SH
            proj = sh_detect_xy[0].reshape(-1, self.N * 2)
            proj = self._normalize_2d(proj, self.c)
            proj = proj.astype(np.float32)
            for j in range(1, length):
                proj1 = sh_detect_xy[j].reshape(-1, self.N * 2)
                proj1 = self._normalize_2d(proj1, self.c)
                proj1 = proj1.astype(np.float32)
                proj = np.concatenate((proj, proj1), axis=0)
            c = np.float32(self.c)
            return proj, A, scale, c
        else:
            # camera->image
            proj = project_point_radial(P[0], **params)[0]
            proj = proj.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
            proj = self._normalize_2d(proj, self.c)
            proj = proj.astype(np.float32)

            for j in range(1, length):
                proj1 = project_point_radial(P[j], **params)[0]
                proj1 = proj1.reshape(-1, self.N * 2)  # shape=(length, 2*n_joints)
                proj1 = self._normalize_2d(proj1, self.c)
                proj1 = proj1.astype(np.float32)
                proj = np.concatenate((proj, proj1), axis=0)

            c = np.float32(self.c)

            return proj, A, scale, c


class MPII(chainer.dataset.DatasetMixin):
    def __init__(self, train=True, use_sh_detection=False, c=10):
        self.c = c
        if use_sh_detection:
            raise NotImplementedError
        else:
            self.poses = np.load('data/mpii_poses.npy')

        np.random.seed(100)
        perm = np.random.permutation(len(self.poses))

    def __len__(self):
        return self.poses.shape[0]

    def get_example(self, i):
        mpii_poses = self.poses[i:i+1]

        # average distance to root joint -> 1
        xs = mpii_poses.T[0::2] - mpii_poses.T[0]
        ys = mpii_poses.T[1::2] - mpii_poses.T[1]
        # hip -> root joint
        mpii_poses = mpii_poses.T / (np.sqrt(xs[1:]**2 + ys[1:]**2).mean(axis=0) * self.c)
        mpii_poses[0::2] -= mpii_poses[0]
        mpii_poses[1::2] -= mpii_poses[1]
        mpii_poses = mpii_poses.T.astype(np.float32)[None]

        dummy_X = np.zeros((1, 17*3), dtype=np.float32)
        dummy_X[0, 0::3] = mpii_poses[0, 0, 0::2]
        dummy_X[0, 1::3] = mpii_poses[0, 0, 1::2]

        dummy_scale = np.array([1], dtype=np.float32)

        mpii_1 = np.zeros((34), dtype=np.float32)
        mpii_1[0::1] = mpii_poses[0, 0, 0::1]
        c = np.float32(self.c)

        return mpii_1, dummy_X, dummy_scale, c



class LSP(chainer.dataset.DatasetMixin):
    def __init__(self, train=True, use_sh_detection=False, c=10):
        self.c = c

        joint_idx = [2,1,0,3,4,5,12,13,9,10,11,8,7,6]

        self.poses = scipy.io.loadmat('data/joints.mat')['joints']
        self.poses = self.poses[:2, joint_idx, :] # 2, 14, 2000
        a = (self.poses[:, 0:1, :] + self.poses[:, 3:4, :]) / 2

        self.posesnew = np.concatenate((a, self.poses[:, :6], a, self.poses[:, 6:7], a, self.poses[:, 7:]), axis=1).transpose(2, 1, 0).reshape(2000, 34)

        time.sleep(10)

    def __len__(self):
        return self.posesnew.shape[0]

    def get_example(self, i):
        lsp_poses = self.posesnew[i:i+1]

        xs = lsp_poses.T[0::2] - lsp_poses.T[0]
        ys = lsp_poses.T[1::2] - lsp_poses.T[1]
        lsp_poses = lsp_poses.T / (np.sqrt(xs[1:]**2 + ys[1:]**2).mean(axis=0) * self.c)
        lsp_poses[0::2] -= lsp_poses[0]
        lsp_poses[1::2] -= lsp_poses[1]
        lsp_poses = lsp_poses.T.astype(np.float32)[None]

        dummy_X = np.zeros((1, 17*3), dtype=np.float32)
        dummy_X[0, 0::3] = lsp_poses[0, 0, 0::2]
        dummy_X[0, 1::3] = lsp_poses[0, 0, 1::2]

        dummy_scale = np.array([1], dtype=np.float32)

        lsp_1 = np.zeros((34), dtype=np.float32)
        lsp_1[0::1] = lsp_poses[0, 0, 0::1]
        c = np.float32(self.c)

        return lsp_1, dummy_X, dummy_scale, c
