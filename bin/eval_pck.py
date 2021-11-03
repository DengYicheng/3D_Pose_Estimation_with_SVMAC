import chainer
from chainer import cuda
from chainer import dataset
from chainer import iterators
from chainer import serializers
from chainer import Variable

import chainer.functions as F

import argparse
import json
import os
import sys
import cupy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "."))
import evaluation_util

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan


def p_mpjpe(predicted, target):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY  # Scale
    t = muX - a * np.matmul(muY, R)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, R) + t

    return predicted_aligned



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--action', '-a', type=str, default='')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--use_mpii', action="store_true")
    parser.add_argument('--use_mpii_inf_3dhp', action="store_true")
    args = parser.parse_args()

    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.json')) as f:
        opts = json.load(f)
    PCK_THRESHOLD = 150  # see "Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision"
    gen = projection_gan.pose.posenet.MLP(mode='generator',
        use_bn=opts['use_bn'], activate_func=getattr(F, opts['activate_func']))
    serializers.load_npz(args.model_path, gen)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()

    if opts['action'] == 'all':
        with open(os.path.join('data', 'actions.txt')) as f:
            actions = f.read().split('\n')[:-1]
    else:
        actions = [opts['action']]

    errors = []

    for act_name in actions:
        test = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.MPII3DDataset(
            annotations_glob="/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/*/*/annot.mat", train=False)
        test_iter = chainer.iterators.SerialIterator(test, batch_size=args.batch, shuffle=False, repeat=False)

        total = 0
        true_positive = 0
        AUC = 0
        AUC_total = 0
        AUC_true_positive = 0

        p2_total = 0
        p2_true_positive = 0
        p2_AUC = 0
        p2_AUC_total = 0
        p2_AUC_true_positive = 0
        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            for batch in test_iter:
                batchsize = len(batch)
                xy_proj, xyz, scale, c = dataset.concat_examples(
                    batch, device=args.gpu)
                xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
                with chainer.no_backprop_mode(), \
                        chainer.using_config('train', False):
                    xy_real = chainer.Variable(xy_proj)
                    xyz_real = chainer.Variable(xyz)

                    c = Variable(c)
                    Cs = F.expand_dims(c, axis=1)
                    Cs = F.repeat(Cs, 17, axis=1)
                    d_pred, cam_pred = gen(xy_real)  # d batch, 16
                    d0 = Variable(cupy.array([[0.0]], np.float32))
                    d0 = F.repeat(d0, batchsize, axis=0)
                    d_pred = F.concat([d0, d_pred], axis=1)
                    d_pred = d_pred + Cs  # batch, 16

                    z_pred = d_pred

                    x_pred = z_pred * xy_real[:, 0::2]
                    y_pred = z_pred * xy_real[:, 1::2]

                    xyz_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None], z_pred[:, :, None]),
                                        axis=2)  # batchsize 17 3
                    xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]


                    xyz_pred1 = F.transpose(xyz_pred, axes=[0, 2, 1]).data  # [batchsize, 17, 3]


                    target = cupy.asnumpy(xyz.reshape(-1, 17, 3)) # [batchsize, 17, 3]
                    aligned_pose = p_mpjpe(cupy.asnumpy(xyz_pred1), cupy.asnumpy(xyz.reshape(-1, 17, 3))) # [~, 17, 3]
                    p_mp = np.mean(np.linalg.norm(aligned_pose - target, axis=len(target.shape) - 1))
                    p_mp = p_mp * cupy.asnumpy(scale[:, 0])

                    p2_z_pred = aligned_pose[:, :, 2][:, evaluation_util.JointsForPCK.from_h36m_joints]
                    p2_z_real = target[:, :, 2][:, evaluation_util.JointsForPCK.from_h36m_joints]
                    p2_x_pred = aligned_pose[:, :, 0][:, evaluation_util.JointsForPCK.from_h36m_joints]
                    p2_x_real = target[:, :, 0][:, evaluation_util.JointsForPCK.from_h36m_joints]
                    p2_y_pred = aligned_pose[:, :, 1][:, evaluation_util.JointsForPCK.from_h36m_joints]
                    p2_y_real = target[:, :, 1][:, evaluation_util.JointsForPCK.from_h36m_joints]

                    scale1 = cupy.asnumpy(scale)

                    per_joint_error = np.sqrt((p2_z_pred - p2_z_real) * (p2_z_pred - p2_z_real) + (p2_x_pred - p2_x_real) * (p2_x_pred - p2_x_real) +
                                              (p2_y_pred - p2_y_real) * (p2_y_pred - p2_y_real)) * scale1.reshape((-1, 1))
                    p2_true_positive += (per_joint_error < PCK_THRESHOLD).sum()
                    p2_total += per_joint_error.size
                    for i in range(0, 150, 5):
                        p2_AUC_true_positive += (per_joint_error < i).sum()


                    z_pred = z_pred.data[:, evaluation_util.JointsForPCK.from_h36m_joints]
                    z_real = xyz[:, 2::3][:, evaluation_util.JointsForPCK.from_h36m_joints]

                    per_joint_error = gen.xp.sqrt((z_pred - z_real) * (z_pred - z_real)) * scale.reshape((-1, 1))
                    true_positive += (per_joint_error < PCK_THRESHOLD).sum()
                    total += per_joint_error.size
                    for i in range(0, 150, 5):
                        AUC_true_positive += (per_joint_error < i).sum()

            print('PCK: ' + str(float(true_positive) / total * 100))
            print('AUC: ' + str(float(AUC_true_positive)  / total / 30 * 100))
            print('p2_PCK: ' + str(float(p2_true_positive) / total * 100))
            print('p2_AUC: ' + str(float(p2_AUC_true_positive) / total / 30 * 100))
            test_iter.finalize()

    with open(args.model_path.replace('.npz', '.csv'), 'w') as f:
        for act_name, error in zip(actions, errors):
            f.write('{},{}\n'.format(act_name, error))
        f.write('{},{}\n'.format('average', sum(errors) / len(errors)))

