# -*- coding: utf-8 -*-

import chainer
from chainer import cuda
from chainer import dataset
from chainer import iterators
from chainer import serializers
from chainer import Variable


import chainer.functions as F
import numpy as np
import cupy
import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import projection_gan
import time


def plot17j(posesreal, poses3d, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()

    if not show_animation:
        plot_idx = 1

        frames = np.linspace(start=0, stop=poses3d.shape[0]-1, num=10).astype(int)

        for i in frames:
            ax = fig.add_subplot(2,5, plot_idx, projection='3d')

            pose = poses3d[i]
            # pose = posesreal[i]

            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]
            ax.view_init(elev=90, azim=-90) #MPII
            # ax.view_init(elev=90, azim=-90) #LSP  正常角度 90 -90

            # x = cupy.asnumpy(pose[0].data)
            # y = cupy.asnumpy(-pose[1].data)
            # z = cupy.asnumpy(pose[2].data)
            x = pose[0]
            y = pose[1] * -1
            z = pose[2]
            # x = pose[0:16]
            # y = pose[16:32] * -1
            # z = pose[32:48]
            # for MPII

            # ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])],'b')
            # ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])],'r')
            # ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])],'r')
            # ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], 'b')
            # ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], 'g')
            # ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], 'g')
            # ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])],'b')
            # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])],'b')
            # ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])],'b')
            # ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])],'b')
            # ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])],'b')
            # ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])],'g')
            # ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])],'g')
            # ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])],'b')
            # ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])],'r')
            # ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])],'r')

            # ax.axis('equal')
            # ax.axis('off')
            #
            # ax.set_title('frame = ' + str(i))
            #
            # plot_idx += 1

            # ax = fig.add_subplot(2,5, plot_idx, projection='3d')

            # pose = poses3d[i]
            # posereal = posesreal[i]

            # x = pose[0:16]
            # y = pose[16:32]
            # z = pose[32:48]
            # ax.view_init(elev=90, azim=-80) #MPII
            # ax.view_init(elev=90, azim=-90)  # LSP
            #
            # x = posereal[0]
            # y = posereal[1] * -1
            # z = posereal[2]
            # ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])], 'b')
            # ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])], 'r')
            # ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], 'r')
            # ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], 'b')
            # ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], 'g')
            # ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], 'g')
            # ax.plot(x[([0, 7])], y[([0, 7])], z[([0, 7])], 'b')
            # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])], 'b')
            # ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])], 'b')
            # ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], 'b')
            # ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])], 'b')
            # ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])], 'g')
            # ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], 'g')
            # ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])], 'b')
            # ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])], 'r')
            # ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])], 'r')

            # # for LSP
            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], 'b')
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], 'g')
            ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], 'g')
            ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])], 'b')
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])], 'r')
            ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], 'r')
            # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])], 'b')
            ax.plot(x[([8, 10])], y[([8, 10])], z[([8, 10])], 'b')
            # ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], 'b')
            ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])], 'b')
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])], 'g')
            ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], 'g')
            ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])], 'b')
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])], 'r')
            ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])], 'r')
            ax.plot(x[([0, 8])], y[([0, 8])], z[([0, 8])], 'b')



            # Create cubic bounding box to simulate equal aspect ratio
            # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            # for xb, yb, zb in zip(Xb, Yb, Zb):
            #     ax.plot([xb], [yb], [zb], 'w')

            ax.axis('equal')
            ax.axis('off')

            ax.set_title('frame = ' + str(i))

            plot_idx += 1

        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        # plt.show()
        plt.savefig('aa.jpg')

def plot17j2d(poses2d, show_animation=False):
    import matplotlib as mpl
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')
    import numpy as np
    import matplotlib.animation as anim

    from mpl_toolkits.mplot3d import axes3d, Axes3D

    fig = plt.figure()

    if not show_animation:
        plot_idx = 1

        frames = np.linspace(start=0, stop=poses2d.shape[0] - 1, num=10).astype(int)

        # for i in frames:
        fig = plt.figure(10)
        # ax = fig.add_subplot(2, 5, plot_idx, projection='2d')

        pose = poses2d[40]

        # x = pose[0:16]
        # y = pose[16:32]
        # z = pose[32:48]
        # ax.view_init(elev=90, azim=-80) #MPII
        # ax.view_init(elev=90, azim=-90)  # LSP

        x = pose[0::2]
        y = -pose[1::2]
        # z = cupy.asnumpy(pose[2].data)
        # for MPII

        # plt.plot(x[([0, 4])], y[([0, 4])], color='b', linewidth=4)
        # plt.plot(x[([4, 5])], y[([4, 5])], color='r', linewidth=4)
        # plt.plot(x[([5, 6])], y[([5, 6])], color='r', linewidth=4)
        # plt.plot(x[([0, 1])], y[([0, 1])], color='b', linewidth=4)
        # plt.plot(x[([1, 2])], y[([1, 2])], color='g', linewidth=4)
        # plt.plot(x[([2, 3])], y[([2, 3])], color='g', linewidth=4)
        # plt.plot(x[([0, 7])], y[([0, 7])], color='b', linewidth=4)
        # plt.plot(x[([7, 8])], y[([7, 8])], color='b', linewidth=4)
        # plt.plot(x[([8, 9])], y[([8, 9])], color='b', linewidth=4)
        # plt.plot(x[([9, 10])], y[([9, 10])], color='b', linewidth=4)
        # plt.plot(x[([8, 11])], y[([8, 11])], color='b', linewidth=4)
        # plt.plot(x[([11, 12])], y[([11, 12])], color='g', linewidth=4)
        # plt.plot(x[([12, 13])], y[([12, 13])], color='g', linewidth=4)
        # plt.plot(x[([8, 14])], y[([8, 14])], color='b', linewidth=4)
        # plt.plot(x[([14, 15])], y[([14, 15])], color='r', linewidth=4)
        # plt.plot(x[([15, 16])], y[([15, 16])], color='r', linewidth=4)

        # ax.axis('equal')
        # ax.axis('off')
        #
        # ax.set_title('frame = ' + str(i))

        # plot_idx += 1


        # for LSP
        ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])], 'b')
        ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])], 'g')
        ax.plot(x[([2, 3])], y[([2, 3])], z[([2, 3])], 'g')
        ax.plot(x[([0, 4])], y[([0, 4])], z[([0, 4])], 'b')
        ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])], 'r')
        ax.plot(x[([5, 6])], y[([5, 6])], z[([5, 6])], 'r')
        # ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])], 'b')
        ax.plot(x[([8, 10])], y[([8, 10])], z[([8, 10])], 'b')
        # ax.plot(x[([9, 10])], y[([9, 10])], z[([9, 10])], 'b')
        ax.plot(x[([8, 11])], y[([8, 11])], z[([8, 11])], 'b')
        ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])], 'g')
        ax.plot(x[([12, 13])], y[([12, 13])], z[([12, 13])], 'g')
        ax.plot(x[([8, 14])], y[([8, 14])], z[([8, 14])], 'b')
        ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])], 'r')
        ax.plot(x[([15, 16])], y[([15, 16])], z[([15, 16])], 'r')
        ax.plot(x[([0, 8])], y[([0, 8])], z[([0, 8])], 'b')



        # Create cubic bounding box to simulate equal aspect ratio
        # max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
        # Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
        # Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
        # Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

        # for xb, yb, zb in zip(Xb, Yb, Zb):
        #     ax.plot([xb], [yb], [zb], 'w')


        # this uses QT5Agg backend
        # you can identify the backend using plt.get_backend()
        # delete the following two lines and resize manually if it throws an error
        # figManager = plt.get_current_fig_manager()
        # figManager.window.showMaximized()

        # plt.show()
        plt.savefig('bb.jpg')

    else:
        def update(i):

            ax.clear()

            pose = poses[i]

            x = pose[0:16]
            y = pose[16:32]
            z = pose[32:48]
            ax.scatter(x, y, z)

            ax.plot(x[([0, 1])], y[([0, 1])], z[([0, 1])])
            ax.plot(x[([1, 2])], y[([1, 2])], z[([1, 2])])
            ax.plot(x[([3, 4])], y[([3, 4])], z[([3, 4])])
            ax.plot(x[([4, 5])], y[([4, 5])], z[([4, 5])])
            ax.plot(x[([0, 6])], y[([0, 6])], z[([0, 6])])
            ax.plot(x[([3, 6])], y[([3, 6])], z[([3, 6])])
            ax.plot(x[([6, 7])], y[([6, 7])], z[([6, 7])])
            ax.plot(x[([7, 8])], y[([7, 8])], z[([7, 8])])
            ax.plot(x[([8, 9])], y[([8, 9])], z[([8, 9])])
            ax.plot(x[([7, 10])], y[([7, 10])], z[([7, 10])])
            ax.plot(x[([10, 11])], y[([10, 11])], z[([10, 11])])
            ax.plot(x[([11, 12])], y[([11, 12])], z[([11, 12])])
            ax.plot(x[([7, 13])], y[([7, 13])], z[([7, 13])])
            ax.plot(x[([13, 14])], y[([13, 14])], z[([13, 14])])
            ax.plot(x[([14, 15])], y[([14, 15])], z[([14, 15])])

            # Create cubic bounding box to simulate equal aspect ratio
            max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max()
            Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (x.max() + x.min())
            Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (y.max() + y.min())
            Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (z.max() + z.min())

            for xb, yb, zb in zip(Xb, Yb, Zb):
                ax.plot([xb], [yb], [zb], 'w')

            plt.axis('equal')

        a = anim.FuncAnimation(fig, update, frames=poses.shape[0], repeat=False)
        # plt.show()
        plt.savefig('aa.jpg')

    return

def p_mpjpe(predicted, target, scale):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    # print(predicted.shape)
    # print(target.shape)
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

    m = np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1)

    # a = m * scale[:, 0]
    # Return MPJPE

    return np.mean(m, axis=1)

def p_mpjpe_z(predicted, target, scale):
    """
    Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
    often referred to as "Protocol #2" in many papers.
    """
    # print(predicted.shape)
    # print(target.shape)
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
    assert predicted.shape[2] == 3
    m = np.abs(predicted_aligned[:, :, 2] - target[:, :, 2])
    # print(m.shape)

    # a = m * scale[:, 0]
    # Return MPJPE

    return np.mean(m, axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str,
                        help='Generatorの重みファイルへのパス')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--batchsize', '-b', type=int, default=200)
    parser.add_argument('--allow_inversion', action="store_true",
                         help='評価時にzの反転を許可するかどうか')
    args = parser.parse_args()

    # 学習時のオプションの読み込み
    with open(os.path.join(
            os.path.dirname(args.model_path), 'options.json')) as f:
        opts = json.load(f)

    # モデルの定義
    gen = projection_gan.pose.posenet.MLP(mode='generator',
        use_bn=opts['use_bn'], activate_func=getattr(F, opts['activate_func']))
    serializers.load_npz(args.model_path, gen)
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        gen.to_gpu()

    print()
    # 行動クラスの読み込み
    if opts['action'] == 'all':
        with open(os.path.join('data', 'actions.txt')) as f:
            actions = f.read().split('\n')[:-1]
    else:
        actions = [opts['action']]
    # actions = ['Eating']


    # 各行動クラスに対して平均エラー(mm)を算出
    errors = []
    errorsxy = []
    errorsx = []
    errorsy = []
    errorsz = []
    errorsp = []
    errorspz = []
    for act_name in actions:
        # test = projection_gan.pose.dataset.pose_dataset.H36M(
        #     action=act_name, train=False,
        #     use_sh_detection=opts['use_sh_detection'])
            # opts['use_sh_detection']
        # test = projection_gan.pose.dataset.pose_dataset.MPII(
          # train=False, use_sh_detection=False)
        # test = projection_gan.pose.dataset.mpii_inf_3dhp_dataset.MPII3DDataset(
        #     annotations_glob="/mnt/dataset/MPII_INF_3DHP/mpi_inf_3dhp/*/*/annot.mat", train=False)
        test = projection_gan.pose.dataset.pose_dataset.LSP(
            train=False, use_sh_detection=opts['use_sh_detection'])
        test_iter = iterators.MultiprocessIterator(
            test, args.batchsize, repeat=False, shuffle=False)
        eds = []
        xs = []
        ys = []
        zs = []
        xys = []
        p_m = []
        p_z = []
        num = 0
        for batch in test_iter:

            xy_proj, xyz, scale, c = dataset.concat_examples(
                batch, device=args.gpu)
            batchsize = len(batch)
            # print(batchsize)
            # print(xy_proj.shape)
            num += batchsize

            # print(xy_proj.shape, xyz.shape)
            # aa
            # xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
            # print(xy_proj.shape)
            # xyz = xyz[:, 0]
            # print()
            with chainer.no_backprop_mode(), \
                    chainer.using_config('train', False):
                # xy_proj[:, 14] = 0
                # xy_proj[:, 15] = 0
                # xy_proj[:, 18] = 0
                # xy_proj[:, 19] = 0
                xy_real = chainer.Variable(xy_proj)
                print(xy_real.shape)

                c = Variable(c)
                Cs = F.expand_dims(c, axis=1)
                Cs = F.repeat(Cs, 17, axis=1)
                d_pred, cam_pred = gen(xy_real)  # d batch, 16
                # print(cam_pred[0])

                d0 = Variable(cupy.array([[0.0]], np.float32))
                d0 = F.repeat(d0, batchsize, axis=0)
                d_pred = F.concat([d0, d_pred], axis=1)
                z_pred = d_pred + Cs  # batch, 16
                # z_pred = d_pred

                x_pred = z_pred * xy_real[:, 0::2]  # xy_real[:, 0::2] * xyz_real[:, 2::3]#z_pred * xy_real[:, 0::2]
                y_pred = z_pred * xy_real[:, 1::2]  # xy_real[:, 1::2] * xyz_real[:, 2::3]#z_pred * xy_real[:, 1::2]
                # z_pred = XY_pred[:, 2::3] + Cs

                xyz_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None], z_pred[:, :, None]),
                                    axis=2)  # batchsize 17 3
                xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]

                xyz_real = cupy.asnumpy(xyz.reshape(-1, 17, 3)).transpose(0, 2, 1)

                # Random rotation.
                theta = gen.xp.random.uniform(0.7*np.pi, 0.7*np.pi, batchsize).astype('f')
                cos_theta = gen.xp.cos(theta)[:, None]
                sin_theta = gen.xp.sin(theta)[:, None]

                rot_x = x_pred * cos_theta + d_pred * sin_theta
                rot_z = d_pred * cos_theta - x_pred * sin_theta
                rot_y = y_pred
                rot_z += Cs
                xyz_pred = F.concat((rot_x[:, :, None], rot_y[:, :, None], rot_z[:, :, None]), axis=2)  # batchsize 17 3
                xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]
                print(type(xyz_pred.data))
                print(type(xyz_real))

                # for MPII
                if num == 200: # 0, 2199 = 1, 2200     2200, 2250 = 2201, 2251
                    print('ploting...')
                    # print(xy_proj[50])
                    # print(xyz_pred[50])
                    plot17j(xyz_real[59:70], cupy.asnumpy(xyz_pred[59:70].data), False)
                    # plot17j2d(cupy.asnumpy(xy_proj[0:11]), False)
                    print('done.')
                    time.sleep(100)


                # xyz_fake = F.concat((xy_real[:, 0::2, None], xy_real[:, 1::2, None], z_pred[:, 0::1, None]), axis=2).data

            # lx = gen.xp.power(xyz[:, 0::3] - xyz_pred[:, 0].data, 2)
            # ly = gen.xp.power(xyz[:, 1::3] - xyz_pred[:, 1].data, 2)
            # lz = gen.xp.power(xyz[:, 2::3] - xyz_pred[:, 2].data, 2)
            # # 200, 17
            #
            # euclidean_distance = gen.xp.sqrt(lx + ly + lz).mean(axis=1)
            # # 200, 17 -> 200, 1
            # # euclidean_distance = gen.xp.sqrt(lz) # .mean(axis=1)
            # # 200, 17
            # # for i in range(17):
            # #     euclidean_distance[:, i] *= scale[:, 0]
            # euclidean_distance *= scale[:, 0]
            # # print(euclidean_distance.shape)
            # # print(type(euclidean_distance.data))
            # # print(xyd[0])
            # # print(euclidean_distance[0])
            # # aa
            # # euclidean_distance = np.sqrt(euclidean_distance ** 2 + xyd)
            # euclidean_distance = gen.xp.mean(euclidean_distance)
            # xy_distance = gen.xp.sqrt(lx + ly).mean(axis=1)
            # xy_distance *= scale[:, 0]
            # xy_distance = gen.xp.mean(xy_distance)
            # x_distance = gen.xp.sqrt(lx).mean(axis=1)
            # x_distance *= scale[:, 0]
            # x_distance = gen.xp.mean(x_distance)
            # y_distance = gen.xp.sqrt(ly).mean(axis=1)
            # y_distance *= scale[:, 0]
            # y_distance = gen.xp.mean(y_distance)
            # z_distance = gen.xp.sqrt(lz).mean(axis=1)
            # z_distance *= scale[:, 0]
            # z_distance = gen.xp.mean(z_distance)
            #
            # # print(type(xyz))
            # xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1]).data  # [batchsize, 17, 3]
            #
            # P_mpjpe = p_mpjpe(cupy.asnumpy(xyz_pred), cupy.asnumpy(xyz.reshape(-1, 17, 3)), cupy.asnumpy(scale))
            # P_mpjpe = P_mpjpe * cupy.asnumpy(scale[:, 0])
            # P_mpjpe = np.mean(P_mpjpe)
            # # print(P_mpjpe.shape)
            # P_mpjpe_z = p_mpjpe_z(cupy.asnumpy(xyz_pred), cupy.asnumpy(xyz.reshape(-1, 17, 3)), cupy.asnumpy(scale))
            # P_mpjpe_z = P_mpjpe_z * cupy.asnumpy(scale[:, 0])
            # P_mpjpe_z = np.mean(P_mpjpe_z)
            #
            #
            # eds.append(euclidean_distance * len(batch))
            # xys.append(xy_distance * len(batch))
            # xs.append(x_distance * len(batch))
            # ys.append(y_distance * len(batch))
            # zs.append(z_distance * len(batch))
            # p_m.append(P_mpjpe * len(batch))
            # p_z.append(P_mpjpe_z * len(batch))


        test_iter.finalize()
        print(act_name, '\t', sum(eds) / len(test),'\t', sum(xys)/len(test),'\t', sum(xs)/len(test), '\t',sum(ys)/len(test), '\t', sum(zs)/len(test),
              '\t', sum(p_m) / len(test), '\t', sum(p_z) / len(test))
        errors.append(sum(eds) / len(test))
        errorsxy.append(sum(xys) / len(test))
        errorsx.append(sum(xs) / len(test))
        errorsy.append(sum(ys) / len(test))
        errorsz.append(sum(zs) / len(test))
        errorsp.append(sum(p_m) / len(test))
        errorspz.append(sum(p_z) / len(test))
    print('-' * 20)
    print('average', '\t',sum(errors) / len(errors), '\t',sum(errorsxy) / len(errorsxy),
          '\t',sum(errorsx) / len(errorsx), '\t',sum(errorsy) / len(errorsy), '\t',sum(errorsz) / len(errorsz),
          '\t', sum(errorsp) / len(errorsp), '\t', sum(errorspz) / len(errorspz))

    # csvとして保存
    with open(args.model_path.replace('.npz', '.csv'), 'w') as f:
        for act_name, error in zip(actions, errors):
            f.write('{},{}\n'.format(act_name, error))
        f.write('{},{}\n'.format('average', sum(errors) / len(errors)))

