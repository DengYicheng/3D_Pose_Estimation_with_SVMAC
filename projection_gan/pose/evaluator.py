import copy
import chainer
from chainer import function, Variable
import chainer.functions as F
from chainer import reporter as reporter_module
from chainer.training import extensions
import cupy
import numpy as np

class JointsForPCK(object):
    """
    @see Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision
    """
    from_h36m_joints = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

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

    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1))

class Evaluator(extensions.Evaluator):
    def evaluate(self):
        iterator = self._iterators['main']
        gen = self._targets['gen']
        if self.eval_hook:
            self.eval_hook(self)
        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)
        summary = reporter_module.DictSummary()
        PCK_THRESHOLD = 150  # see "Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision"

        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                xy_proj, xyz, scale, c = self.converter(batch, self.device)

                xy_proj, xyz = xy_proj[:, 0], xyz[:, 0]
                batchsize = len(batch)
                with function.no_backprop_mode(), \
                     chainer.using_config('train', False):
                    xy_real = chainer.Variable(xy_proj)

                    c = Variable(c)
                    Cs = F.expand_dims(c, axis=1)
                    Cs = F.repeat(Cs, 17, axis=1)
                    d_pred, cam_pred = gen(xy_real)  # d batch, 16
                    d0 = Variable(cupy.array([[0.0]], np.float32))
                    d0 = F.repeat(d0, batchsize, axis=0)
                    d_pred = F.concat([d0, d_pred], axis=1)
                    d_pred = d_pred + Cs  # batch, 16

                    z_pred = d_pred

                    z_pred1 = z_pred.data[:, JointsForPCK.from_h36m_joints]
                    z_real = xyz[:, 2::3][:, JointsForPCK.from_h36m_joints]

                    per_joint_error = gen.xp.sqrt((z_pred1 - z_real) * (z_pred1 - z_real)) * scale.reshape(
                        (-1, 1, 1, 1))
                    true_positive = (per_joint_error < PCK_THRESHOLD).sum()
                    total = per_joint_error.size

                    PCK = float(true_positive) / total * 100
                    chainer.report({'PCK3D': PCK}, gen)

                    x_pred = z_pred * xy_real[:, 0::2]
                    y_pred = z_pred * xy_real[:, 1::2]

                    xyz_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None], z_pred[:, :, None]),
                                        axis=2)  # batchsize 17 3
                    xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]

                    z_mse = F.mean_squared_error(xyz_pred[:, 2], xyz[:, 2::3])

                    chainer.report({'z_mse': z_mse}, gen)

                    lx = gen.xp.power(xyz[:, 0::3] - xyz_pred[:, 0].data, 2)
                    ly = gen.xp.power(xyz[:, 1::3] - xyz_pred[:, 1].data, 2)
                    lz = gen.xp.power(xyz[:, 2::3] - xyz_pred[:, 2].data, 2)
                    euclidean_distance = gen.xp.sqrt(lx + ly + lz).mean(axis=1)
                    euclidean_distance *= scale[:, 0]
                    euclidean_distance = gen.xp.mean(euclidean_distance)
                    chainer.report(
                        {'mpjpe': euclidean_distance}, gen)


                    xyz_pred1 = F.transpose(xyz_pred, axes=[0, 2, 1]).data  # [batchsize, 17, 3]
                    xyz_mse = F.mean_squared_error(F.reshape(xyz_pred1,(-1, 51)), xyz)
                    chainer.report({'xyz': xyz_mse}, gen)

                    # calculate p-mpjpe
                    p_mp = p_mpjpe(cupy.asnumpy(xyz_pred1), cupy.asnumpy(xyz.reshape(-1, 17, 3)))
                    p_mp = p_mp * cupy.asnumpy(scale[:, 0])
                    p_mp = np.mean(p_mp)
                    chainer.report({'p_mp': p_mp}, gen)

                    # calculate xy and z respectively
                    xy_distance = gen.xp.sqrt(lx + ly).mean(axis=1)
                    xy_distance *= scale[:, 0]
                    xy_distance = gen.xp.mean(xy_distance)
                    z_distance = gen.xp.sqrt(lz).mean(axis=1)
                    z_distance *= scale[:, 0]
                    z_distance = gen.xp.mean(z_distance)

                    chainer.report(
                       {'xy': xy_distance}, gen)
                    chainer.report(
                       {'z': z_distance}, gen)

        
            summary.add(observation)
        return summary.compute_mean()


