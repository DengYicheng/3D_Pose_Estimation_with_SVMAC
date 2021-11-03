import chainer
import numpy as np


class Normalization(object):
    @staticmethod
    def normalize_3d(pose, c):
        # average distance to hip = 1/c
        # distance from camera to hip = c
        xs = pose.T[0::3] - pose.T[0]
        ys = pose.T[1::3] - pose.T[1]

        # average distance
        ls = np.sqrt(xs[1:] ** 2 + ys[1:] ** 2)
        scale = ls.mean(axis=0)
        pose = pose.T / scale

        # hip as root joint
        pose[0::3] -= pose[0].copy()
        pose[1::3] -= pose[1].copy()
        pose[2::3] -= pose[2].copy() - c

        return pose.T, scale

    @staticmethod
    def normalize_2d(pose, c):
        xs = pose.T[0::2] - pose.T[0]  # translation
        ys = pose.T[1::2] - pose.T[1]  # translation

        pose = pose.T / (np.sqrt(xs[1:] ** 2 + ys[1:] ** 2).mean(axis=0) * c)

        mu_x = pose[0].copy()
        mu_y = pose[1].copy()
        pose[0::2] -= mu_x
        pose[1::2] -= mu_y

        return pose.T


class PoseDatasetBase(chainer.dataset.DatasetMixin):
    def _normalize_3d(self, pose, c):
        return Normalization.normalize_3d(pose, c)

    def _normalize_2d(self, pose, c):
        return Normalization.normalize_2d(pose, c)
