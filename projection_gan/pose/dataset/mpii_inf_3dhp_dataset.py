import glob

import numpy
import scipy.io
import collections
import typing
import h5py
from . import pose_dataset_base


class H36CompatibleJoints(object):
    joint_names = ['spine3', 'spine4', 'spine2', 'spine', 'pelvis',  # 4
                   'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow',  # 10
                   'left_wrist', 'left_hand', 'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist',  # 16
                   'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',  # 22
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe']  # 27
    joint_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 7, 6, 9, 10, 11, 14, 15, 16]
    # joint_idx4test = [14, 8, 9, 10, 11, 12, 13, 15, 1, 0, 16, 5, 6, 7, 2, 3, 4]
    # joint_idx = [4, 23, 24, 25, 18, 19, 20, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16] # 正确的位置
    joint_idx4test = [14, 8, 9, 10, 11, 12, 13, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]  # 正确的位置

    @staticmethod
    def convert_points(raw_vector):
        return numpy.array(
            [(int(raw_vector[i * 2]), int(raw_vector[i * 2 + 1])) for i in H36CompatibleJoints.joint_idx])

    @staticmethod
    def convert_points_3d(raw_vector):
        return numpy.array([
            (float(raw_vector[i * 3]), float(raw_vector[i * 3 + 1]), float(raw_vector[i * 3 + 2])) for i in
            H36CompatibleJoints.joint_idx])

    @staticmethod
    def convert_points4test(raw_vector):
        return numpy.array(
            [(int(raw_vector[i * 2]), int(raw_vector[i * 2 + 1])) for i in H36CompatibleJoints.joint_idx4test])

    @staticmethod
    def convert_points_3d4test(raw_vector):
        return numpy.array([
            (float(raw_vector[i * 3]), float(raw_vector[i * 3 + 1]), float(raw_vector[i * 3 + 2])) for i in
            H36CompatibleJoints.joint_idx4test])


class MPII3DDatasetUtil(object):
    mm3d_chest_cameras = [
        0, 2, 4, 7, 8
    ]  # Subset of chest high, used in "Monocular 3D Human Pose Estimation in-the-wild Using Improved CNN supervision"

    @staticmethod
    def read_cameraparam(path):
        params = collections.defaultdict(dict)
        index = 0
        for line in open(path):
            key = line.split()[0].strip()
            if key == "name":
                value = line.split()[1].strip()
                index = int(value)
            if key == "intrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = numpy.array(values).reshape((4, 4))
                params[index]["intrinsic"] = values
            if key == "extrinsic":
                values = line.split()[1:]
                values = [float(value) for value in values]
                values = numpy.array(values).reshape((4, 4))
                params[index]["extrinsic"] = values
        return params


MPII3DDatum = typing.NamedTuple('MPII3DDatum', [
    ('annotation_2d', numpy.ndarray),
    ('annotation_3d', numpy.ndarray),
    ('normalized_annotation_2d', numpy.ndarray),
    ('normalized_annotation_3d', numpy.ndarray),
    ('normalize_3d_scale', float),
])


class MPII3DDataset(pose_dataset_base.PoseDatasetBase):
    def __init__(self, annotations_glob="data/mpi-inf-3dhp/*/*/annot.mat", train=True, c=10):
        self.c = 10
        self.dataset = []
        if train == True:
            for annotation_path in glob.glob(annotations_glob):
                # print("load ", annotation_path)
                annotation = scipy.io.loadmat(annotation_path)
                for camera in MPII3DDatasetUtil.mm3d_chest_cameras:
                    for frame in range(len(annotation["annot2"][camera][0])):
                        annot_2d = H36CompatibleJoints.convert_points(annotation["annot2"][camera][0][frame])
                        annot_3d = H36CompatibleJoints.convert_points_3d(annotation["annot3"][camera][0][frame])
                        annot_3d_normalized, scale = self._normalize_3d(
                            annot_3d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 3), c)
                        self.dataset.append(MPII3DDatum(
                            annotation_2d=annot_2d,
                            annotation_3d=annot_3d,
                            normalized_annotation_2d=self._normalize_2d(
                                annot_2d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 2), c),
                            normalized_annotation_3d=annot_3d_normalized,
                            normalize_3d_scale=scale,
                        ))
        else:
            annotations_glob = "data/mpii3d-test/*/annot_data.mat"
            for annotation_path in glob.glob(annotations_glob):
                annotation = h5py.File(annotation_path)
                for camera in range(annotation["annot2"].shape[0]):
                    if annotation["valid_frame"][camera] == 1:

                        annot_2d = H36CompatibleJoints.convert_points4test(
                            annotation["annot2"][camera].reshape(-1, 34)[0])
                        # annot_3d = annotation["annot3"][camera].reshape(-1, 51)
                        annot_3d = H36CompatibleJoints.convert_points_3d4test(
                            annotation["annot3"][camera].reshape(-1, 51)[0])

                        annot_3d_normalized, scale = self._normalize_3d(
                            annot_3d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 3), c)
                        self.dataset.append(MPII3DDatum(
                            annotation_2d=annot_2d,
                            annotation_3d=annot_3d,
                            normalized_annotation_2d=self._normalize_2d(
                                annot_2d.reshape(-1, len(H36CompatibleJoints.joint_idx) * 2), c),
                            normalized_annotation_3d=annot_3d_normalized,
                            normalize_3d_scale=scale,
                        ))

    def __len__(self):
        return len(self.dataset)

    def get_example(self, i):
        c = numpy.float32(self.c)
        return self.dataset[i].normalized_annotation_2d.astype(numpy.float32), \
               self.dataset[i].normalized_annotation_3d.astype(numpy.float32), \
               self.dataset[i].normalize_3d_scale.astype(numpy.float32), c
