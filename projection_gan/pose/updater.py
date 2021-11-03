from __future__ import print_function


import chainer
import chainer.functions as F
from chainer import Variable
import cupy
import numpy as np

class H36M_Updater(chainer.training.StandardUpdater):

    def __init__(self, gan_accuracy_cap, use_heuristic_loss,  frame_interval, frame_per,
                 heuristic_loss_weight, mode, *args, **kwargs):
        if not mode in ['supervised', 'unsupervised']:
            raise ValueError("only 'supervised' and 'unsupervised' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        self.gan_accuracy_cap = gan_accuracy_cap
        self.use_heuristic_loss = use_heuristic_loss
        self.heuristic_loss_weight = heuristic_loss_weight
        self.mode = mode
        self.frame_interval = frame_interval
        self.frame_per = frame_per
        super(H36M_Updater, self).__init__(*args, **kwargs)

    @staticmethod
    def calculate_rotation(xy_real, z_pred):
        xy_split = F.split_axis(xy_real, xy_real.data.shape[1], axis=1)
        z_split = F.split_axis(z_pred, z_pred.data.shape[1], axis=1)
        # Vector v0 (neck -> nose) on zx-plain. v0=(a0, b0).
        a0 = z_split[9] - z_split[8]
        b0 = xy_split[9 * 2] - xy_split[8 * 2]
        n0 = F.sqrt(a0 * a0 + b0 * b0)
        # Vector v1 (left shoulder -> right shoulder) on zx-plain. v1=(a1, b1).
        a1 = z_split[14] - z_split[11]
        b1 = xy_split[14 * 2] - xy_split[11 * 2]
        n1 = F.sqrt(a1 * a1 + b1 * b1)
        # Return sine value of the angle between v0 and v1.
        return (a0 * b1 - a1 * b0) / (n0 * n1)

    @staticmethod
    def gradient_penalty(dis, xy_real, xy_fake, batchsize):
        # WGAN-GP
        T = dis.xp.random.uniform(0., 1., (batchsize, 1))
        mid = xy_real * T + xy_fake * (1 - T)

        grad, = chainer.grad([dis(mid)], [mid],
                             enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = 10 * F.mean_squared_error(grad,
                                                  dis.xp.ones_like(grad.data))
        return loss_grad


    @staticmethod
    def calculate_cam_loss(cam_pred, batchsize):
        # loss for weak perspective camera
        CAM = F.matmul(cam_pred, F.transpose(cam_pred, axes=[0, 2, 1]))
        cam_trace = CAM[:, 0, 0] + CAM[:, 1, 1]
        eye = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        eye = Variable(cupy.array(eye))
        eye = F.expand_dims(eye, axis = 0)
        eye = F.repeat(eye, batchsize, axis=0)
        loss_mat = F.mean_squared_error(F.reshape((2 / cam_trace), (-1, 1, 1)) * CAM, eye)

        return loss_mat

    @staticmethod
    def calculate_heuristic_loss(xy_real, z_pred):
        return F.average(F.relu(
            -H36M_Updater.calculate_rotation(xy_real, z_pred)))

    @staticmethod
    def cal_symm_loss(pose):
        l_shoulder = pose[:, :, 8] - pose[:, :, 11] # b, 3
        r_shoulder = pose[:, :, 8] - pose[:, :, 14] # b, 3
        shoulder_symm = F.absolute_error(F.matmul(F.expand_dims(l_shoulder, axis=1), l_shoulder[:, :, None]),
                                         F.matmul(F.expand_dims(r_shoulder, axis=1), r_shoulder[:, :, None])) # b, 1, 1

        l_elbow = pose[:, :, 11] - pose[:, :, 12]
        r_elbow = pose[:, :, 14] - pose[:, :, 15]
        elbow_symm = F.absolute_error(F.matmul(F.expand_dims(l_elbow, axis=1), l_elbow[:, :, None]),
                                      F.matmul(F.expand_dims(r_elbow, axis=1), r_elbow[:, :, None]))

        l_wrist = pose[:, :, 12] - pose[:, :, 13]
        r_wrist = pose[:, :, 15] - pose[:, :, 16]
        wrist_symm = F.absolute_error(F.matmul(F.expand_dims(l_wrist, axis=1), l_wrist[:, :, None]),
                                      F.matmul(F.expand_dims(r_wrist, axis=1), r_wrist[:, :, None]))

        l_hip = pose[:, :, 0] - pose[:, :, 4]
        r_hip = pose[:, :, 0] - pose[:, :, 1]
        hip_symm = F.absolute_error(F.matmul(F.expand_dims(l_hip, axis=1), l_hip[:, :, None]),
                                    F.matmul(F.expand_dims(r_hip, axis=1), r_hip[:, :, None]))

        l_knee = pose[:, :, 4] - pose[:, :, 5]
        r_knee = pose[:, :, 1] - pose[:, :, 2]
        knee_symm = F.absolute_error(F.matmul(F.expand_dims(l_knee, axis=1), l_knee[:, :, None]),
                                     F.matmul(F.expand_dims(r_knee, axis=1), r_knee[:, :, None]))

        l_foot = pose[:, :, 5] - pose[:, :, 6]
        r_foot = pose[:, :, 2] - pose[:, :, 3]
        foot_symm = F.absolute_error(F.matmul(F.expand_dims(l_foot, axis=1), l_foot[:, :, None]),
                                     F.matmul(F.expand_dims(r_foot, axis=1), r_foot[:, :, None]))

        symm_loss = F.average(shoulder_symm + elbow_symm + wrist_symm + hip_symm + knee_symm + foot_symm)

        return symm_loss

    @staticmethod
    def cal_KCS(X, batchsize, C):
        Matrix = F.matmul(X, C)  # batchsize 3 16
        M_T = F.transpose(Matrix, axes=[0, 2, 1])  # batchsize 16 3
        KCS = F.matmul(M_T, Matrix)  # batchsize 16 16
        return KCS

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')
        dis_optimizer = self.get_optimizer('dis')

        gen, dis = gen_optimizer.target, dis_optimizer.target

        batch = self.get_iterator('main').next()
        batchsize = len(batch)

        xy_proj, xyz, scale, c = self.converter(batch, self.device) # [batchsize, frame_num, 3 or 2, 17]
        c = Variable(c)


        weights = np.array([[1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 0, 1, 0.1, 0.1, 1, 1, 1, 1, 1, 1]], dtype=np.float32)  # 2, 17
        weights = Variable(cupy.array(weights.T))  # 17, 2
        weights = F.reshape(weights, (-1, 34))


        # calculate wKCS matrix
        C = np.array([[1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [1, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, -1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1]], dtype=np.float32)
        C = C.T
        C = Variable(cupy.array(C))
        C = F.expand_dims(C, axis=0)
        C = F.repeat(C, batchsize, axis=0)
        KCS_diff = np.array([[0, 1, 2, 1, 2, 3, 1, 2, 3, 4, 3, 4, 5, 3, 4, 5],
                             [1, 0, 1, 2, 3, 4, 2, 3, 4, 5, 4, 5, 6, 4, 5, 6],
                             [2, 1, 0, 3, 4, 5, 3, 4, 5, 6, 5, 6, 7, 5, 6, 7],
                             [1, 2, 3, 0, 1, 2, 1, 2, 3, 4, 3, 4, 5, 3, 4, 5],
                             [2, 3, 4, 1, 0, 1, 2, 3, 4, 5, 4, 5, 6, 4, 5, 6],
                             [3, 4, 5, 2, 1, 0, 3, 4, 5, 6, 5, 6, 7, 5, 6, 7],
                             [1, 2, 3, 1, 2, 3, 0, 1, 2, 3, 2, 3, 4, 2, 3, 4],
                             [2, 3, 4, 2, 3, 4, 1, 0, 1, 2, 1, 2, 3, 1, 2, 3],
                             [3, 4, 5, 3, 4, 5, 2, 1, 0, 1, 1, 2, 3, 1, 2, 3],
                             [4, 5, 6, 4, 5, 6, 3, 2, 1, 0, 2, 3, 4, 2, 3, 4],
                             [3, 4, 5, 3, 4, 5, 2, 1, 1, 2, 0, 1, 2, 1, 2, 3],
                             [4, 5, 6, 4, 5, 6, 3, 2, 2, 3, 1, 0, 1, 2, 3, 4],
                             [5, 6, 7, 5, 6, 7, 4, 3, 3, 4, 2, 1, 0, 3, 4, 5],
                             [3, 4, 5, 3, 4, 5, 2, 1, 1, 2, 1, 2, 3, 0, 1, 2],
                             [4, 5, 6, 4, 5, 6, 3, 2, 2, 3, 2, 3, 4, 1, 0, 1],
                             [5, 6, 7, 5, 6, 7, 4, 3, 3, 4, 3, 4, 5, 2, 1, 0]], dtype=np.float32)
        KCS_portion = np.array([1, 0.76, 0.46, 0.32, 0.24, 0.20, 0.17], dtype=np.float32)
        KCS_weight = np.zeros((16, 16), dtype=np.float32)

        for i in range(16):
            for j in range(16):
                if KCS_diff[i, j] == -1:
                    KCS_weight[i, j] = 1.50
                elif KCS_diff[i, j] == 0.0:
                    KCS_weight[i, j] = 1
                else:
                    KCS_weight[i, j] = KCS_portion[(np.int)(KCS_diff[i, j]) - 1]

        KCS_weight = Variable(cupy.array(KCS_weight))
        KCS_weight = F.expand_dims(KCS_weight, axis=0)
        KCS_weight = F.repeat(KCS_weight, batchsize, axis=0)


        # for 0, 10, 20...   1, 11, 21...
        for i in range(self.frame_interval):
            # 0, 2, 4...for training，1, 3, 5...for discriminating
            train_xy_proj = xy_proj[:, i::2*self.frame_interval] # fetch 1 frame per 2*interval
            train_xyz = xyz[:, i::2*self.frame_interval]
            dis_xy_proj = xy_proj[:, i+self.frame_interval::2*self.frame_interval]
            dis_xyz = xyz[:, i+self.frame_interval::2*self.frame_interval]
            loss_gen = 0
            loss_dis = 0
            cam_preds = []
            xyz_preds = []
            wKCS_preds = []
            loss_pred = 0
            loss_cam = 0
            loss_cam_eq = 0
            loss_wKCS = 0

            for j in range(int(self.frame_per/2)): # frame_per/2 images for training
                xy_real = Variable(train_xy_proj[:, j]) # batch, i
                xy_dis_real = Variable(dis_xy_proj[:, j])
                xyz_real = train_xyz[:, j]

                # estimate d and camera
                d_pred, cam_pred = gen(xy_real)

                d0 = Variable(cupy.array([[0.0]], np.float32))
                d0 = F.repeat(d0, batchsize, axis=0)
                d_pred = F.concat([d0, d_pred], axis=1)
				
                Cs = F.expand_dims(c, axis=1)
                Cs = F.repeat(Cs, 17, axis=1)
                z_pred = d_pred + Cs  # batch, 17

                x_pred = xy_real[:, 0::2] * z_pred  # batch, 17
                y_pred = xy_real[:, 1::2] * z_pred  # batch, 17


                xyz_real = F.concat((xyz_real[:, 0::3, None], xyz_real[:, 1::3, None], xyz_real[:, 2::3, None]),
                                    axis=2)  # batchsize 17 3
                xyz_real = F.transpose(xyz_real, axes=[0, 2, 1])  # batchsize 3 17

                xyz_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None], z_pred[:, :, None]),
                                    axis=2) # batchsize 17 3
                xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1]) # [batchsize, 3, 17]

                # for calculation of Langle
                xy_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None]), axis=2)  # batchsize, 17, 2
                xy_pred = F.reshape(xy_pred, (batchsize, -1))  # xyxyxy

                # mse for show
                z_mse = F.mean_squared_error(z_pred, xyz_real[:, 2])
                xyz_mse = F.mean_squared_error(xyz_pred, xyz_real)

                xyz_preds.append(xyz_pred)

                # reshape camera to 2*3
                cam_pred = F.reshape(cam_pred, (batchsize, 2, 3))  # bts, 2, 3

                # Random rotation
                theta = gen.xp.random.uniform(0, 2 * np.pi, batchsize).astype('f')
                cos_theta = gen.xp.cos(theta)[:, None]
                sin_theta = gen.xp.sin(theta)[:, None]

                rot_x = x_pred * cos_theta + d_pred * sin_theta
                rot_z = d_pred * cos_theta - x_pred * sin_theta
                rot_y = y_pred

                rot_z += Cs

                rot_xyz = F.concat((rot_x[:, :, None], rot_y[:, :, None], rot_z[:, :, None]), axis=2)  # batchsize 17 3
                rot_xyz = F.transpose(rot_xyz, axes=[0, 2, 1])  # batchsize 3 17
                rot_x_1 = rot_x / rot_z
                rot_y_1 = rot_y / rot_z
                rot_cam_xy = F.reshape(F.transpose(F.matmul(cam_pred, rot_xyz), axes=[0, 2, 1]),
                                       (batchsize, -1))  # bts, 34
                rot_xy = F.concat((rot_x_1[:, :, None], rot_y_1[:, :, None]), axis=2)  # batchsize 17 2
                rot_xy = F.reshape(rot_xy, (-1, 34))  # batchsize


                # the second estimation
                rot_d_pred, rot_cam_pred = gen(rot_cam_xy)

                d00 = Variable(cupy.array([[0.0]], np.float32))
                d00 = F.repeat(d00, batchsize, axis=0)
                rot_d_pred = F.concat([d00, rot_d_pred], axis=1)

                rot_z_pred = rot_d_pred + Cs  # batch, 17
                rot_x_pred = rot_cam_xy[:, 0::2] * rot_z_pred  # batch, 17
                rot_y_pred = rot_cam_xy[:, 1::2] * rot_z_pred  # batch, 17

                rot_xyz_pred = F.concat(
                    (rot_x_pred[:, :, None], rot_y_pred[:, :, None], rot_z_pred[:, :, None]),
                    axis=2)  # batchsize 17 3
                rot_xyz_pred = F.transpose(rot_xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]

                rot_cam_pred = F.reshape(rot_cam_pred, (batchsize, 2, 3))  # bts, 2, 3

                cam_preds.append(cam_pred)  # bts, 2, 3

                # if video
                if j != 0: loss_cam_eq += F.sum(abs(cam_preds[j] - cam_preds[j - 1])) / batchsize
                if j != 0: loss_pred += F.mean_squared_error(xyz_preds[j], xyz_preds[j - 1])
                if j != 0:
                    wKCS_preds.append(self.cal_KCS(xyz_pred, C=C, W=KCS_weight, batchsize=batchsize))
                    loss_wKCS += F.average(abs(wKCS_preds[j] - wKCS_preds[j-1]))

                # discrimination
                y_real = dis(xy_dis_real)
                y_fake = dis(rot_cam_xy)

                loss_gp = self.gradient_penalty(dis=dis, xy_real=xy_dis_real, xy_fake=rot_cam_xy, batchsize=batchsize)
                loss_gen += F.average(-y_fake)# + F.average(-dd_y_fake)
                loss_dis += F.average(y_fake) + F.average(-y_real) + loss_gp

                # multiple 2D reprojections
                rep_1_1 = F.reshape(F.transpose(F.matmul(cam_pred, xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_1_2 = F.reshape(F.transpose(F.matmul(rot_cam_pred, xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_2_1 = F.reshape(F.transpose(F.matmul(cam_pred, rot_xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_2_2 = F.reshape(F.transpose(F.matmul(rot_cam_pred, rot_xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34

                # simplified SVMAC loss
                loss_rep1 = F.mean_squared_error(weights*rep_1_1, weights*xy_real)*2 + \
                            F.mean_squared_error(weights*rep_1_2, weights*xy_real)*2 + \
                            F.mean_squared_error(weights*rep_1_1, weights*rep_1_2)
                loss_rep2 = F.mean_squared_error(weights*rep_2_1, weights*rep_2_2) + \
                            F.mean_squared_error(weights*rot_cam_xy, weights*rot_xy)
                loss_rep = loss_rep1 * 10 / 3 + loss_rep2 * 10/2

                # Langle
                loss_heuristic = self.calculate_heuristic_loss(xy_real=xy_pred, z_pred=z_pred)
                # Lsymm
                loss_symm = self.cal_symm_loss(xyz_pred)
                # L 3d equal
                loss_3d = F.mean_squared_error(rot_xyz_pred, rot_xyz)
                # Lcam
                loss_cam_1 = self.calculate_cam_loss(cam_pred, batchsize)
                loss_cam_2 = self.calculate_cam_loss(rot_cam_pred, batchsize)
                loss_cam_eq_f = F.average(abs(cam_pred - rot_cam_pred))
                loss_cam += loss_cam_1 * 1 + loss_cam_2 * 1 + loss_cam_eq_f * 1

                loss_gen += loss_cam + loss_rep * 1 + loss_heuristic * 1 + loss_3d * 0.1 + loss_symm * 0.01  # + loss_heuristic * 0.01

            loss_gen += loss_cam_eq

            gen.cleargrads()
            loss_gen.backward()
            gen_optimizer.update()

            rot_cam_xy.unchain_backward()

            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()



            # 1, 3, 5...for training，0, 2, 4...for discrimination
            train_xy_proj = xy_proj[:, i+self.frame_interval::2 * self.frame_interval]
            train_xyz = xyz[:, i+self.frame_interval::2 * self.frame_interval]
            dis_xy_proj = xy_proj[:, i::2 * self.frame_interval]
            dis_xyz = xyz[:, i::2 * self.frame_interval]
            loss_gen = 0
            loss_dis = 0
            cam_preds = []
            xyz_preds = []
            wKCS_preds = []
            loss_pred = 0
            loss_cam = 0
            loss_cam_eq = 0
            loss_rep = 0
            loss_wKCS = 0

            for j in range(int(self.frame_per / 2)):  # frame_per/2 images for training
                xy_real = Variable(train_xy_proj[:, j])  # batch, i
                xy_dis_real = Variable(dis_xy_proj[:, j])
                xyz_real = train_xyz[:, j]

                # estimate d and camera
                d_pred, cam_pred = gen(xy_real)

                d0 = Variable(cupy.array([[0.0]], np.float32))
                d0 = F.repeat(d0, batchsize, axis=0)
                d_pred = F.concat([d0, d_pred], axis=1)

                Cs = F.expand_dims(c, axis=1)
                Cs = F.repeat(Cs, 17, axis=1)
                z_pred = d_pred + Cs  # batch, 17

                x_pred = xy_real[:, 0::2] * z_pred  # batch, 17
                y_pred = xy_real[:, 1::2] * z_pred  # batch, 17

                xyz_real = F.concat((xyz_real[:, 0::3, None], xyz_real[:, 1::3, None], xyz_real[:, 2::3, None]),
                                    axis=2)  # batchsize 17 3
                xyz_real = F.transpose(xyz_real, axes=[0, 2, 1])  # batchsize 3 17

                xyz_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None], z_pred[:, :, None]),
                                    axis=2)  # batchsize 17 3
                xyz_pred = F.transpose(xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]

                # for calculation of Langle
                xy_pred = F.concat((x_pred[:, :, None], y_pred[:, :, None]), axis=2)  # batchsize, 17, 2
                xy_pred = F.reshape(xy_pred, (batchsize, -1))  # xyxyxy

                # mse for show
                z_mse = F.mean_squared_error(z_pred, xyz_real[:, 2])
                xyz_mse = F.mean_squared_error(xyz_pred, xyz_real)

                xyz_preds.append(xyz_pred)

                # reshape camera to 2*3
                cam_pred = F.reshape(cam_pred, (batchsize, 2, 3))  # bts, 2, 3

                # Random rotation
                theta = gen.xp.random.uniform(0, 2 * np.pi, batchsize).astype('f')
                cos_theta = gen.xp.cos(theta)[:, None]
                sin_theta = gen.xp.sin(theta)[:, None]

                rot_x = x_pred * cos_theta + d_pred * sin_theta
                rot_z = d_pred * cos_theta - x_pred * sin_theta
                rot_y = y_pred

                rot_z += Cs

                rot_xyz = F.concat((rot_x[:, :, None], rot_y[:, :, None], rot_z[:, :, None]), axis=2)  # batchsize 17 3
                rot_xyz = F.transpose(rot_xyz, axes=[0, 2, 1])  # batchsize 3 17
                rot_x_1 = rot_x / rot_z
                rot_y_1 = rot_y / rot_z
                rot_cam_xy = F.reshape(F.transpose(F.matmul(cam_pred, rot_xyz), axes=[0, 2, 1]),
                                       (batchsize, -1))  # bts, 34
                rot_xy = F.concat((rot_x_1[:, :, None], rot_y_1[:, :, None]), axis=2)  # batchsize 17 2
                rot_xy = F.reshape(rot_xy, (-1, 34))  # batchsize

                # the second estimation
                rot_d_pred, rot_cam_pred = gen(rot_cam_xy)

                d00 = Variable(cupy.array([[0.0]], np.float32))
                d00 = F.repeat(d00, batchsize, axis=0)
                rot_d_pred = F.concat([d00, rot_d_pred], axis=1)

                rot_z_pred = rot_d_pred + Cs  # batch, 17
                rot_x_pred = rot_cam_xy[:, 0::2] * rot_z_pred  # batch, 17
                rot_y_pred = rot_cam_xy[:, 1::2] * rot_z_pred  # batch, 17

                rot_xyz_pred = F.concat(
                    (rot_x_pred[:, :, None], rot_y_pred[:, :, None], rot_z_pred[:, :, None]),
                    axis=2)  # batchsize 17 3
                rot_xyz_pred = F.transpose(rot_xyz_pred, axes=[0, 2, 1])  # [batchsize, 3, 17]

                rot_cam_pred = F.reshape(rot_cam_pred, (batchsize, 2, 3))  # bts, 2, 3

                cam_preds.append(cam_pred)  # bts, 2, 3

                # if video
                if j != 0: loss_cam_eq += F.sum(abs(cam_preds[j] - cam_preds[j - 1])) / batchsize
                if j != 0: loss_pred += F.mean_squared_error(xyz_preds[j], xyz_preds[j - 1])
                if j != 0:
                    wKCS_preds.append(self.cal_KCS(xyz_pred, C=C, W=KCS_weight, batchsize=batchsize))
                    loss_wKCS += F.average(abs(wKCS_preds[j] - wKCS_preds[j - 1]))

                # discrimination
                y_real = dis(xy_dis_real)
                y_fake = dis(rot_cam_xy)

                loss_gp = self.gradient_penalty(dis=dis, xy_real=xy_dis_real, xy_fake=rot_cam_xy, batchsize=batchsize)
                loss_gen += F.average(-y_fake)  # + F.average(-dd_y_fake)
                loss_dis += F.average(y_fake) + F.average(-y_real) + loss_gp

                # multiple 2D reprojections
                rep_1_1 = F.reshape(F.transpose(F.matmul(cam_pred, xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_1_2 = F.reshape(F.transpose(F.matmul(rot_cam_pred, xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_2_1 = F.reshape(F.transpose(F.matmul(cam_pred, rot_xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34
                rep_2_2 = F.reshape(F.transpose(F.matmul(rot_cam_pred, rot_xyz_pred), axes=[0, 2, 1]),
                                    (batchsize, -1))  # bts, 34

                # simplified SVMAC loss
                loss_rep1 = F.mean_squared_error(weights * rep_1_1, weights * xy_real) * 2 + \
                            F.mean_squared_error(weights * rep_1_2, weights * xy_real) * 2 + \
                            F.mean_squared_error(weights * rep_1_1, weights * rep_1_2)
                loss_rep2 = F.mean_squared_error(weights * rep_2_1, weights * rep_2_2) + \
                            F.mean_squared_error(weights * rot_cam_xy, weights * rot_xy)
                loss_rep = loss_rep1 * 10 / 3 + loss_rep2 * 10 / 2

                # Langle
                loss_heuristic = self.calculate_heuristic_loss(xy_real=xy_pred, z_pred=z_pred)
                # Lsymm
                loss_symm = self.cal_symm_loss(xyz_pred)
                # L 3d equal
                loss_3d = F.mean_squared_error(rot_xyz_pred, rot_xyz)
                # Lcam
                loss_cam_1 = self.calculate_cam_loss(cam_pred, batchsize)
                loss_cam_2 = self.calculate_cam_loss(rot_cam_pred, batchsize)
                loss_cam_eq_f = F.average(abs(cam_pred - rot_cam_pred))
                loss_cam += loss_cam_1 * 1 + loss_cam_2 * 1 + loss_cam_eq_f * 1

                loss_gen += loss_cam + loss_rep * 1 + loss_heuristic * 1 + loss_3d * 0.1 + loss_symm * 0.01  # + loss_heuristic * 0.01

            loss_gen += loss_cam_eq

            gen.cleargrads()
            loss_gen.backward()
            gen_optimizer.update()

            rot_cam_xy.unchain_backward()

            dis.cleargrads()
            loss_dis.backward()
            dis_optimizer.update()


            chainer.report({'loss': xyz_mse, 'z_mse': z_mse, 'loss_pred': 0, 'loss_heu':loss_heuristic, 'loss_symm':loss_symm,
                            'loss_cam': loss_cam, 'loss_3d': loss_3d, 'loss_rep': loss_rep}, gen)

            chainer.report({
                'loss': loss_dis, 'real': F.sum(y_real) / batchsize, 'fake': F.sum(y_fake) / batchsize, 'loss_gp': loss_gp}, dis)
