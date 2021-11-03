# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L



class Discriminator(chainer.Chain):

    def __init__(self, n_in=34, n_unit=1024, mode='discriminator',
                 use_bn=False, activate_func=F.leaky_relu):
        if not mode == 'discriminator':
            raise ValueError("only 'discriminator' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        super(Discriminator, self).__init__()
        n_out = 1
        print('MODEL: {}, N_OUT: {}, N_UNIT: {}'.format(mode, n_out, n_unit))
        self.mode = mode
        self.use_bn = False
        self.activate_func = activate_func
        self.dropout = 0.0
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_unit, initialW=w)
            self.l2 = L.Linear(n_unit, n_unit, initialW=w)
            self.l3 = L.Linear(n_unit, n_unit, initialW=w)
            self.l4 = L.Linear(n_unit, n_unit, initialW=w)
            self.l5 = L.Linear(n_unit, n_unit, initialW=w)
            self.l6 = L.Linear(n_unit, n_unit, initialW=w)
            self.l7 = L.Linear(n_unit, n_unit, initialW=w)
            self.l8 = L.Linear(n_unit, n_out, initialW=w)

            if self.use_bn:
                self.bn1 = L.BatchNormalization(n_unit)
                self.bn2 = L.BatchNormalization(n_unit)
                self.bn3 = L.BatchNormalization(n_unit)
                self.bn4 = L.BatchNormalization(n_unit)
                self.bn5 = L.BatchNormalization(n_unit)
                self.bn6 = L.BatchNormalization(n_unit)
                self.bn7 = L.BatchNormalization(n_unit)
                self.bn8 = L.BatchNormalization(1)

    def __call__(self, x):
        if self.use_bn:
            down1 = self.activate_func(self.l1(x))
            down2 = self.activate_func(self.bn1(self.l2(down1)))
            down3 = self.activate_func(self.bn2(self.l3(down2)) + down1)
            down4 = self.activate_func(self.bn3(self.l4(down3)))
            down5 = self.activate_func(self.bn4(self.l5(down4)) + down3)
            down6 = self.activate_func(self.bn5(self.l6(down5)))
            down7 = self.activate_func(self.bn6(self.l7(down6)) + down5)
            return self.bn8(self.l8(down7))
        else:

            down1 = F.dropout(self.activate_func(self.l1(x)), self.dropout)
            down2 = F.dropout(self.activate_func(self.l2(down1)), self.dropout)
            down3 = F.dropout(self.activate_func(self.l3(down2) + down1), self.dropout)
            down4 = F.dropout(self.activate_func(self.l4(down3)), self.dropout)
            down5 = F.dropout(self.activate_func(self.l5(down4) + down3), self.dropout)
            down6 = F.dropout(self.activate_func(self.l6(down5)), self.dropout)
            down7 = F.dropout(self.activate_func(self.l7(down6) + down5), self.dropout)
            return self.l8(down7)
