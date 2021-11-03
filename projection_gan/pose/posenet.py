import chainer
import chainer.functions as F
import chainer.links as L

class MLP(chainer.Chain):

    def __init__(self, n_in=34, n_unit=1024, mode='generator',
                 use_bn=True, activate_func=F.leaky_relu):
        if n_in % 2 != 0:
            raise ValueError("'n_in' must be divisible by 2.")
        if not mode in ['generator', 'discriminator']:
            raise ValueError("only 'generator' and 'discriminator' are valid "
                             "for 'mode', but '{}' is given.".format(mode))
        super(MLP, self).__init__()
        n_out = 16 if mode == 'generator' else 1
        print('MODEL: {}, N_OUT: {}, N_UNIT: {}'.format(mode, n_out, n_unit))
        self.mode = mode
        self.use_bn = use_bn
        self.activate_func = activate_func
        self.pubdropout = 0.1
        self.posedropout = 0.1
        self.camdropout=0.00
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.l1 = L.Linear(n_in, n_unit, initialW=w)
            self.l2 = L.Linear(n_unit, n_unit, initialW=w)
            self.l3 = L.Linear(n_unit, n_unit, initialW=w)
            self.l4 = L.Linear(n_unit, n_unit, initialW=w)
            self.l5 = L.Linear(n_unit, n_unit, initialW=w)
            self.l6 = L.Linear(n_unit, n_unit, initialW=w)
            self.l7 = L.Linear(n_unit, n_unit, initialW=w)
            self.l8 = L.Linear(n_unit, n_unit, initialW=w)
            self.l9 = L.Linear(n_unit, 16, initialW=w)

            self.l10 = L.Linear(n_unit, n_unit, initialW=w)
            self.l11 = L.Linear(n_unit, n_unit, initialW=w)
            self.l12 = L.Linear(n_unit, n_unit, initialW=w)
            self.l13 = L.Linear(n_unit, n_unit, initialW=w)
            self.l14 = L.Linear(n_unit, n_unit, initialW=w)
            self.l15 = L.Linear(n_unit, 6, initialW=w)


            if self.use_bn:
                self.bn1 = L.BatchNormalization(n_unit)
                self.bn2 = L.BatchNormalization(n_unit)
                self.bn3 = L.BatchNormalization(n_unit)
                self.bn4 = L.BatchNormalization(n_unit)
                self.bn5 = L.BatchNormalization(n_unit)
                self.bn6 = L.BatchNormalization(n_unit)
                self.bn7 = L.BatchNormalization(n_unit)
                self.bn8 = L.BatchNormalization(n_unit)
                self.bn9 = L.BatchNormalization(n_unit)
                self.bn10 = L.BatchNormalization(n_unit)
                self.bn11 = L.BatchNormalization(n_unit)
                self.bn12 = L.BatchNormalization(n_unit)
                self.bn13 = L.BatchNormalization(n_unit)
                self.bn14 = L.BatchNormalization(n_unit)
                self.bn15 = L.BatchNormalization(n_unit)

    def __call__(self, x):
        self.x = x
        if self.use_bn:
            # dimensions up to n_unit
            self.h1 = F.dropout(self.activate_func(self.bn1(self.l1(x))), self.pubdropout)  # 34 -> n_unit
            # first public block
            self.h2 = F.dropout(self.activate_func(self.bn2(self.l2(self.h1))), self.pubdropout)  # n_unit
            self.h3 = F.dropout(self.activate_func(self.bn3(self.l3(self.h2)) + self.h1), self.pubdropout)  # n_unit
            # first pose estimation public block
            self.h4 = F.dropout(self.activate_func(self.bn4(self.l4(self.h3))), self.posedropout)
            self.h5 = F.dropout(self.activate_func(self.bn5(self.l5(self.h4)) + self.h3), self.posedropout)
            # second pose estimation block
            self.h6 = F.dropout(self.activate_func(self.bn6(self.l6(self.h5))), self.posedropout)
            self.h7 = F.dropout(self.activate_func(self.bn7(self.l7(self.h6)) + self.h5), self.posedropout)
            self.h8 = F.dropout(self.activate_func(self.bn8(self.l8(self.h7))), self.posedropout)
            self.h9 = self.l9(self.h8)
            # first cam estimation block
            self.h10 = F.dropout(self.activate_func(self.bn9(self.l10(self.h3))), self.camdropout)
            self.h11 = F.dropout(self.activate_func(self.bn10(self.l11(self.h10)) + self.h3), self.camdropout)
            # second cam estimation block
            self.h12 = F.dropout(self.activate_func(self.bn11(self.l12(self.h11))), self.camdropout)
            self.h13 = F.dropout(self.activate_func(self.bn12(self.l13(self.h12)) + self.h11), self.camdropout)
            self.h14 = F.dropout(self.activate_func(self.bn13(self.l14(self.h13))), self.camdropout)
            self.h15 = self.l15(self.h14)
        else:
            # dimensions up to n_unit
            self.h1 = F.dropout(self.activate_func(self.l1(x)), self.pubdropout)  # 34 -> n_unit
            # first public block
            self.h2 = F.dropout(self.activate_func(self.l2(self.h1)), self.pubdropout)  # n_unit
            self.h3 = F.dropout(self.activate_func(self.l3(self.h2) + self.h1), self.pubdropout)  # n_unit
            # first pose estimation public block
            self.h4 = F.dropout(self.activate_func(self.l4(self.h3)), self.posedropout)
            self.h5 = F.dropout(self.activate_func(self.l5(self.h4) + self.h3), self.posedropout)
            # second pose estimation block
            self.h6 = F.dropout(self.activate_func(self.l6(self.h5)), self.posedropout)
            self.h7 = F.dropout(self.activate_func(self.l7(self.h6) + self.h5), self.posedropout)
            self.h8 = F.dropout(self.activate_func(self.l8(self.h7)), self.posedropout)
            self.h9 = self.l9(self.h8)
            # first cam estimation block
            self.h10 = F.dropout(self.activate_func(self.l10(self.h3)), self.camdropout)
            self.h11 = F.dropout(self.activate_func(self.l11(self.h10) + self.h3), self.camdropout)
            # second cam estimation block
            self.h12 = F.dropout(self.activate_func(self.l12(self.h11)), self.camdropout)
            self.h13 = F.dropout(self.activate_func(self.l13(self.h12) + self.h11), self.camdropout)
            self.h14 = F.dropout(self.activate_func(self.l14(self.h13)), self.camdropout)
            self.h15 = self.l15(self.h14)
        return self.h9, self.h15