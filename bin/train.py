from __future__ import print_function

import argparse
import json
import multiprocessing
import time
import chainer
import chainer.functions as F
from chainer import training
from chainer.training import extensions
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from projection_gan.pose.posenet import MLP
from projection_gan.pose.dataset.pose_dataset import H36M, MPII
from projection_gan.pose.dataset.mpii_inf_3dhp_dataset import MPII3DDataset
from projection_gan.pose.updater import H36M_Updater
from projection_gan.pose.evaluator import Evaluator
from projection_gan.pose.Dnet import Discriminator


def create_result_dir(dirname):

    if not os.path.exists('results'):
        os.mkdir('results')
    if dirname:
        result_dir = os.path.join('results', dirname)
    else:
        result_dir = os.path.join(
            'results', time.strftime('%Y-%m-%d_%H-%M-%S'))

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    return result_dir



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-b', '--batchsize', type=int, default=512) #16
    parser.add_argument('-B', '--test_batchsize', type=int, default=1024)  #32
    parser.add_argument('-F', '--frame_interval', type=int, default=5) # fetch 1 frame per F
    parser.add_argument('-f', '--frame_per', type=int, default=2) # fetch f frames for training and discriminating
    parser.add_argument('-r', '--resume', default='')
    parser.add_argument('-o', '--out', type=str, default='')
    parser.add_argument('-e', '--epoch', type=int, default=20)
    parser.add_argument('-m', '--mode', type=str, default='unsupervised',
                        choices=['supervised', 'unsupervised'])
    parser.add_argument('-d', '--dataset', type=str, default='h36m',
                        choices=['h36m', 'mpii', 'mpi_inf'])
    parser.add_argument('-a', '--activate_func',
                        type=str, default='leaky_relu')
    parser.add_argument('-A', '--action', type=str, default='all')
    parser.add_argument('-s', '--snapshot_interval', type=int, default=1)
    parser.add_argument('-l', '--log_interval', type=int, default=1)
    parser.add_argument('-p', '--model_path', type=str, default='')
    parser.add_argument('--heuristic_loss_weight', type=float, default=1.0)
    parser.add_argument('--use_heuristic_loss', action="store_true")
    parser.add_argument('--use_sh_detection', action="store_true")
    parser.add_argument('--use_bn', action="store_true")
    args = parser.parse_args()
    args.out = create_result_dir(args.out)


    # Save options.
    with open(os.path.join(args.out, 'options.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(json.dumps(vars(args), indent=2))


    gen = MLP(mode='generator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))
    dis = Discriminator(mode='discriminator', use_bn=args.use_bn,
              activate_func=getattr(F, args.activate_func))

    if args.gpu >= 0:

        chainer.cuda.get_device_from_id(args.gpu).use()

        gen.to_gpu()
        dis.to_gpu()

    # Setup an optimizer

    def make_optimizer4gen(model):

        optimizer = chainer.optimizers.Adam(alpha=5.5e-5, beta1=0.7, beta2=0.9)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        return optimizer
	
    def make_optimizer4dis(model):

        optimizer = chainer.optimizers.Adam(alpha=5.5e-5, beta1=0.7, beta2=0.9)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-5))

        return optimizer

    opt_gen = make_optimizer4gen(gen)
    opt_dis = make_optimizer4dis(dis)



    # Load dataset.

    if args.dataset == 'h36m':
        train = H36M(action=args.action, train=True,
                     use_sh_detection=args.use_sh_detection, frame_interval=args.frame_interval,
                     frame_per=args.frame_per)
        test = H36M(action=args.action, train=False,
                    use_sh_detection=args.use_sh_detection)

    elif args.dataset == 'mpii':
        train = MPII(train=True, use_sh_detection=args.use_sh_detection)
        test = MPII(train=False, use_sh_detection=args.use_sh_detection)

    elif args.dataset == 'mpi_inf':
        train = MPII3DDataset(train=True)
        test = MPII3DDataset(train=False)

    print('TRAIN: {}, TEST: {}'.format(len(train), len(test)))



    multiprocessing.set_start_method('spawn')
    train_iter = chainer.iterators.MultiprocessIterator(train, args.batchsize)
    test_iter = chainer.iterators.MultiprocessIterator(
        test, args.test_batchsize, repeat=False, shuffle=False)


    # Set up a trainer
    updater = H36M_Updater(
        gan_accuracy_cap=args.gan_accuracy_cap,
        use_heuristic_loss=args.use_heuristic_loss,
        heuristic_loss_weight=args.heuristic_loss_weight,
        frame_interval=args.frame_interval,
        frame_per=args.frame_per,
        mode=args.mode, iterator={'main': train_iter, 'test': test_iter},
        optimizer={'gen': opt_gen, 'dis': opt_dis}, device=args.gpu)


    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    log_interval = (args.log_interval, 'epoch')

    snapshot_interval = (10, 'epoch')
    gen_interval = (1, 'epoch')




    trainer.extend(Evaluator(test_iter, {'gen': gen}, device=args.gpu),
                   trigger=log_interval)
    trainer.extend(extensions.snapshot(
       filename='snapshot_epoch_{.updater.epoch}'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(
       gen, 'gen_epoch_{.updater.epoch}.npz'), trigger=gen_interval)


    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport([
        'epoch', 'iteration', 'gen/loss', 'gen/z_mse', 'gen/loss_heu', 'gen/loss_pred', 'gen/loss_cam', 'gen/loss_symm', 'gen/loss_3d', 'dis/loss_gp',
        'dis/loss', 'dis/real', 'dis/fake', 'gen/loss_rep', 'validation/gen/xyz', 'validation/gen/z_mse',
        'validation/gen/mpjpe', 'validation/gen/xy', 'validation/gen/z', 'validation/gen/p_mp', 'validation/gen/PCK'
    ]), trigger=log_interval)

    trainer.extend(extensions.ProgressBar(update_interval=10))


    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)


    # Run the training
    trainer.run()





if __name__ == '__main__':

    main()
    time.sleep(3)
