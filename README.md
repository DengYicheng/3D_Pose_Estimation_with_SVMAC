# SVMAC: Unsupervised 3D Human Pose Estimation from a Single Image with Single-view-multi-angle Consistency

This is the authors' implementation of [SVMAC: Unsupervised 3D Human Pose Estimation from a Single Image with Single-view-multi-angle Consistency]

## Dependencies(Recommended versions)
  - Python 3.6.5
  - Cupy 5.4.0
  - Chainer 5.4.0

## Training
#### Human3.6M dataset
  - [x] Unsupervised learning of 3D points from ground truth 2D points

    ```
    python bin/train.py --gpu 0 --mode unsupervised --dataset h36m --use_heuristic_loss --use_bn --epoch 200
    ```
  - [x] Unsupervised learning of 3D points from detected 2D points by Stacked Hourglass

    python bin/train.py --gpu 0 --mode unsupervised --dataset h36m --use_heuristic_loss --use_bn --epoch 200 --use_sh_detection

#### MPI-INF-3DHP dataset
- [x] Unsupervised learning of 3D points from ground truth 2D points

    ```
    python bin/train.py --gpu 0 --mode unsupervised --dataset mpi_inf --use_heuristic_loss --use_bn --epoch 200
    ```

## Evaluation
#### MPJPE
    ```
    python bin/eval.py results/***/gen_epoch_*.npz
    ```
#### PCK and AUC
    ```
    python bin/eval_pck.py results/***/gen_epoch_*.npz
    ```
