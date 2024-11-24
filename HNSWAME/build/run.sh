#!/bin/bash

efs_values=(10 20 40 80 160 320 640)
ratio_values=(1 2 4 8 16 32 64)

for i in "${!efs_values[@]}"; do
  EFS=${efs_values[$i]}
  RATIO=${ratio_values[$i]}
  echo "efs = $EFS, ratio = $RATIO"
  ./HNSWAME --M 40 --efc 600 --efs $EFS --efsm $EFS --efss 10 --s 1024 --beta 450 --k 10 --ratio $RATIO --exp 3 --database /home/zhangyd/data/sift/sift_base.fvecs --dataquery /home/zhangyd/data/sift/sift_query.fvecs --groundtruth /home/zhangyd/data/sift/sift_groundtruth.ivecs --dim 128
  ./HNSWAME --M 40 --efc 600 --efs $EFS --efsm $EFS --efss 10 --s 1024 --beta 2.5 --k 10 --ratio $RATIO --exp 3 --database /home/zhangyd/data/gist/gist_base.fvecs --dataquery /home/zhangyd/data/gist/gist_query.fvecs --groundtruth /home/zhangyd/data/gist/gist_groundtruth.ivecs --dim 960
  ./HNSWAME --M 40 --efc 600 --efs $EFS --efsm $EFS --efss 10 --s 1024 --beta 5 --k 10 --ratio $RATIO --exp 3 --database /home/zhangyd/data/glove/glove_base.fvecs --dataquery /home/zhangyd/data/glove/glove_query.fvecs --groundtruth /home/zhangyd/data/glove/glove_groundtruth.ivecs --dim 100
  ./HNSWAME --M 40 --efc 600 --efs $EFS --efsm $EFS --efss 10 --s 1024 --beta 1.1 --k 10 --ratio $RATIO --exp 3 --database /home/zhangyd/data/deep/deep_base.fvecs --dataquery /home/zhangyd/data/deep/deep_query.fvecs --groundtruth /home/zhangyd/data/deep/deep_groundtruth.ivecs --dim 96
done

