#!/bin/bash
for i in 1 2 3 4 5
do
  ./BPPANNS --M 40 --efc 600 --efs 20 --efsm 1000 --efss 10 --s 1024 --beta 360 --k 10 --ratio 1 --exp 3 --database /home/zhangyd/data/sift1B/sift1b_base_75M.fvecs --dataquery /home/zhangyd/data/sift1B/sift1b_query.fvecs --groundtruth /home/zhangyd/data/sift1B/sift1b_groundtruth_75M.ivecs
done
