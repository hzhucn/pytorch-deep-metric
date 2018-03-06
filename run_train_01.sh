#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=6 python train.py -data car  -net bn  -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 601 -log_dir branchKS  -save_step 50
python test.py  -data car -r  checkpoints/branchKS/model.pkl  >branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/100_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/200_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/300_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/400_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/500_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl -test 0 >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/700_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/800_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/900_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/1000_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/1100_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl -test 0 >>branchKS1e5.txt

CUDA_VISIBLE_DEVICES=6 python train.py -data car  -net bn -orth 1e-1 -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 601 -log_dir branchKS  -save_step 50
python test.py  -data car -r  checkpoints/branchKS/model.pkl  >branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/100_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/200_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/300_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/400_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/500_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl -test 0 >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/700_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/800_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/900_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/1000_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/1100_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl  >>branchKS1e1.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl -test 0 >>branchKS1e1.txt

CUDA_VISIBLE_DEVICES=6 python train.py -data car  -net bn -orth 1e-2 -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 601 -log_dir branchKS  -save_step 50
python test.py  -data car -r  checkpoints/branchKS/model.pkl  >branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/100_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/200_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/300_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/400_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/500_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/600_model.pkl -test 0 >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/700_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/800_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/900_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/1000_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/1100_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl  >>branchKS1e2.txt
python test.py  -data car -r  checkpoints/branchKS/1200_model.pkl -test 0 >>branchKS1e2.txt