#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=5 python train.py -data car  -net bn -orth 1e-3 -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 1201 -log_dir branchKS_orth -save_step 50
python test.py  -data car -r  checkpoints/branchKS_orth/model.pkl  >branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/100_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/200_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/300_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/400_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/500_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl -test 0 >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/700_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/800_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/900_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1000_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1100_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl  >>branchKS1e3.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl -test 0 >>branchKS1e3.txt

CUDA_VISIBLE_DEVICES=5 python train.py -data car  -net bn -orth 1e-4 -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 1201 -log_dir branchKS_orth -save_step 50
python test.py  -data car -r  checkpoints/branchKS_orth/model.pkl  >branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/100_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/200_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/300_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/400_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/500_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl -test 0 >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/700_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/800_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/900_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1000_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1100_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl  >>branchKS1e4.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl -test 0 >>branchKS1e4.txt

CUDA_VISIBLE_DEVICES=5 python train.py -data car  -net bn -orth 1e-5 -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss branchKS  -epochs 1201 -log_dir branchKS_orth -save_step 50
python test.py  -data car -r  checkpoints/branchKS_orth/model.pkl  >branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/100_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/200_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/300_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/400_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/500_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/600_model.pkl -test 0 >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/700_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/800_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/900_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1000_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1100_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl  >>branchKS1e5.txt
python test.py  -data car -r  checkpoints/branchKS_orth/1200_model.pkl -test 0 >>branchKS1e5.txt