#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=7 python JSDiv_train.py -data car  -net bn -init orth -alpha 40 -gama 1 -sigma 1 -beta 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss JSDivKS  -epochs 1201 -log_dir JSDivKS_gama_1  -save_step 100
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/model.pkl  >JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/100_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/200_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/300_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/400_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/1000_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/600_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/600_model.pkl -test 0 >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/700_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/800_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/900_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/1000_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/1100_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/1200_model.pkl  >>JSDivKS_gama_1.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_1/1200_model.pkl -test 0 >>JSDivKS_gama_1.txt