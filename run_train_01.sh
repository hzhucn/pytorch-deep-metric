#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=8 python JSDiv_train.py -data car  -net bn -init orth -alpha 40 -gama 0.5 -sigma 1 -beta 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss JSDivKS  -epochs 1201 -log_dir JSDivKS_gama_05  -save_step 100
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/model.pkl  >JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/100_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/200_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/300_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/400_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/1000_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/600_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/600_model.pkl -test 0 >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/700_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/800_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/900_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/1000_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/1100_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/1200_model.pkl  >>JSDivKS_gama_05.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_05/1200_model.pkl -test 0 >>JSDivKS_gama_05.txt


CUDA_VISIBLE_DEVICES=8 python JSDiv_train.py -data car  -net bn -init orth -alpha 40 -gama 0.1 -sigma 1 -beta 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss JSDivKS  -epochs 1201 -log_dir JSDivKS_gama_01  -save_step 100
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/model.pkl  >JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/100_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/200_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/300_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/400_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/1000_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/600_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/600_model.pkl -test 0 >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/700_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/800_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/900_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/1000_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/1100_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/1200_model.pkl  >>JSDivKS_gama_01.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_01/1200_model.pkl -test 0 >>JSDivKS_gama_01.txt


CUDA_VISIBLE_DEVICES=8 python JSDiv_train.py -data car  -net bn -init orth -alpha 40 -gama 0 -sigma 1 -beta 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss JSDivKS  -epochs 1201 -log_dir JSDivKS_gama_00  -save_step 100
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/model.pkl  >JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/100_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/200_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/300_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/400_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/1000_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/600_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/600_model.pkl -test 0 >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/700_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/800_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/900_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/1000_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/1100_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/1200_model.pkl  >>JSDivKS_gama_00.txt
python test.py  -data car -r  checkpoints/JSDivKS_gama_00/1200_model.pkl -test 0 >>JSDivKS_gama_00.txt