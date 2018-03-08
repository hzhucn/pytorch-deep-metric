#!/usr/bin/env bash
# evaluate the effect of k
# random init  baseline

CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 16  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_k16  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_k16/600_model.pkl

CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 32  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_k32  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_k32/600_model.pkl

CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 64  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_64  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_64/600_model.pkl

CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax -epochs 601 -log_dir knnsoftmax_96  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_96/600_model.pkl


# orth init
CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -init orth -k 16  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_orth_16  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_orth_16/600_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -init orth -k 32  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_orth_32  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_orth_32/600_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -init orth -k 64  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_orth_64  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_orth_64/600_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -init orth -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 601 -log_dir knnsoftmax_orth_96  -save_step 100
python test.py  -data car -r  checkpoints/knnsoftmax_orth_96/600_model.pkl


#  orth reg
CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -orth 1e-1 -init orth -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1601 -log_dir knnsoftmax_orth_96_1e_1  -save_step 100
#python test.py  -data car -r  checkpoints/knnsoftmax_orth_96_1e_1/900_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -orth 1.0 -init orth -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1601 -log_dir knnsoftmax_orth_96_1  -save_step 100
#python test.py  -data car -r  checkpoints/knnsoftmax_orth_96_1/1300_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -orth 1e1 -init orth -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1601 -log_dir knnsoftmax_orth_96_1e1  -save_step 100
#python test.py  -data car -r  checkpoints/knnsoftmax_orth_96_1e1/1300_model.pkl

CUDA_VISIBLE_DEVICES=8 python train.py -data car  -net bn  -alpha 40 -orth 1e2 -init orth -k 96  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1601 -log_dir knnsoftmax_orth_96_1e2  -save_step 100
#python test.py  -data car -r  checkpoints/knnsoftmax_orth_96_1e2/1300_model.pkl