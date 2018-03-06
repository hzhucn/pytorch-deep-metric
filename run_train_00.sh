#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 100  -lr 1e-6 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1201 -log_dir knnsoftmax  -save_step 50
python test.py  -data car -r  checkpoints/knnsoftmax/model.pkl  >knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/100_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/200_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/300_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/400_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/500_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl -test 0 >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/700_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/800_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/900_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1000_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1100_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl  >>knnsoftmax.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl -test 0 >>knnsoftmax.txt

CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 100  -lr 1e-5 -dim 512   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1201 -log_dir knnsoftmax  -save_step 50
python test.py  -data car -r  checkpoints/knnsoftmax/model.pkl  >knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/100_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/200_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/300_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/400_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/500_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl -test 0 >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/700_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/800_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/900_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1000_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1100_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl  >>knnsoftmax1e5.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl -test 0 >>knnsoftmax1e5.txt


CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 100  -lr 1e-5 -dim 256   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1201 -log_dir knnsoftmax  -save_step 50
python test.py  -data car -r  checkpoints/knnsoftmax/model.pkl  >knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl  >>knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/800_model.pkl  >>knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1000_model.pkl  >>knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1100_model.pkl  >>knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl  >>knnsoftmax1e5d256.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl -test 0 >>knnsoftmax1e5d256.txt


CUDA_VISIBLE_DEVICES=7 python train.py -data car  -net bn  -alpha 40 -k 100  -lr 1e-5 -dim 128   -num_instances 8 -BatchSize 128  -loss knnsoftmax  -epochs 1601 -log_dir knnsoftmax  -save_step 50
python test.py  -data car -r  checkpoints/knnsoftmax/model.pkl  >knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/350_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/600_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/800_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1000_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1200_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1400_model.pkl  >>knnsoftmax1e5d128.txt
python test.py  -data car -r  checkpoints/knnsoftmax/1600_model.pkl -test 1 >>knnsoftmax1e5d128.txt