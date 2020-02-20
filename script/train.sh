CUDA_VISIBLE_DEVICES=0 python main_lm.py --data data/pennchar --batch_size 64 --emsize 512 --nlayers 1 --nhid 512 --bptt 100 --lr 0.002 --epochs 200 \
--dropouto 0. --dropouth 0. --dropouti 0. --dropoute 0. --alpha 0. --beta 0. --wdecay 0. --save BINARY_AVE_WEIGHT_PENNCHAR \
 --optimizer adam --clip 5 --norm layer --char --log-interval 10
