CUDA_VISIBLE_DEVICES=0 python demo.py \
--center-num 256 \
--embed-dim 256 \
--patch-size 64 \
--checkpoint './checkpoints/RNGNet_realsense_checkpoint' \
--rgb-path '/home/firstmove/workspace/HGGD_WS/src/captures/frame__rgb.png' \
--depth-path '/home/firstmove/workspace/HGGD_WS/src/captures/frame__depth.png'
