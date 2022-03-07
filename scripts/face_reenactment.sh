set -ex

# set data path
# target_dir : directory for training video
# source_dir : directory for driving video
# tgt_video_dir : path for training video
# src_video_dir : path for driving video

target_dir="data/hwang"
source_dir="data/hwang"
tgt_video_dir=$target_dir/KKJ04_5min.mp4
src_video_dir=$source_dir/KKJ04_5min.mp4


# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio

# mkdir -p $source_dir/full
# mkdir -p $source_dir/crop
# mkdir -p $source_dir/audio
# mkdir -p $source_dir/results

# 1. Take all frames and audio of target and source data
# ffmpeg -hide_banner -y -i $tgt_video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $tgt_video_dir -ar 16000 $target_dir/audio/audio.wav

# ffmpeg -hide_banner -y -i $src_video_dir -r 25 $source_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $src_video_dir -ar 16000 $source_dir/audio/audio.wav


# 2. crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.3 \
#     --vertical_adjust 0.2

# python audiodvp_utils/crop_portrait.py \
#     --data_dir $source_dir \
#     --crop_level 1.3 \
#     --vertical_adjust 0.2

# 3. 3D face reconstruction
# WARNING! num_epoch, epoch_tex, epoch_warm_up is set assuming that input video is 5 min. If input video is shorter, each value should be adjusted according to input length
# ex) input video : 2min 30 sec
#     num_epoch : 30 -> 60
#     epoch_tex : 10 -> 20
#     epoch_warm_up : 20 -> 40

# python train.py \
#     --data_dir $target_dir \
#     --num_epoch 40 \
#     --serial_batches False \
#     --display_freq 200 \
#     --print_freq 200 \
#     --batch_size 5 \
#     --epoch_tex 10 \
#     --epoch_warm_up 20

# python expression_extract.py \
#     --data_dir $source_dir


# 4. build neural face renderer data pair
# python audiodvp_utils/build_nfr_dataset.py --data_dir $target_dir

# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/nfr/A/train/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/mask/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/nfr/B/train/%05d.png \
#     -i $target_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/nfr_dataset_debug.mp4


# 5. train neural face renderer
# python vendor/neural_face_renderer/train.py \
#     --dataroot $target_dir --name nfr --model nfr --checkpoints_dir $target_dir/ckpts \
#     --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode exemplar_train --norm batch --pool_size 0 --use_refine \
#     --input_nc 42 --Nw 7 --batch_size 8 --preprocess none --num_threads 4 --n_epochs 250 \
#     --n_epochs_decay 0 --load_size 256


# 6. Reenact and create input images of neural renderer for inference
python reenact.py --src_dir $source_dir --tgt_dir $target_dir

# choose best epoch with lowest loss
epoch=100

# 7. neural rendering the reenact face sequence
python vendor/neural_face_renderer/test.py --model test \
    --netG unet_256 \
    --direction BtoA \
    --dataset_mode exemplar_test \
    --norm batch \
    --input_nc 42 \
    --Nw 7 \
    --preprocess none \
    --eval \
    --use_refine \
    --name nfr \
    --checkpoints_dir $target_dir/ckpts \
    --dataroot $source_dir \
    --results_dir $source_dir \
    --epoch $epoch


# 8. composite lower face back to original video
python comp.py --src_dir $source_dir --tgt_dir $target_dir

# create final result

# ------ commands for making video using image files ------
# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $source_dir/comp/%05d.png \
#     -thread_queue_size 8192 -i $source_dir/reenact_landmarks/%05d.png \
#     -i $source_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $source_dir/results/comp_landmark_test.mp4

ffmpeg -y -loglevel warning \
    -thread_queue_size 8192 -i $source_dir/audio/audio.wav \
    -thread_queue_size 8192 -i $source_dir/comp/%05d.png \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $source_dir/results/comp_self_reenact_test.mp4
