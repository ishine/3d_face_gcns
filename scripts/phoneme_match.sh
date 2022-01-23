set -ex

# set data path
# target_dir : directory for training data (contains video and textgrid files)
#  /target_dir
#       - studio.mp4
#       - studio_1280.mp4
#       - /textgrid
#           - studio_1_0.TextGrid 
#           - ...
#       - /clip
#           - studio_1_0.mp4
#           - ...
#       - flist.txt
# source_dir : directory for inference data, put test audio in source_dir/audio directory
#   /source_dir
#       - studio_2_0.mp4
#       - studio_2_0.TextGrid
# video_dir : path for training video

target_dir="data/studio1"
source_dir="data/studio_2_0_test"
tg_path=$source_dir/studio_2_0.TextGrid
target_video_dir=$target_dir/studio_1280.mp4
source_video_dir=$source_dir/studio_2_0.mp4

# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio
# mkdir -p $source_dir/audio
# mkdir -p $source_dir/results
mkdir -p $target_dir/results

# 1. Take all frames and audio of training data
# ffmpeg -hide_banner -y -i $target_video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $target_video_dir -ar 16000 $target_dir/audio/audio.wav
# ffmpeg -hide_banner -y -i $source_video_dir -ar 16000 $source_dir/audio/audio.wav


# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.3 \
#     --vertical_adjust 0.2


# 3D face reconstruction
# python train.py \
#     --data_dir $target_dir \
#     --num_epoch 30 \
#     --serial_batches False \
#     --display_freq 200 \
#     --print_freq 200 \
#     --batch_size 5 \
#     --epoch_tex 5 \
#     --epoch_warm_up 15


# build neural face renderer data pair
python audiodvp_utils/build_nfr_dataset.py --data_dir $target_dir

# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/nfr/A/train/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/mask/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/nfr/B/train/%05d.png \
#     -i $target_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/nfr_dataset_debug.mp4


# train neural face renderer
python vendor/neural_face_renderer/train.py \
    --dataroot $target_dir --name nfr --model nfr --checkpoints_dir $target_dir/ckpts \
    --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode exemplar_train --norm batch --pool_size 0 --use_refine \
    --input_nc 42 --Nw 7 --batch_size 8 --preprocess none --num_threads 4 --n_epochs 250 \
    --n_epochs_decay 0 --load_size 256

# python reenact_tg.py --tgt_dir $target_dir --tg_path $tg_path

# choose best epoch with lowest loss
# epoch=100

# neural rendering the reenact face sequence
# python vendor/neural_face_renderer/test.py --model test \
#     --netG unet_256 \
#     --direction BtoA \
#     --dataset_mode exemplar_test \
#     --norm batch \
#     --input_nc 42 \
#     --Nw 7 \
#     --preprocess none \
#     --eval \
#     --use_refine \
#     --name nfr \
#     --checkpoints_dir $target_dir/ckpts \
#     --dataroot $source_dir \
#     --results_dir $source_dir \
#     --epoch $epoch


# composite lower face back to original video
# python comp.py --src_dir $source_dir --tgt_dir $target_dir


# create final result
# ------ commands for making video using image files ------
# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/audio/audio.wav \
#     -thread_queue_size 8192 -i $target_dir/render/%05d.png \
#     -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $target_dir/results/render.mp4
