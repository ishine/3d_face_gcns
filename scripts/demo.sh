set -ex

# set data path
# target_dir : directory for training data
# source_dir : directory for inference data, put test audio in source_dir/audio directory
# video_dir : path for training video

target_dir="data/kkj/kkj04_lipsync3d"
source_dir="data/kkj/kkj04_lipsync3d"
video_dir="data/kkj/kkj04_lipsync3d/KKJ_slow_04_stand.mp4"


# set video clip duration
start_time="00:00:00"
end_time="240"

# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio
# mkdir -p $source_dir/audio

# 1. Take all frames and audio of training data
# warning! the number of extracted frames should be dividable by 5. 
# If the number of frames of training video is not dividable by 5, delete some frames manually to make the number of frames dividable by 5

# ffmpeg -hide_banner -y -i $video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $video_dir -ar 16000 $target_dir/audio/audio.wav

# 2. Take deep-speech audio feature of training audio
# extract high-level feature from train audio
# python audio_feature_extract.py --data_dir $target_dir

# 3. Take deep-speech audio feature of test audio
# extract high-level feature from test audio
# mkdir -p $source_dir/feature
# python audio_feature_extract.py --data_dir $source_dir


# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.5 \
#     --vertical_adjust 0.2

# pose normalization
python lipsync3d/pose_normalization.py --data_dir $target_dir

# # 3D face reconstruction
# python train.py \
#     --data_dir $target_dir \
#     --num_epoch 20 \
#     --serial_batches False \
#     --display_freq 200 \
#     --print_freq 200 \
#     --batch_size 5 \
#     --epoch_tex 5 \
#     --epoch_warm_up 10

# build neural face renderer data pair
# python audiodvp_utils/build_nfr_dataset.py --data_dir $target_dir

# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/nfr/A/train/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/mask/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/nfr/B/train/%05d.png \
#     -i $target_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/debug.mp4


# train neural face renderer
# python vendor/neural_face_renderer/train.py \
#     --dataroot $target_dir/nfr/AB --name nfr --model nfr --checkpoints_dir $target_dir/ckpts \
#     --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode temporal --norm batch --pool_size 0 --use_refine \
#     --input_nc 21 --Nw 7 --batch_size 16 --preprocess none --num_threads 4 --n_epochs 250 \
#     --n_epochs_decay 0 --load_size 256


# # # train audio2delta network
# python train_delta.py \
#     --dataset_mode audio2expression \
#     --num_epoch 10 \
#     --serial_batches False \
#     --display_freq 800 \
#     --print_freq 800 \
#     --batch_size 16 \
#     --gpu_ids 3 \
#     --lr 1e-4 \
#     --data_dir $target_dir \
#     --net_dir $target_dir  \
#     --net_name_prefix ''


# # predict expression parameter from audio feature
# python test_delta.py --dataset_mode audio2expression \
#     --gpu_ids 3 \
#     --data_dir $source_dir \
#     --net_dir $target_dir \
#     --net_name_prefix ''

# python reenact.py --src_dir $source_dir --tgt_dir $target_dir

# choose best epoch with lowest loss
# epoch=25

# neural rendering the reenact face sequence
# python vendor/neural_face_renderer/test.py --model test \
#     --netG unet_256 \
#     --direction BtoA \
#     --dataset_mode temporal_single \
#     --norm batch \
#     --input_nc 21 \
#     --Nw 7 \
#     --preprocess none \
#     --eval \
#     --use_refine \
#     --name nfr \
#     --checkpoints_dir $target_dir/ckpts \
#     --dataroot $source_dir/reenact \
#     --results_dir $source_dir \
#     --epoch $epoch

# composite lower face back to original video
# python comp.py --src_dir $source_dir --tgt_dir $target_dir

# create final result
# mkdir -p $source_dir/results

# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $source_dir/audio/audio.wav \
#     -thread_queue_size 8192 -i $source_dir/comp/%05d.png \
#     -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $source_dir/results/self_reenact_and_test.mp4

# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/nfr/B/train/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/reenact/%05d.png \
#     -i $target_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/results/debug.mp4

# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/full/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/comp/%05d.png \
#     -i $target_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/results/inference_kkj03.mp4