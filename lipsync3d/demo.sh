set -ex

# set data path
# target_dir : directory for training data
# source_dir : directory for inference data, put test audio in source_dir/audio directory
# video_dir : path for training video

bfm_dir="data/tcdtimit4"
tg_path="data/tcdtimit_test/42_sx97.TextGrid"
tg_audio_path="data/tcdtimit_test/sx97_audio.wav"
# mesh_dir="data/kkj/kkj04_lipsync3d/mesh_dict"
# reenact_mesh_dir="data/kkj/kkj04_lipsync3d/reenact_norm_mesh"
# target_dir="data/kkj/kkj04_lipsync3d"
# target_dir="data/tcdtimit4_datadir"
source_dir="data/tcdtimit4"
# video_dir="data/kkj/TTS_E/a-cough-nose.mp4"


# set video clip duration
start_time="00:00:00"
end_time="240"

# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio
# mkdir -p $target_dir/results
# mkdir -p $source_dir/audio
# mkdir -p $source_dir/results

# 1. Take all frames and audio of training data
# warning! the number of extracted frames should be dividable by 5. 
# If the number of frames of training video is not dividable by 5, delete some frames manually to make the number of frames dividable by 5

# ffmpeg -hide_banner -y -i $video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $video_dir -ar 16000 $target_dir/audio/audio.wav
# ffmpeg -hide_banner -y -i $video_dir -ar 16000 $source_dir/audio/audio.wav

# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.5 \
#     --vertical_adjust 0.2

# pose normalization
# python lipsync3d/pose_normalization.py --data_dir $target_dir

# train lipsync3d net
# python lipsync3d/train.py --src_dir $target_dir --tgt_dir $target_dir

# test lipsync3d net
# python lipsync3d/test.py \
#     --batch_size 1 \
#     --serial_batches False \
#     --isTrain False \
#     --gpu_ids 0 \
#     --src_dir $source_dir \
#     --tgt_dir $target_dir

# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/mesh_image/%05d.png \
#     -thread_queue_size 8192 -i $source_dir/reenact_mesh_image/%05d.png \
#     -i $source_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -shortest -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $source_dir/results/tgt_kkj04_src_kkj00_mesh_reenact.mp4

# python lipsync3d/train_landmark2BFM.py \
#     --bfm_dir $bfm_dir \
#     --tgt_dir $target_dir \
#     --mesh_dir $mesh_dir \
#     --num_epoch 50 \
#     --batch_size 32 

# python lipsync3d/test_landmark2BFM.py \
#     --batch_size 1 \
#     --serial_batches False \
#     --isTrain False \
#     --tgt_dir $target_dir \
#     --src_dir $source_dir


# python lipsync3d/train_audio2BFM.py \
#     --tgt_dir $bfm_dir \
#     --src_dir $bfm_dir \
#     --num_epoch 400 \
#     --batch_size 32 \
#     --gpu_ids 3 \
#     --lr 1e-5 \
#     --lambda_emotion 1 \
#     --lambda_temporal 1 \
#     --moodlen 8 \
#     --suffix "emotion_mood8_stft"

# python lipsync3d/test_audio2BFM.py \
#     --batch_size 1 \
#     --serial_batches False \
#     --isTrain False \
#     --tgt_dir $bfm_dir \
#     --src_dir $source_dir \
#     --moodlen 8 \
#     --suffix "emotion_mood8_stft"

#---------------------------------
# python lipsync3d/train_audioviseme2BFM.py \
#     --tgt_dir $bfm_dir \
#     --src_dir $bfm_dir \
#     --num_epoch 400 \
#     --batch_size 32 \
#     --gpu_ids 0 \
#     --lr 1e-5 \
#     --lambda_emotion 1 \
#     --lambda_temporal 1 \
#     --moodlen 8 \
#     --suffix "stft_viseme_energy_pitch"

# python lipsync3d/test_audioviseme2BFM.py \
#     --batch_size 1 \
#     --serial_batches False \
#     --isTrain False \
#     --tgt_dir $bfm_dir \
#     --src_dir $source_dir \
#     --moodlen 8 \
#     --suffix "stft_viseme_energy_pitch"
#---------------------------------

# python reenact.py --src_dir $source_dir --tgt_dir $bfm_dir

python reenact_tg.py --tgt_dir $bfm_dir --tg_path $tg_path

# ffmpeg -y -loglevel warning \
#     -thread_queue_size 8192 -i $source_dir/reenact/%05d.png \
#     -thread_queue_size 8192 -i $source_dir/reenact_bfm_mesh/%05d.png \
#     -i $source_dir/audio/audio.wav \
#     -filter_complex hstack=inputs=2 -shortest -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $source_dir/results/self_reenact_stft_viseme_energy_pitch.mp4

ffmpeg -y -loglevel warning \
    -thread_queue_size 8192 -i $tg_audio_path\
    -thread_queue_size 8192 -i $bfm_dir/reenact_tg/42_sx97/%05d.png \
    -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p -shortest $bfm_dir/results/reenact_viterbi_sx97_interpolation.mp4