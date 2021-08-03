set -ex

# set data path
target_dir="data/kkj/kkj03"
# source_dir="data/obama296"
video_dir="data/kkj/kkj03/KKJ_slow_03_stand.mp4"

# set video clip duration
start_time="00:00:00"
end_time="240"

# mkdir -p $target_dir/full
# mkdir -p $target_dir/crop
# mkdir -p $target_dir/audio

# ffmpeg -hide_banner -y -i $video_dir -r 25 $target_dir/full/%05d.png
# ffmpeg -hide_banner -y -i $video_dir $target_dir/audio/audio.aac

# mkdir -p $target_dir/feature
# python vendor/ATVGnet/code/test.py -i $target_dir/


# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.5 \
#     --vertical_adjust 0.2


# # 3D face reconstruction
# python train.py \
#     --data_dir $target_dir \
#     --num_epoch 30 \
#     --serial_batches False \
#     --display_freq 400 \
#     --print_freq 400 \
#     --batch_size 5 \
#     --epoch_tex 10 \
#     --epoch_warm_up 15

# python audiodvp_utils/rescale_image.py \
#       --data_dir $target_dir


# /usr/bin/ffmpeg -hide_banner -y -loglevel warning \
#     -thread_queue_size 8192 -i $target_dir/nfr/A/train/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/mask/%05d.png \
#     -thread_queue_size 8192 -i $target_dir/nfr/B/train/%05d.png \
#     -i $target_dir/audio/audio.aac \
#     -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/debug.mp4

# build neural face renderer data pair
# python audiodvp_utils/build_nfr_dataset.py --data_dir $target_dir

# python reenact.py --src_dir $source_dir --tgt_dir $target_dir

# train neural face renderer
python vendor/neural_face_renderer/train.py \
    --dataroot $target_dir/nfr/AB --name nfr --model nfr --checkpoints_dir $target_dir/ckpts \
    --netG unet_256 --direction BtoA --lambda_L1 100 --dataset_mode temporal --norm batch --pool_size 0 --use_refine \
    --input_nc 21 --Nw 7 --batch_size 16 --preprocess none --num_threads 4 --n_epochs 250 \
    --n_epochs_decay 0 --load_size 256
