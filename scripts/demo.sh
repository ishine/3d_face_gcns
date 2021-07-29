set -ex

# set data path
target_dir="data/obama296"

# set video clip duration
start_time="00:00:00"
end_time="240"

# mkdir -p $target_dir/crop

# mkdir -p $target_dir/feature
# python vendor/ATVGnet/code/test.py -i $target_dir/

# # crop and resize video frames
# python audiodvp_utils/crop_portrait.py \
#     --data_dir $target_dir \
#     --crop_level 1.5 \
#     --vertical_adjust 0.2


# # 3D face reconstruction
python train.py \
    --data_dir $target_dir \
    --num_epoch 30 \
    --serial_batches False \
    --display_freq 400 \
    --print_freq 400 \
    --batch_size 5

/usr/bin/ffmpeg -hide_banner -y -loglevel warning \
    -thread_queue_size 8192 -i $target_dir/render/%05d.png \
    -thread_queue_size 8192 -i $target_dir/crop/%05d.png \
    -thread_queue_size 8192 -i $target_dir/overlay/%05d.png \
    -i $target_dir/audio/audio.aac \
    -filter_complex hstack=inputs=3 -vcodec libx264 -preset slower -profile:v high -crf 18 -pix_fmt yuv420p $target_dir/debug.mp4

# build neural face renderer data pair
# python audiodvp_utils/build_nfr_dataset.py --data_dir $target_dir

# python reenact.py --src_dir $target_dir --tgt_dir $target_dir