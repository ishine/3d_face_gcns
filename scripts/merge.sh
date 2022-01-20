set -ex

target_dir="data/studio1"

for f in $target_dir/clip/*.mp4
do
    echo file \'clip/${f##*/}\' >> $target_dir/flist.txt
done

ffmpeg -f concat -safe 0 -i $target_dir/flist.txt -c copy $target_dir/studio.mp4

ffmpeg -i $target_dir/studio.mp4 -vf scale=1280:720 $target_dir/studio_1280.mp4