set -ex

target_dir="data/studio1"
inputs=""
i=0

for f in $target_dir/clip/*.mp4
do
    inputs="${inputs} -i ${f}"
    ((i=i+1))
    echo file \'clip/${f##*/}\' >> $target_dir/flist.txt
done

ffmpeg $inputs -filter_complex concat=n=$i:v=1:a=1 $target_dir/studio.mp4 -y

ffmpeg -i $target_dir/studio.mp4 -vf scale=1280:720 $target_dir/studio_1280.mp4