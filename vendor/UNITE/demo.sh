# ---------- train UNITE neural face renderer ----------
# python3 vendor/UNITE/train.py \
# 	--name tcdtimit4 \
# 	--dataset_mode exemplar_train \
# 	--dataroot $target_dir \
#     --checkpoints_dir $target_dir/ckpts \
#     --Nw 7 \
#     --preprocess none \
#     --num_threads 2 \
# 	--niter 150 \
# 	--niter_decay 150 \
# 	--use_attention \
# 	--vgg_normal_correct \
# 	--fm_ratio 1.0 \
# 	--PONO \
# 	--PONO_C \
# 	--use_coordconv \
# 	--adaptor_nonlocal \
# 	--ctx_w 1.0 \
# 	--batch_size 8 \
# 	--gpu_ids 0,1,2,3


# ---------- inference UNITE neural face renderer ----------
# epoch=latest

# python3 vendor/UNITE/test.py \
# 	--name tcdtimit4 \
# 	--dataset_mode exemplar_test \
# 	--dataroot $target_dir \
#     --checkpoints_dir $target_dir/ckpts \
#     --Nw 7 \
#     --preprocess none \
#     --num_threads 1 \
# 	--use_attention \
# 	--vgg_normal_correct \
# 	--PONO \
# 	--PONO_C \
# 	--use_coordconv \
# 	--adaptor_nonlocal \
# 	--batch_size 1 \
# 	--gpu_ids 0 \
# 	--no_flip \
# 	--serial_batches \
# 	--epoch $epoch