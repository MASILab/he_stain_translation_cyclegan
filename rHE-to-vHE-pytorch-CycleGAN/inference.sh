conic_data=$1
site=$2
weights=$3
epoch=$4
num_patches=$5

python test.py \
--dataroot "$conic_data" \
--name "$site" \
--dataset_mode unaligned_onehotseg_onecycle \
--model cycle_gan_nucseg \
--batch_size 2 \
--norm batch \
--num_threads 16 \
--load_size 256 \
--crop_size 256 \
--patch_size 256 \
--num_test "$num_patches" \
--checkpoints_dir "$weights"/ \
--epoch "$epoch"
