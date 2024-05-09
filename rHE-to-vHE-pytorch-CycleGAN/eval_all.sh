site=$1
epoch=$2
num_patches=$3

python test.py --dataroot ./datasets/rHE2vHE/conic_patches_for_style_transfer/"$site"/all/ --name "$site"_all --dataset_mode unaligned_onehotseg_onecycle --model cycle_gan_nucseg --batch_size 2 --norm batch --gpu_ids 0 --num_threads 16 --load_size 256 --crop_size 256 --patch_size 256 --num_test "$num_patches" --checkpoints_dir jmi_cycle_gan_weights/"$site"/ --epoch "$epoch"
