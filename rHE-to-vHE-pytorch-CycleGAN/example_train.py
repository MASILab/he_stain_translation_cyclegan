DATAROOT="/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/example_training_data/consep_example_less_trainB_data"

CUR_NAME="consep_example_less_trainB_data"

SAVE_WEIGHTS_HERE="/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/example_training_temp_weights"

python train.py \
        --dataroot $DATAROOT \
        --name $CUR_NAME \
        --checkpoints_dir $SAVE_WEIGHTS_HERE \
        --dataset_mode unaligned_onehotseg_onecycle \
        --model cycle_gan_nucseg \
        --batch_size 2 \
        --norm batch \
        --gpu_ids 0 \
        --num_threads 16 \
        --load_size 286 \
        --crop_size 256 \
        --patch_size 256 \
        --preprocess scale_and_width_crop
