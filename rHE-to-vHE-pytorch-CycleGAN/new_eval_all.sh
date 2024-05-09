#!/bin/bash

# Update these paths depending on where you have your data, saved weights, and directory to save style transferred output
dataroot="/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/inference_data_to_make_conic_look_like_virtual_he_style"
saved_weights_dir="/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/weights"
outdir="/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/example_output"


sites=("consep" \
	"crag" \
	"dpath" \
	"glas" \
	"pannuke")

epochs=(25 \
	25 \
	35 \
	35 \
	20)

# This must be updated with the actual number of patches to style transfer.
# I picked 10 only for repo testing purposes after cleaning up code.
# Make these the total number of patches for each site.
num_patches=(10 \
	     10 \
	     10 \
	     10 \
	     10)

for ((i=0; i<${#sites[@]}; i++)); do
    site="${sites[i]}"
    epoch="${epochs[i]}"
    num_patch="${num_patches[i]}"

    echo Evaluating site: $site with weights from epoch $epoch on $num_patch patches
    echo Dataroot with site: "$dataroot"/"$site"
    echo

    python test.py --dataroot "$dataroot"/"$site" \
	    	   --name "$site" \
		   --dataset_mode unaligned_onehotseg_onecycle \
		   --model cycle_gan_nucseg \
		   --batch_size 2 \
		   --norm batch \
		   --gpu_ids 0 \
		   --num_threads 16 \
		   --load_size 256 \
		   --crop_size 256 \
		   --patch_size 256 \
		   --num_test "$num_patches" \
		   --checkpoints_dir "$saved_weights_dir" \
		   --epoch "$epoch" \
		   --results_dir "$outdir"

done



