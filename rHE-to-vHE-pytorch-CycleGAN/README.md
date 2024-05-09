# dataset prepare
create a folder in ./datasets/rHE2vHE
create four subfolders as ./datasets/rHE2vHE/trainA, ./datasets/rHE2vHE/trainB, ./datasets/rHE2vHE/testA, ./datasets/rHE2vHE/testB

in trainA, each image should be a 256height * (256 * 4)width. where the first three patches would be RGB separate channels of realHE. If you use cv2, please take care that cv2 split would use split image to BGR order. The fouth channel would be the semantic segmentation binary mask as [0,1]

in trainB, just put each image as 256*256 RGB vHE patches there. 

Do not worry about the name matching in trainA and trainB. The number of images in trainA and trainB is not necessary same.

The testA should place the rHE with mask image to be inferenced. The testB should be RGB image of vHE. well we don't care about the synthesis from vHE to rHE direction. 



# training sample script, in code home dir, please run
python train.py --dataroot ./datasets/rHE2vHE --name exp_rHE2vHE --dataset_mode unaligned_onehotseg_onecycle --model cycle_gan_nucseg --batch_size 2 --norm batch --gpu_ids 1 --num_threads 16 --load_size 286 --crop_size 256 --patch_size 256 --preprocess scale_width_and_crop

where exp_rHE2vHE can be replaced by any customized name

the batch size is tunable. I usually let the batch size saturate the GPU. The synthesis model is different from training a segmentation model, a larger batch size usually leads better performance.
num_thread is tunable based on the workstation
gpu_ids should be refer to the target device via nvidia-smi
the value of load_size and crop_size is referenced from the original cycleGAN
patch_size is hard code as 256, when dataloading, we will use this to split the input image in trainA to separte patches.

the training results should be under ./checkpoints/exp_rHE2vHE/, where you can check the loss per epoch. The model will be saved every 5 epoch (if you have enough data), I usually sum up the overall loss and find the minimum one to get the best epoch. If the best epoch is not divided by 5, just choose an epoch whose value is divided by 5. You can visualize the results at ./checkpoints/rHE2vHE/web/images/, where
* realA should be visulize as realH&E, the code is able to merge the separate channels intro RGB
* realB should be vHE
* fakeA is synthesized rHE
* fakeB is synthesized vHE, which you'd need for validating your current segmentation models
* recA recB are used to calculate cycleloss that input fake image to see if the recreated image is able to match the input image that is used to generate fake image
* Atruth is the segmenation truth that is related to realA, which is the 4th channel of images in trainA.
* seg_B is the generated segmentation mask that takes fakeB as input. We calculate dice_loss between seg_B and Atruth.
* the segmentaiton task is a very easy task, I usually see 'perfect' labels after just a few epochs.


# testing sample script, in code home dir, please run
python test.py --dataroot ./datasets/rHE2vHE --name exp_rHE2vHE --dataset_mode unaligned_onehotseg_onecycle --model cycle_gan_nucseg --batch_size 2 --norm batch --gpu_ids 1 --num_threads 16 --load_size 256 --crop_size 256 --patch_size 256 --num_test xxxxx

the infernece results should be under ./results/exp_rHE2vHE. 

where the batch_size should match the training script. here we set load_size as 256 so there won't random crop the input dataset.
num_test is the total images to be inferenced under testA folder.
