# Style transfer to transform H&E stain style

<img width="444" alt="image" src="https://github.com/MASILab/he_stain_translation_cyclegan_MASI_INTERNAL/assets/32654540/9da37ff4-95ba-418b-8f56-97329a1fd477">

**This repo is for performing H&E style transfer associated with the manuscript titled:
"Data-driven Nucleus Subclassification on Colon H&E using Style-transferred Digital Pathology"**

The purpose of this repo is to 
1) Use pretrained cyclegans to convert the public CoNIC 2022 data into the virtual H&E style used in the manuscript (there are 5 cyclegan weights, one for each CoNIC site)
3) Be an easy starting place to train a new cyclegan for style transfer of H&E staining

This enables H&E stain translation while limiting nucleus movement by incorporating the nucleus segmentation mask.

This repo was adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
mainly by Shunxing Bao.

Install requirements from ```stain_transform_env.yaml```

# Citations
If you use this repo, please cite:
- the relevant papers from the source repo https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master
- "Data-driven Nucleus Subclassification on Colon H&E using Style-transferred Digital Pathology"

# Data formatting to convert CoNIC H&E to virtual H&E style
- Note that the CoNIC data is public data from the CoNIC Challenge 2022, and so this data can be found online elsewhere.

Each CoNIC patch & 
corresponding binarized ground truth segmentation mask (identifying all nucleus instances) 
need to be reformatted to match the below diagram.
Each conic patch in the new formatting (256 pixels x 1024 pixels) needs to be saved as a PNG.
If you are in the MASI lab, I provide paths in the default inference script (given below).
If you are not in the MASI lab, you will need to reconfigure the inference script to point to the 
expected weights, data, and results output directory.

We evaluate in the A => B direction, where testA is CoNIC patches in the format below, 
and testB can be dummy data

To better understand how data paths and testA and testB work, please see the original repo 
we based this on: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix 

<img width="1398" alt="image" src="https://github.com/MASILab/he_stain_translation_cyclegan_MASI_INTERNAL/assets/32654540/55277249-fea9-4439-9b1f-a76091b9cf98">

# Inference on CoNIC patches
Use this file to transfer CoNIC H&E patches (reformatted as above) to the virtual H&E style staining.

```bash rHE-to-vHE-pytorch-CycleGAN/new_eval_all.sh```

If you are in MASI lab, the paths are already set up in that script.
If you are not in MASI lab, you will need to update the paths as appropriate.

There are many expected outputs, but with this scheme of A=>B direction being used to convert each CoNIC patch (reformatted as above) to virtual H&E style, we care about:
- real_A (the CoNIC patch as an RGB PNG)
- fake_B (the style-transferred CoNIC patch to look like the virtual H&E staining)

Because we have dummy data on the MASI paths for testB (empty white PNGs), other outputs will look weird because it is style transferring empty patches.
Please ignore those dumy outputs.

# Pretrained weights
There are 5 weights, 1 for each CoNIC site

The weights are publicly available here: **TODO**

# Training data
trainA is CoNIC patches (formatted as above) and trainB is virtual H&E patches (RGB PNGs).

If you are in MASI lab, 20,000 virtual H&E patches can be found here: ```/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/virtual_he_patches```

If you are not in MASI lab, the virtual H&E data is private and not accessible to you. If you want to use a model that operates on the virtual H&E style, for example the resnet that classifies 14 types of cells/nuclei on virutal H&E, an option for you is to use these pretrained cyyclegans from this repo to perform inference on the CoNIC data (reformatted as above) to get the CoNIC data into the virtual H&E stain style. Then, you could train a new cyclegan or other approach to change the stain style of your H&E data to look like the staining on the virtual H&E styled CoNIC H&E. We have not tried this, and so the quality of stain transformation and applicability of virtual H&E models like the nucleus classification model are unknown and unmeasured in this circumstance.

# Example Training script
- **MASI Lab Note: Please do not train like this. The data needs to be moved locally for actual training. This script will slow down the network. It is just an example for how to set up data paths and flags to train the model. This just shows that the code is functional for training.**

```python rHE-to-vHE-pytorch-CycleGAN/example_train.py```

If you are in MASI lab, the example training script will work automatically.
Please note that there is very little data in trainB, and this needs to be replenished with way more patches to train an effective model.
Please see above for the path to more virtual H&E patches. This data this script points to is just an example. Please carefully consider which data you want to train on.

The example outputs will be here: ```/nfs/masi/remedilw/paper_journal_nucleus_subclassification/style_transfer_he/example_training_temp_weights/consep_example_less_trainB_data```

If you are not in MASI lab, just change the paths to work with your data.

Best of luck!



