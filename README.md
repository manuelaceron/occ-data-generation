# occ-data-generation

To occlude a dataset:

1. Specify important paths:

	oc_img_path= path of the occluders RGB images
	oc_mask_path= path of the occluders binary mask
	base_in_dir = path of dataset to occlude
	base_out_dir = path to locate the new occluded dataset

2. run create-simulated-dataset.py with parameter -m and -r:

-m: split dataset to occlude, eg. train, val, test
-r: occlusion ratio, % of samples in the dataset to occlude

eg. python create-simulated-dataset.py -m train -r 0.6
