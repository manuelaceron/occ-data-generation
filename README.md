# Simulated occlusion generation

To overcome the lack of fully annotated façade occlusion dataset, we develop a workflow to generate a simulated dataset by overlaying synthetic occlusions onto existing façade benchmarks which originally lacks such occlusions. The objective is to render images that realistically simulate occlusions of various types and sizes. These simulated images can be used to train models that effectively handle real-world occlusions in downstream applications. To create this comprehensive occluded dataset, we deploy the existing benchmarks that are commonly used in façade parsing tasks: [ECP](https://ieeexplore.ieee.org/document/5540068), [CMP](https://cmp.felk.cvut.cz/~tylecr1/facade/), [Graz-50](https://ieeexplore.ieee.org/document/6247857) and our in-house annotated KIT dataset. Thus, the newly established dataset, named *façade-occ*, consists of 1090 images.

Building upon the previous work of [Voo et al](https://arxiv.org/pdf/2205.06218), our dataset generation process begins with defining common façade obstructions, followed by realistically incorporating synthetic occlusions into the existing benchmark. Our framework enables configuring the percentage of occluded samples, setting size and location constraints for occluding objects. The approach is adaptable across building styles and image resolutions, supports adding new occluding object categories, and facilitates generating datasets of any scale. This dataset addresses challenges in manual occlusion annotation and enables automated real occlusion detection.

![](images/sim-dataset.png)
*Synthetic dataset generation process.*

We include tree, bush, branch, pedestrian, street light, car, truck, overhead line, flag, street sign and pole as synthetic occlusions in our dataset.
![](images/occluders-2.png)
*Examples of occluding objects.*

![](images/occ-info.png)
*Samples of façade-occ dataset with different occlusion rate.*

## Run Synthetic dataset generation process:

1. Set important paths:
   - oc_img_path: path of the occluders RGB images
   - oc_mask_path: path of the occluders binary mask
   - base_in_dir: path of dataset to occlude
   - base_out_dir: path to locate the new occluded dataset

2. run create-simulated-dataset.py with parameter -m and -r:
   - -m: split dataset to occlude, eg. train, val, test
   - -r: occlusion ratio, % of samples in the dataset to occlude
   ```bash
     python create-simulated-dataset.py -m train -r 0.6
   ```
