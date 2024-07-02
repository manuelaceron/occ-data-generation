import os
import numpy as np
import cv2
from utils.utils import *
from utils.paste_over import *
import random
import skimage
import glob
import sys
import pdb
import argparse

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    
def resize(occluder_img, occluder_mask,factor,occ_w,occ_h):
    dim = (int(occ_w*factor), int(occ_h*factor))
    occluder_img = cv2. resize(occluder_img, dim)
    occluder_mask = cv2. resize(occluder_mask, dim)
    return occluder_img, occluder_mask

def occlude_images(img_file, img_path, mask_path, oc_file, oc_img_path, oc_mask_path):

    oc_type = oc_file.split('_')[0]
    
    # Get source img and label
    src_img, src_mask = get_srcNmask(img_file,img_path,mask_path)

    # Get occluder img and label
    occluder_img, occluder_mask = get_occluderNmask(oc_file,oc_img_path,oc_mask_path)

    src_h, src_w, src_d = src_img.shape
    occ_h, occ_w, occ_d = occluder_img.shape

    # Rectangle around mask
    src_rect = cv2.boundingRect(src_mask) 
    x,y,w,h = src_rect
    height, width = src_mask.shape
    
    #------- Location constraints-----------#

    in_floor = {"truck", "car"}
    if oc_type in  in_floor:
        occluder_coord = np.random.uniform([x,y+1.2*h], [x+w,height]) 
        
    elif oc_type == "treet":
        occluder_coord = np.random.uniform([x,y+0.8*h], [x+w,height]) 
    
    elif oc_type == "ppl":
        occluder_coord = np.random.uniform([x,y+1.1*h], [x+w,height]) 
    
    elif oc_type == "lamp" or oc_type == "sign":
        occluder_coord = np.random.uniform([x,y+0.60*h], [x+w,height]) 
    
    elif oc_type == "elec":
        occluder_coord = np.random.uniform([x,y+0.5*h], [x+w,0.8*h]) 
    
    elif oc_type == "tree":
        occluder_coord = np.random.uniform([x+(w*2/3),y+0.5*h], [x+(2*w),0.8*h])

    else:
        occluder_coord = np.random.uniform([x,y], [x+w,y+h]) #random

    
    #------- Size constraints-----------#

    if  oc_type == "car":
        #factor = np.random.uniform((src_w*1.1)/occ_w, (src_w*0.85)/occ_w) 
        factor = np.random.uniform((src_w*1.0)/occ_w, (src_w*0.75)/occ_w) #for modern
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)

    elif oc_type == "truck":        
        #factor = np.random.uniform((src_w*1.5)/occ_w, (src_w*0.85)/occ_w) 
        factor = np.random.uniform((src_w*1.0)/occ_w, (src_w*0.75)/occ_w) #for modern
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "treet" or oc_type == "tree":
        factor = np.random.uniform((src_h)/occ_h, (src_h*0.75)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "flag":
        factor = np.random.uniform((src_w*0.5)/occ_w, (src_w*0.25)/occ_w) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "lamp" or oc_type == "sign":
        factor = np.random.uniform((src_h)/occ_h, (src_h*0.8)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    elif oc_type == "ppl":
        factor = np.random.uniform((src_h*0.5)/occ_h, (src_h*0.35)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
    # Constraint added for modern dataset
    elif oc_type == "elec":
        factor = np.random.uniform((src_h)/occ_h, (src_h*0.8)/occ_h) 
        occluder_img, occluder_mask = resize(occluder_img, occluder_mask, factor,occ_w,occ_h)
    
        
    occlusion_mask=np.zeros(src_mask.shape, np.uint8)
    occlusion_mask[(occlusion_mask>0) & (occlusion_mask<255)]=255

    # Paste occluder to src image

    gsr = 1 # Radius for Gaussian blur on alpha channel (label)
    gsr_e = 1 # Radius for Gaussian blur on edgeds on occluded image
    result_img, result_mask, occlusion_mask= paste_over(occluder_img,occluder_mask,src_img,src_mask,occluder_coord,occlusion_mask,False, gsr, oc_type)

    # Blur edges of occluder
    kernel = np.ones((3,3),np.uint8) 
    occlusion_mask_edges=cv2.dilate(occlusion_mask,kernel,iterations = 2)-cv2.erode(occlusion_mask,kernel,iterations = 2)
    ret, filtered_occlusion_mask_edges = cv2.threshold(occlusion_mask_edges, 240, 255, cv2.THRESH_BINARY)
    blurred_image = cv2.GaussianBlur(result_img,(gsr_e,gsr_e),0)
    result_img = np.where(np.dstack((np.invert(filtered_occlusion_mask_edges==255),)*3), result_img, blurred_image)

    # Save images
    save_images(img_file,result_img,result_mask,occlusion_mask, gsr, gsr_e)
    
def save_images(img_name,image,mask,occlusion_mask,src_img, src_mask):
    if not os.path.exists(outputImgDir):
        os.makedirs(outputImgDir)
    if not os.path.exists(outputMaskDir):
        os.makedirs(outputMaskDir)
    if not os.path.exists(occlusionMaskDir):
        os.makedirs(occlusionMaskDir)
    if not os.path.exists(output_ori_img_mask):
        os.makedirs(output_ori_img_mask)
     
    cv2.imwrite(os.path.join(outputImgDir, img_name),image) 
    cv2.imwrite(os.path.join(outputMaskDir, img_name.split('.')[0]+".png"),mask)
    cv2.imwrite(os.path.join(occlusionMaskDir, img_name.split('.')[0]+".png"),occlusion_mask)    

def prepare_data_to_occlude(oc_img_path, img_path, ratio): 
    
    # List occlusions and RGB images to occlude
    oc_files = os.listdir(oc_img_path)
    im_files = os.listdir(img_path)
    random.shuffle(im_files)

    # Set amount of occluded samples in dataset according to given ratio
    oc_ratio = int(len(im_files)*ratio)

    # Randomly choose the samples to occlude
    im_sample = random.sample(im_files, oc_ratio)
    print("Total ", len(im_files), " files, occlusion ", len(im_sample), " samples")

    # Separate occluders: trunk trees, hanging vegetation, any other occluder
    occ_vegetation = [x for x in oc_files if x.split('_')[0] in {'treet'}]
    occ_vegetation_up = [x for x in oc_files if x.split('_')[0] in {'tree'}]
    occ_any = [x for x in oc_files if x.split('_')[0] not in {'tree', 'treet'}]

    random.shuffle(occ_vegetation)
    random.shuffle(occ_vegetation_up)
    random.shuffle(occ_any)

    # Determine total amount of vegetation occluder
    occ_veg_total = len(occ_vegetation) + len(occ_vegetation_up) 

    # Ensure that 50% of the occlusions are vegetation, of which 40% is hanging vegetation and 60% is trunk trees
    any_occ_size = int(len(im_sample)*0.5)
    veg_occ_size_up = int((len(im_sample) - any_occ_size)*0.4)
    veg_occ_size = (len(im_sample) - any_occ_size - veg_occ_size_up)
    veg_occ_size_total = veg_occ_size + veg_occ_size_up
    
    """
        Set a list with vegetation occluders to use.
        Since the number of occluders is limited, the number of synthetic occluders may be less than the number of occluders required for the dataset, 
        therefore, it may be necessary to repeat occluder objects to comply with veg_occ_size_total.
    """
    veg_file = []
    
    if occ_veg_total < veg_occ_size: # If there are less occluders than needed
        veg_file = random.sample(occ_vegetation_up, len(occ_vegetation_up)) # Take all hanging trees
        veg_file = veg_file + random.sample(occ_vegetation, len(occ_vegetation)) # Take all trunk trees
        veg_file = veg_file + random.choices(occ_vegetation + occ_vegetation_up, k= (veg_occ_size_total- occ_veg_total) ) # Take additional elements
    
    else:              
        if len(occ_vegetation) < veg_occ_size:
            veg_file = veg_file + random.sample(occ_vegetation, len(occ_vegetation))
            veg_file = veg_file + random.choices(occ_vegetation, k= (veg_occ_size - len(occ_vegetation)))
        else:
            veg_file = veg_file + random.sample(occ_vegetation, veg_occ_size)
        

        if len(occ_vegetation_up) < veg_occ_size_up:
            veg_file = veg_file + random.sample(occ_vegetation_up, len(occ_vegetation_up))
            veg_file = veg_file + random.choices(occ_vegetation_up, k= (veg_occ_size_up - len(occ_vegetation_up)))
        else:
            veg_file = veg_file + random.sample(occ_vegetation_up, veg_occ_size_up)


    if len(occ_any) < any_occ_size:
        any_file = random.sample(occ_any, len(occ_any))
        any_file = any_file + random.choices(occ_any, k = (any_occ_size - len(occ_any)))
    else:
        any_file = random.sample(occ_any, any_occ_size)

    
    woc_file = veg_file + any_file
    random.shuffle(woc_file)

    #woc_file = random.choices(oc_files, weights=test_prio, k = len(im_sample)) #with replacement?
    print('Total occlusions: ', woc_file)
    
    return woc_file, im_sample, im_files


"""
Set seeds: 
    - ECP: 112
    - Graz: 112
    - full-occ60: 112
    - modern-occ100: 110
    - ecp-refmcv-occluded60: 112
"""

set_random_seed(110)

# Parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", default="test")
parser.add_argument("-r", "--ratio", default=0.6)
args = parser.parse_args()
mode = args.mode
ratio = float(args.ratio)
print('Mode: ',mode)
print('Occ ratio: ',ratio)

"""
    Set input paths:
    - oc_img_path: path of the occluder RGB images
    - oc_mask_path: path of the occluder binary mask
    - base_in_dir: path of dataset to occlude
    - base_out_dir: path to locate the new occluded dataset
    
    Output directories:
     - outputImgDir: occluded RGB image
     - outputMaskDir: occluded binary label
     - occlusionMaskDir: binary mask of occluder object
     - output_ori_img_mask: original unoccluded binary label
     - output_ori_img_dir: original unoccluded RGB image 
"""

oc_img_path= "/home/manuela/Dev/occ-generation/occluders/images/"
oc_mask_path="/home/manuela/Dev/occ-generation/occluders/labels/"
base_in_dir = "/home/manuela/Dev/occ-generation/dataset/split-data/cmp/"
base_out_dir = "/home/manuela/Dev/occ-generation/dataset/split-data/cmp/occ/"

img_path= os.path.join(base_in_dir, mode, "images")
mask_path= os.path.join(base_in_dir, mode, "labels")

outputImgDir = os.path.join(base_out_dir, mode, "images")
outputMaskDir = os.path.join(base_out_dir, mode, "occ_labels")
occlusionMaskDir = os.path.join(base_out_dir, mode, "occ_masks")
output_ori_img_mask = os.path.join(base_out_dir, mode, "labels")
#output_ori_img_dir = os.path.join(base_out_dir, mode, "clean_images")


# Prepare the samples to occlude and the occluders to use
woc_file, im_sample, im_files = prepare_data_to_occlude(oc_img_path, img_path, ratio)

count = 0
countup = 0

# Occlude samples
for img_file in im_sample:
    try:
        oc_file = woc_file.pop() 
        if oc_file.split('_')[0] in {'treet'}:
            count = count + 1
        if oc_file.split('_')[0] in {'tree'}:
            countup = countup + 1
        
        occlude_images(img_file, img_path, mask_path, oc_file, oc_img_path, oc_mask_path)
    
    except Exception as e:
        print(e)
        print(f'Failed: {img_file} , {oc_file}')

        cmd1 = 'cp -r ' + ' ' + os.path.join(img_path,img_file) + ' ' +  os.path.join(outputImgDir,img_file)
        cmd2 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(outputMaskDir,img_file.split('.')[0]+'.png')
        cmd3 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(output_ori_img_mask,img_file.split('.')[0]+'.png')
        os.system(cmd1)
        os.system(cmd2)
        os.system(cmd3)

        im = cv2.imread(os.path.join(img_path,img_file))
        im.shape[0:2]
        black_img = np.zeros(im.shape[0:2], dtype = np.uint8)
        cv2.imwrite(os.path.join(occlusionMaskDir, img_file.split('.')[0]+'.png'),black_img)
        
print('Total trunk trees: ',count, 'and hanging vegetation ', countup)


# If the sample is not occluded, fill output directoties with unoccluded samples
for img_file in im_files:
    if not img_file in im_sample:
        cmd1 = 'cp -r ' + ' ' + os.path.join(img_path,img_file) + ' ' +  os.path.join(outputImgDir,img_file)
        cmd2 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(outputMaskDir,img_file.split('.')[0]+'.png')

        im = cv2.imread(os.path.join(img_path,img_file))
        im.shape[0:2]
        black_img = np.zeros(im.shape[0:2], dtype = np.uint8)
        cv2.imwrite(os.path.join(occlusionMaskDir, img_file.split('.')[0]+'.png'),black_img)
        #cmd3 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(occlusionMaskDir,img_file.split('.')[0]+'.png')

        os.system(cmd1)
        os.system(cmd2)
        #os.system(cmd3)
    
    # Fill output_ori_img_mask folder with original labels (without occlusion)
    cmd4 = 'cp -r ' + ' ' + os.path.join(mask_path,img_file.split('.')[0]+'.png') + ' ' + os.path.join(output_ori_img_mask,img_file.split('.')[0]+'.png')
    os.system(cmd4)