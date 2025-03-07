import numpy as np
import cv2


#https://github.com/isarandi/synthetic-occlusion
def paste_over(im_src,occluder_mask, im_dst,dst_mask, center,occlusion_mask,randOcc, gsr, oc_type):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visie).
    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """
    
    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]]) #shape occlusion
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]]) #shape destination

    center = np.round(center).astype(np.int32) 
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst) #start cord x,y in dst img
    end_dst = np.clip(raw_end_dst, 0, width_height_dst) #start cord x,y in dst img
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] #region_dst limited by start and end coordinates, where occ is gonna be pasted

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    occluder_mask =occluder_mask[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]


    alpha = (region_src[..., 3:].astype(np.float32)/255) #alpha channel of occluder
    
    if randOcc:
        
        if np.random.rand()<0.3:
            alpha*=np.random.uniform(0.4, 0.7)
   

    #alpha blending edge processing
    kernel = np.ones((3,3),np.uint8) 
    
    no_ero ={"tree", "treet", "fenc"}
    if oc_type in no_ero:
        alpha = cv2.GaussianBlur(alpha,(gsr,gsr),0)
        alpha= np.expand_dims(alpha, axis=2)
    else:
        alpha = cv2.erode(alpha,kernel,iterations = 1) # does not work well for trees
        alpha = cv2.GaussianBlur(alpha,(gsr,gsr),0)
        alpha= np.expand_dims(alpha, axis=2)

    
    occlusion_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]= cv2.add(occlusion_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
        
    dst_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]=cv2.subtract(dst_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
    
    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = ((alpha * color_src) + ( (1 - alpha) * region_dst))

    return im_dst,dst_mask,occlusion_mask