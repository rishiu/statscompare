import numpy as np
from PIL import Image
import os
import sys
from utils import pre_process_image, get_common_synsets
from stats import get_avg_fft, get_wavelet_coeffs, fit_fft_power_law, fit_wmm
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

def get_imgs_from_id(id, data_dir):
    imgs = []

    id_s = "{:08d}".format(id)

    count = 0
    if not os.path.exists(data_dir+"n"+id_s):
        return []
    for file in os.listdir(data_dir+"n"+id_s):
        if count == 40:
            break
        img = np.array(Image.open(data_dir+"n"+id_s+"/"+file).convert('L'))
        if img.shape[0] < 256 and img.shape[1] < 256:
            continue
        count += 1
        img = pre_process_image(img)
        imgs.append(img)
    return imgs

def get_class_fft(id, data_dir, shape=None):
    imgs = get_imgs_from_id(id, data_dir)

    if len(imgs) == 0:
        return None

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def get_class_wmm(id, data_dir, height, order):
    imgs = get_imgs_from_id(id, data_dir)

    if len(imgs) == 0:
        return None

    pyr_coeffs = {}
    for band in range(order+1):
        pyr_coeffs[band] = []

    for img in imgs:
        wmm_coeffs = get_wavelet_coeffs(img, height=height, order=order)
        for key in wmm_coeffs.keys():
            pyr_coeffs[key].extend(wmm_coeffs[key])
    
    return pyr_coeffs

def test(fname):
    common_synsets = get_common_synsets(fname)
    order = 4
    height = 5

    data_dict = {}
    for synset in tqdm(common_synsets):
        class_data = {}
        
        fft = get_class_fft(synset, "../ImageNet/Data/CLS-LOC/train/", shape=(224,224))
        A, g = -100, -100
        if fft is not None:
            A, g = fit_fft_power_law(fft, shape=224)

        coeffs = get_class_wmm(synset, "../ImageNet/Data/CLS-LOC/train/", height=height, order=order)
        params = {idx: (-1,-1) for idx in range(order)}
        if coeffs is not None:
            params = fit_wmm(coeffs, order=order)
        
        class_data["A"] = A
        class_data["g"] = g
        class_data["params"] = params
        
        data_dict[synset] = class_data

    return data_dict
    
def test_all(fname, height, order):
    common_synsets = get_common_synsets(fname)
    
    all_imgs = []
    for synset in tqdm(common_synsets):
        imgs = get_imgs_from_id(synset, "../ImageNet/Data/CLS-LOC/train/")
        all_imgs.extend(imgs)
        #break
    As = []
    gs = []
    params_arr = []
    
    for img in all_imgs:
    	fft = get_avg_fft([img], shape=(224,224))
    	A, g = -100, -100
        if fft is not None:
            A, g = fit_fft_power_law(fft, shape=224)
    	As.append(A)
    	gs.append(g)
    	
    	wmm_coeffs = get_wavelet_coeffs(img, height=height, order=order)
    	params = {idx: (-1,-1) for idx in range(order)}
        if coeffs is not None:
            params = fit_wmm(coeffs, order=order)
        params_arr.append(params)
    
    return As, gs, params_arr

if __name__ == "__main__":
    #test_all(sys.argv[1])
    data = test(sys.argv[1])
    with open(sys.argv[2], "w") as out_file:
        json.dump(data, out_file)
