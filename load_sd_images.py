import numpy as np
from PIL import Image
import sys
from utils import file_to_dict, get_common_synsets, pre_process_image
from stats import get_avg_fft, get_wavelet_coeffs, fit_fft_power_law, fit_wmm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from tqdm import tqdm
import json

def get_imgs_from_id(id, fname, data_dir):
    map_dict = file_to_dict(fname)
    id_s = "{:08d}".format(id)
    try:
        class_name = map_dict["n"+id_s]
        #print(class_name)
    except:
        print("Class "+str(id)+" not found")
        return []

    imgs = []
    for i in range(10):
        img = np.array(Image.open(data_dir+class_name[1]+str(i)+".jpg").convert('L'))
        img = pre_process_image(img)
        imgs.append(img)
    return imgs

def get_class_fft(id, data_dir, map_fname, shape=None):
    imgs = get_imgs_from_id(id, map_fname, data_dir)

    if len(imgs) == 0:
        return None

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def get_class_wmm(id, data_dir, map_fname, height, order):
    imgs = get_imgs_from_id(id, map_fname, data_dir)

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

        fft = get_class_fft(synset, "../bilderjpg/", "./clsloc_dict.txt", shape=(224,224))
        A, g = -100, -100
        if fft is not None:            
            A, g = fit_fft_power_law(fft, shape=224)

        coeffs = get_class_wmm(synset, "../bilderjpg/", "./clsloc_dict.txt", height=height, order=order)
        params = {idx: (-1,-1) for idx in range(order)}
        if coeffs is not None:
            params = fit_wmm(coeffs, order=order) 
        
        #class_data["fft"] = list(fft)
        #class_data["coeffs"] = coeffs
        
        class_data["A"] = A
        class_data["g"] = g
        class_data["params"] = params
        
        data_dict[synset] = class_data
    
    return data_dict
    
def test_all(fname):
    common_synsets = get_common_synsets(fname)
    
    all_imgs = []
    for synset in tqdm(common_synsets):
        imgs = get_imgs_from_id(synset, "./clsloc_dict.txt", "../bilderjpg/")
        all_imgs.extend(imgs)
        #break
    
    fft = get_avg_fft(all_imgs, shape=(224,224))
    A, g = fit_fft_power_law(fft, shape=224)
    print(A,g)
    
    fft = np.log(np.square(np.abs(fft)))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xx, yy = np.meshgrid(np.arange(224), np.arange(224))

    ax.contour3D(xx,yy,fft)
    
    plt.savefig("sd_all_fft.jpg")

if __name__ == "__main__":
    test_all(sys.argv[1])
    #data = test(sys.argv[1])
    #with open("sd_output_ps.json", "w") as out_file:
    #    json.dump(data, out_file)
