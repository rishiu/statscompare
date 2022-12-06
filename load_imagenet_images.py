import numpy as np
from PIL import Image
import os
import sys
from utils import pre_process_image, get_common_synsets
from stats import get_avg_fft, get_wavelet_coeffs, fit_power_law, fit_gen_gaussian
import matplotlib.pyplot as plt

def get_imgs_from_id(id, data_dir):
    imgs = []

    for file in os.listdir(data_dir+"n0"+str(id)):
        img = np.array(Image.open(file).convert('L'))
        img = pre_process_image(img)
        imgs.append(img)
    return imgs

def get_class_fft(id, data_dir, shape=None):
    imgs = get_imgs_from_id(id, data_dir)

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def get_class_wmm(id, data_dir, height, order):
    imgs = get_imgs_from_id(id, data_dir)

    pyr_coeffs = {}
    for band in range(order+1):
        for h in range(height):
            pyr_coeffs[(h,band)] = []

    for img in imgs:
        wmm_coeffs = get_wavelet_coeffs(img, height=3)
        for key in wmm_coeffs.keys():
            pyr_coeffs[key].extend(wmm_coeffs[key])
    
    return pyr_coeffs

def fit_wmm(coeffs, height, order):
    params = {}
    for band in range(order+1):
        for h in range(height):
            y,binEdges=np.histogram(coeffs[(h,band)],bins=200)
            y = y.astype(np.float64)
            bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
            
            y[y<=0] = 1.
            y = np.log(y)
            y /= np.max(y)

            s, p = fit_gen_gaussian(bincenters / len(bincenters) / 2, y)
            params[(h,band)] = (s,p)
    return params

def fit_fft_power_law(fft, shape):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(np.arange(shape), np.arange(shape)) 
    
    fft = np.log(np.square(np.abs(fft)))
    
    s2 = int(shape/2)
    fft_1 = fft[:s2-1,s2]
    fft_2 = fft[s2+1:,s2]
    fft_3 = fft[s2,:s2-1]
    fft_4 = fft[s2,s2+1:]
    ffts = [fft_1, fft_2, fft_3, fft_4]

    xx = np.arange(1,shape/2)
    xx_ = np.arange(shape/2-1,0,-1)

    As = []
    gs = []
    for i in range(4):
        if i % 2 == 1:
            A, g = fit_power_law(xx,ffts[i])
        else:
            A, g = fit_power_law(xx_,ffts[i])
        As.append(A)
        gs.append(g)

    return As, gs

def test(fname):
    common_synsets = get_common_synsets(fname)

    for synset in common_synsets:
        fft = get_class_fft(synset, "../bilderjpg/", "./clsloc_dict.txt", shape=(224,224))
        A, g = fit_fft_power_law(fft, shape=224)

        coeffs = get_class_wmm(synset, "../bilderjpg/", "./clsloc_dict.txt", height=5, order=4)
        params = fit_wmm(coeffs, height=5, order=4)
        print(params)
        print(A,g)

if __name__ == "__main__":
    test(sys.argv[1])