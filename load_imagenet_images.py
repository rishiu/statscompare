import numpy as np
from PIL import Image
import os
import sys
from utils import pre_process_image, get_common_synsets
from stats import get_avg_fft, get_wavelet_coeffs, fit_power_law, fit_gen_gaussian, gen_gaussian
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
        if count == 10:
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

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def get_class_wmm(id, data_dir, height, order):
    imgs = get_imgs_from_id(id, data_dir)

    pyr_coeffs = {}
    for band in range(order+1):
        for h in range(height):
            pyr_coeffs[band] = []

    for img in imgs:
        wmm_coeffs = get_wavelet_coeffs(img, height=height, order=order)
        for key in wmm_coeffs.keys():
            pyr_coeffs[key].extend(wmm_coeffs[key])
    
    return pyr_coeffs

def fit_wmm(coeffs, height, order):
    params = {}
    if len(coeffs[list(coeffs.keys())[0]]) == 0:
        return params
    for band in range(order+1):
        y,binEdges=np.histogram(coeffs[band],bins=100)
        y = y.astype(np.float64)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        
        y[y<=0] = 1.
        y = np.log(y)
        y /= np.max(y)
        
        bc = bincenters# / np.max(bincenters)

        s, p = fit_gen_gaussian(bc, y)
        params[band] = (s,p)
        
        #y2 = gen_gaussian(bc, s, p)
    
        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.plot(bc, y)
        #ax.plot(bc, y2)
        
        #plt.show()
        #plt.savefig("wmm.jpg")
    return params

def fit_fft_power_law(fft, shape):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(np.arange(shape), np.arange(shape)) 
    
    fft = np.abs(fft)**2
    
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
        
    fig, ax = plt.subplots(4)

    for i in range(4):
        if i % 2 == 1:
            ax[i].plot(xx,ffts[i])
        else:
            ax[i].plot(xx_,ffts[i])            
        yy = As[i] / (xx**gs[i])
        ax[i].plot(xx,yy)
    plt.savefig("test.jpg")

    return As, gs

def test(fname):
    common_synsets = get_common_synsets(fname)

    data_dict = {}
    for synset in tqdm(common_synsets):
        class_data = {}
        
        fft = get_class_fft(synset, "../ImageNet/Data/CLS-LOC/train/", shape=(224,224))
        A, g = fit_fft_power_law(fft, shape=224)

        coeffs = get_class_wmm(synset, "../ImageNet/Data/CLS-LOC/train/", height=5, order=4)
        params = fit_wmm(coeffs, height=5, order=4)
        
        class_data["A"] = A
        class_data["g"] = g
        class_data["params"] = params
        
        data_dict[synset] = class_data

    return data_dict
    
def test_all(fname):
    common_synsets = get_common_synsets(fname)
    
    all_imgs = []
    for synset in tqdm(common_synsets):
        imgs = get_imgs_from_id(synset, "../ImageNet/Data/CLS-LOC/train/")
        all_imgs.extend(imgs)
        #break
    
    fft = get_avg_fft(all_imgs, shape=(224,224))
    A, g = fit_fft_power_law(fft, shape=224)
    print(A,g)
    
    fft = np.log(np.square(np.abs(fft)))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    xx, yy = np.meshgrid(np.arange(224), np.arange(224))

    ax.contour3D(xx,yy,fft,levels=10)
    ax.view_init(azim=0, elev=90)
    
    plt.savefig("imn_all_fft.jpg")

if __name__ == "__main__":
    test_all(sys.argv[1])
    #data = test(sys.argv[1])
    #with open("imn_output_ps.json", "w") as out_file:
    #    json.dump(data, out_file)
