import numpy as np
from PIL import Image
import sys
from utils import file_to_dict, get_common_synsets, pre_process_image
from stats import get_avg_fft, get_wavelet_coeffs, fit_power_law, fit_gen_gaussian, gen_gaussian
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

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def get_class_wmm(id, data_dir, map_fname, height, order):
    imgs = get_imgs_from_id(id, map_fname, data_dir)

    pyr_coeffs = {}
    for band in range(order+1):
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
        #print(np.max(coeffs[band]))
        y,binEdges=np.histogram(coeffs[band],bins=100)
        y = y.astype(np.float64)
        bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
        
        y[y<=0] = 1.
        y = np.log(y)
        y /= np.max(y)

        s, p = fit_gen_gaussian(bincenters, y)
        params[band] = (s,p)
        
        y2 = gen_gaussian(bincenters, s, p)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(bincenters, y)
        ax.plot(bincenters, y2)
        
        plt.show()
        plt.savefig("wmm.jpg")
        
        plt.close()
    return params

def fit_fft_power_law(fft, shape):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    xx, yy = np.meshgrid(np.arange(shape), np.arange(shape)) 
    
    fft = np.abs(fft)
    
    # ax.view_init(90,00,0)
    # ax.plot_wireframe(xx,yy,fft)
    # plt.savefig("test.jpg")
    # #return
    
    #print(shape/2, shape)
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
    plt.close()

    return As, gs

def test(fname):
    common_synsets = get_common_synsets(fname)

    data_dict = {}
    for synset in tqdm(common_synsets):
        fft = get_class_fft(synset, "../bilderjpg/", "./clsloc_dict.txt", shape=(224,224))
        A, g = fit_fft_power_law(fft, shape=224)

        coeffs = get_class_wmm(synset, "../bilderjpg/", "./clsloc_dict.txt", height=5, order=4)
        params = fit_wmm(coeffs, height=5, order=4)
        
        class_data = {}
        
        #class_data["fft"] = list(fft)
        #class_data["coeffs"] = coeffs
        class_data["A"] = A
        class_data["g"] = g
        class_data["params"] = params
        
        data_dict[synset] = class_data
    
    return data_dict

if __name__ == "__main__":
    data = test(sys.argv[1])
    with open("sd_output.json", "w") as out_file:
        json.dump(data, out_file)
