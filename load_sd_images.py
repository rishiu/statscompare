import numpy as np
from PIL import Image
import sys
from utils import file_to_dict, get_common_synsets
from stats import get_avg_fft, get_wavelet_coeffs, fit_power_law, fit_gen_gaussian
import matplotlib.pyplot as plt

def pre_process_image(img):
    I = Image.fromarray(img).resize((256,256)).crop((16,16,240,240))
    return np.array(I)

def get_imgs_from_id(id, fname, data_dir):
    map_dict = file_to_dict(fname)
    class_name = map_dict["n"+str(id)]

    imgs = []
    for i in range(10):
        img = np.array(Image.open(data_dir+class_name+str(i)+".jpg").convert('L'))
        imgs.append(img)
    return imgs

def get_class_fft(id, data_dir, map_fname, shape=None):
    imgs = get_imgs_from_id(id, map_fname, data_dir)

    shape = imgs[0].shape[:2] if shape is None else shape
    fft = get_avg_fft(imgs, shape=shape)

    return fft

def fit_fft_power_law(fft, shape):
    fft_1 = fft[:shape/2,shape/2]
    fft_2 = fft[shape/2+1:,shape/2]
    fft_3 = fft[shape/2,:shape/2]
    fft_4 = fft[shape/2,shape/2+1:]
    ffts = [fft_1, fft_2, fft_3, fft_4]

    xx = np.arange(0,shape/2)

    As = []
    gs = []
    for i in range(4):
        A, g = fit_power_law(xx,ffts[i])
        As.append(A)
        gs.append(g)
    
    fig, ax = plt.subplots(4)

    for i in range(4):
        ax[i].plot(xx,ffts[i])
        yy = As[i] / (xx**gs[i])
        ax[i].plot(xx,yy)
    plt.show()

def test(fname):
    common_synsets = get_common_synsets(fname)

    for synset in common_synsets:
        fft = get_class_fft(synset, "./bilderjpg/", "./map_clsloc.txt", shape=(224,224))
        fit_fft_power_law(fft)
        break

if __name__ == "__main__":
    test(sys.argv[1])