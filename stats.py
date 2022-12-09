import numpy as np
from PIL import Image
import pyrtools as pt
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import scipy
from scipy.optimize import curve_fit

def get_avg_fft(imgs, shape):
    avg_fft = np.zeros(shape).astype(np.complex128)
    if len(imgs) == 0:
        return avg_fft
    for img in imgs:
        if not np.any(img):
            continue   
        img = np.array(Image.fromarray(img).convert('L'))

        fft = np.fft.fftn(img, s=shape)
        fft_viz = np.fft.fftshift(fft)
        avg_fft += fft_viz
    avg_fft /= len(imgs)
    return avg_fft

def gen_gaussian(xx_, s, p): # From https://www.cns.nyu.edu/pub/eero/simoncelli05a-preprint.pdf
        num = np.exp(-np.abs(xx_ / s)**p)
        denom = 2 * (s / p) * scipy.special.gamma(1 / p)
        return num#(num / denom)

def get_contour_plot(fft):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.contour3D(fft)
    ax.view_init(-90,0,0)
    fig.canvas.draw()

    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def get_wavelet_coeffs(img, height, order):
    # Return data in a way that can be plotted easily
    # if order not in [0,1,3,5]:
    #     print("order must be one of 0, 1, 3, or 5!")
    #     return
    pyr = pt.pyramids.SteerablePyramidFreq(img, height=height, order=order)
    types = []
    for i in range(height):
        for j in range(order+1):
            types.append((i,j))
    coeff_dict = {}
    for band in range(order+1):
        band_coeffs = []
        for h in range(height):
            coeffs = pyr.pyr_coeffs[(h,band)].flatten()
            band_coeffs.extend(list(coeffs))
        coeff_dict[band] = band_coeffs
    return coeff_dict


def fit_power_law2d(xx, yy, zz):
    def power_law(X, A, gamma):
        xx_, yy_ = X
        zz_ = A / ((np.sqrt(xx_**2 + yy_**2))**gamma)
        return zz_

    popt, pcov = curve_fit(power_law, [xx,yy], zz)

    return popt

def fit_power_law(xx, yy):
    def power_law(xx_, A, gamma):
        yy_ = A / (xx_**gamma + 1e-15)
        return yy_

    popt, pcov = curve_fit(power_law, xx, yy, bounds=(0,[150000,5]))

    return popt

def fit_gen_gaussian(xx, yy):
    def gen_gaussian(xx_, s, p): # From https://www.cns.nyu.edu/pub/eero/simoncelli05a-preprint.pdf
        num = np.exp(-np.abs(xx_ / s)**p)
        #denom = 2 * (s / p) * scipy.special.gamma(1 / p)
        return num# / denom)

    popt, pcov = curve_fit(gen_gaussian, xx, yy, bounds=([0,0],[200000,2]))

    return popt

def fit_fft_power_law(fft, shape, plot=False):
    xx, yy = np.meshgrid(np.arange(shape), np.arange(shape)) 
    
    fft = np.abs(fft)
    
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
    
    if plot:
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

def fit_wmm(coeffs, order, plot=False):
    params = {}
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
        
        y2 = gen_gaussian(bc, s, p)
    
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(bc, y)
            ax.plot(bc, y2)
            
            plt.show()
            plt.savefig("wmm.jpg")
    return params