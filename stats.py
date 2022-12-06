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
    for img in imgs:   
        img = np.array(Image.fromarray(img).convert('L'))

        fft = np.fft.fftn(img, s=shape)
        fft_viz = np.fft.fftshift(fft)
        avg_fft += fft_viz
    avg_fft /= len(imgs)
    return avg_fft

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
            coeffs = pyr.pyr_coeffs[(h,band)]
            band_coeffs.extend(coeffs)
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
        denom = 2 * (s / p) * scipy.special.gamma(1 / p)
        return (num / denom)

    popt, pcov = curve_fit(gen_gaussian, xx, yy, bounds=([0,0.4],[10,0.8]))

    return popt

def test_power_law_fit():
    xx = np.arange(0,50).astype(np.float64)
    A = 100
    gamma = 2
    yy = A / (xx**gamma + 1e-15)
    print(yy)
    print(fit_power_law(xx,yy))
