import numpy as np
import matplotlib.pyplot as plt
import json
from utils import get_common_synsets

def read_file(fname):
    with open(fname, "r") as in_file:
        data = json.load(in_file)
    return data

def plot_hist(syn_fname, real_fname, synset_fname):
    syn_data = read_file(syn_fname)
    real_data = read_file(real_fname)

    synsets = get_common_synsets(synset_fname)

    sA = [[]*4]
    sg = [[]*4]
    ss = [[]*4]
    sp = [[]*4]

    rA = [[]*4]
    rg = [[]*4]
    rs = [[]*4]
    rp = [[]*4]

    for synset in synsets:
        sdata = syn_data[synset]
        rdata = real_data[synset]
        for i in range(4):
            sA[i].append(sdata["A"][i])
            sg[i].append(sdata["g"][i])
            ss[i].append(sdata["params"][i][0])           
            sp[i].append(sdata["params"][i][1])

            rA[i].append(rdata["A"][i])
            rg[i].append(rdata["g"][i])
            rs[i].append(rdata["params"][i][0])           
            rp[i].append(rdata["params"][i][1])

    plt.hist(sA, color="red", label="Synthetic A")
    plt.hist(rA, color="red", label="Real A")
    plt.legend()
    plt.show()