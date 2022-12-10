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

    sA = [[],[],[],[]]
    sg = [[],[],[],[]]
    ss = [[],[],[],[]]
    sp = [[],[],[],[]]

    rA = [[],[],[],[]]
    rg = [[],[],[],[]]
    rs = [[],[],[],[]]
    rp = [[],[],[],[]]
    
    print(sA)

    for synset in synsets:
        skip = False
        sdata = syn_data[str(synset)]
        rdata = real_data[str(synset)]
        if len(sdata["params"].keys()) == 0 or len(rdata["params"].keys()) == 0:
            continue
        for k in range(2):
            if sdata["A"] == -100 or rdata["A"] == -100:
                skip = True
                continue
            sA[k].append(0.5*(sdata["A"][k*2] + sdata["A"][k*2+1]))      
            sg[k].append(0.5*(sdata["g"][k*2] + sdata["g"][k*2+1]))
            rA[k].append(0.5*(rdata["A"][k*2] + sdata["A"][k*2+1]))
            rg[k].append(0.5*(rdata["g"][k+2] + sdata["g"][k*2+1]))
        for i in range(4):
            if skip:
                continue
            #sA[i].append(sdata["A"][i])
            #sg[i].append(sdata["g"][i])
            ss[i].append(sdata["params"][str(i)][0])           
            sp[i].append(sdata["params"][str(i)][1])

            #rA[i].append(rdata["A"][i])
            #rg[i].append(rdata["g"][i])
            rs[i].append(rdata["params"][str(i)][0])           
            rp[i].append(rdata["params"][str(i)][1])

    fig, ax = plt.subplots(2,4)
    fig.set_size_inches(10,10)
        
    colors = ['r', 'g', 'b', 'm']
    arr_of_in = [sp,rp]
    s = arr_of_in[0]
    r = arr_of_in[1]
    for i in range(4):
        ax[0][i].hist(s[i], bins=100, range=(np.min(s[i]) * 0.9,np.max(s[i])), color=colors[i], label="Synthetic "+str(i))
        #ax[0][i].axvline(x=np.median(s[i]),linestyle='--',color='yellow')
        ax[0][i].axvline(x=np.mean(s[i]),linestyle='-',label="Mean: "+str(np.mean(s[i])))
        ax[1][i].hist(r[i], bins=100, range=(np.min(s[i])*0.9,np.max(s[i])), color=colors[i], label="Real" + str(i))
        #ax[1][i].axvline(x=np.median(r[i]),linestyle='--',color='yellow')
        ax[1][i].axvline(x=np.mean(r[i]),linestyle='-',label="Mean: "+str(np.mean(r[i])))
    for i in range(4):
        ax[0][i].legend()
        ax[1][i].legend()
        ax[0][i].set_ylim(0,60)
        ax[1][i].set_ylim(0,60)
    plt.show()
    plt.savefig("p_hist2.jpg")
    
if __name__ == "__main__":
    plot_hist("./sd_latest.json", "./imn_latest.json", "out.txt")
