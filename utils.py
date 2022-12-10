import json
import os
from PIL import Image
import numpy as np

def file_to_file(fname, out_fname):
    map_dict = {}
    with open(fname, "r") as map_file:
        for line in map_file:
             num, cls_num, name = line.split(" ")
             map_dict[num] = (cls_num.strip(), name.strip())
    with open(out_fname, "w") as out_file:
        json.dump(map_dict, out_file)

def file_to_file2(fname, out_fname):
    map_dict = {}
    with open(fname, "r") as map_file:
        for line in map_file:
             num, cls_num, name = line.split(" ")
             map_dict[name.strip()] = (cls_num.strip(), num.strip())
    with open(out_fname, "w") as out_file:
        json.dump(map_dict, out_file)

def file_to_dict(fname):
    with open(fname, "r") as dict_file:
        data = json.load(dict_file)
        return data
        
def get_names_from_file(fname, map_fname):
    names = []
    map_dict = file_to_dict(map_fname)
    with open(fname, 'r') as in_file:
        for line in in_file:
            synset = line.strip()
            name = map_dict[synset][1]
            names.append((synset,name))
    return names
    
def check_grayscale(data_dir, fname):
    synsets = get_imagenet100_synsets(fname)
    for synset in synsets:
        syn_id = "{:08d}".format(synset)
        for f in os.listdir(data_dir+"n"+syn_id):
            base_dir = data_dir+"n"+syn_id+"/"
            img = np.array(Image.open(base_dir+f))
            if len(img.shape) < 3 or img.shape[2] == 1:
                print(img.shape)
                print("Found it! " + f)

def get_common_synsets(fname):
    synsets = []
    with open(fname, "r") as common_file:
        for line in common_file:
            synsets.append(int(line.strip()))
    return synsets
    
def get_imagenet100_synsets(fname):
    synsets = []
    with open(fname, "r") as imn100_file:
        for line in imn100_file:
            synsets.append(int(line.strip()[1:]))
    return synsets

def pre_process_image(img):
    img = img.astype(float) / 255
    I = Image.fromarray(img).resize((256,256)).crop((16,16,240,240))
    return np.array(I)
