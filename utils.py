import json
import os

def file_to_file(fname, out_fname):
    map_dict = {}
    with open(fname, "r") as map_file:
        for line in map_file:
             num, cls_num, name = line.split(" ")
             map_dict[num] = (cls_num.strip(), name.strip())
    with open(out_fname, "w") as out_file:
        json.dump(map_dict, out_file)

def file_to_dict(fname):
    with open(fname, "r") as dict_file:
        data = json.load(dict_file)
        return data

def get_common_synsets(fname):
    synsets = []
    with open(fname, "r") as common_file:
        for line in common_file:
            synsets.append(int(line.strip()))
    return synsets