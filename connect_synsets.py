import os
import sys
import json

def file_to_file(fname, out_fname):
    map_dict = {}
    with open(fname, "r") as map_file:
        for line in map_file:
             num, cls_num, name = line.split(" ")
             map_dict[num] = (cls_num.strip(), name.strip())
    with open(out_fname, "w") as out_file:
        json.dump(map_dict, out_file)

def file_to_dist(fname):
    with open(fname, "r") as dict_file:
        data = json.load(dict_file)
        return data

if __name__ == "__main__":
    file_to_file("map_clsloc.txt", "clsloc_dict.txt")
    exit(0)

    if len(sys.argv) < 4:
        exit("Need paths to SD dataset, ImageNet, and output file")
    sd_dataset = sys.argv[1]
    imagenet = sys.argv[2]
    num_to_class_file = sys.argv[3]
    output_file = sys.argv[4]
    
    num_to_class = file_to_dist(num_to_class_file)

    sd_nums = []
    for file in os.listdir(sd_dataset):
        if file.find("txt") >= 0:
            with open(sd_dataset+file, "r") as word_file:
                wordnet_num = int(word_file.readline().strip()[1:])
                sd_nums.append(wordnet_num)

    in_datadir = imagenet + "Data/CLS-LOC/"

    common_nums = []
    for phase in ["train/"]:
        data_dir = in_datadir + phase

        for file in os.listdir(data_dir):
            for num in sd_nums:
                if file.find(str(num)) >= 0:
                    common_nums.append(num)
    with open(output_file, "w") as out_file:
        joined = '\n'.join(map(str, common_nums))
        out_file.write(joined)

