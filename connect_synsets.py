import os
import sys

if __name__ == "__main__":
    if len(sys.argv) < 4:
        exit("Need paths to SD dataset, ImageNet, and output file")
    sd_dataset = sys.argv[1]
    imagenet = sys.argv[2]
    output_file = sys.argv[3]

    sd_nums = []
    for file in os.listdir(sd_dataset):
        if file.find("txt") >= 0:
            with open(file, "r") as word_file:
                wordnet_num = int(word_file.readline())
                sd_nums.append(wordnet_num)

    in_datadir = imagenet + "Data/CLS-LOC/"

    common_nums = []
    for phase in ["train/","test/","valid/"]:
        data_dir = in_datadir + phase

        for file in os.listdir(data_dir):
            for num in sd_nums:
                if file.find(str(num)) >= 0:
                    common_nums.append(num)
    with open(output_file, "w") as out_file:
        joined = '\n'.join(common_nums)
        out_file.write(joined)

