import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import file_to_dict, pre_process_image, get_imagenetsubset_synset_nums
from stats import farid_stats
import sklearn

def get_imn_imgs_from_id(id, data_dir):
    imgs = []

    id_s = "{:08d}".format(id)

    count = 0
    if not os.path.exists(data_dir+"n"+id_s):
        return []
    for file in os.listdir(data_dir+"n"+id_s):
        if count == 1000:
            break
        img = np.array(Image.open(data_dir+"n"+id_s+"/"+file).convert('L'))
        if img.shape[0] < 256 and img.shape[1] < 256:
            continue
        count += 1
        img = pre_process_image(img)
        imgs.append(img)
    return imgs

def get_sd_imgs_from_id(synset, fname, data_dir):
    map_dict = file_to_dict(fname)
    imgs = []
    syn_id = "n{:08d}".format(synset)
    syn_info = map_dict[syn_id]
    class_num = int(syn_info[0])
    class_name = syn_info[1].strip().lower()
    base_dir = data_dir+class_name+"/"
    files = os.listdir(base_dir)
    for f in files:
        img = np.array(Image.open(base_dir+f))
        img = rand_pre_process_image(img)
        imgs.append(img)
    return imgs

def compare(syn_fname, real_fname, map_fname, synsets_fname, height):
	synsets = get_imagenetsubset_synset_nums(synsets_fname)

	syn_data = []
	real_data = []
	for synset in synsets:

		# Real
		real_imgs = get_imn_imgs_from_id(synset, real_fname)
		for img in real_imgs:
			feat_vec = []
			for c in range(3):
				feat_vec.extend(farid_stats(img[:,:,c]))
			real_data.append(feat_vec)
		real_data = np.array(real_data)

		# Synthetic
		syn_imgs = get_sd_imgs_from_id(synset, syn_fname)
		for img in syn_imgs:
			feat_vec = []
			for c in range(3):
				feat_vec.extend(farid_stats(img[:,:,c]))
			syn_data.append(feat_vec)
		syn_data = np.array(syn_data)

		real_PCA = sklearn.decomposition.PCA(n_components=3)
		real_out = real_PCA.fit(real_data).transform(real_data)

		syn_PCA = sklearn.decomposition.PCA(n_components=3)
		syn_out = syn_PCA.fit(syn_data).transform(syn_data)

		np.save("output_higherorder/sd"+synset, syn_out)
		np.save("output_higherorder/imn"+synset, real_out)



