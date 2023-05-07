import numpy as np
import os
import json
from PIL import Image
import _pickle as cPickle
import re

def preprocess_pretrain():
    image_size = (224, 224)
    data_root = './dataset/RGC'

    # saved images in numpy_array and natural language captions
    save_root = ['./dataset/RGC/train/', './dataset/RGC/test/']
    save_pkl = ['train_img_idx2path.pkl', 'test_img_idx2path.pkl']

    ann_path = os.path.join(data_root, 'RGC_annotation.json')
    RGC_data = json.load(open(ann_path, 'r'))
    split_set = ['train', 'test']
    for k in range(2):
        img_idx2path = {}
        split_root = save_root[k]
        if not os.path.exists(split_root):
            os.mkdir(split_root)
        for i, entry in enumerate(RGC_data[split_set[k]]):

            image_path_rlt = entry['image']
            img_id = entry['img_id']
            caption = entry['caption']
            caption_id = entry['cap_id']

            image_path = os.path.join(data_root, 'images', image_path_rlt)
            im = Image.open(image_path, 'r')
            im = im.resize(image_size)
            im = im.convert('RGB') # for MIMIC-CXR, the image has only one channel

            im_np = np.array(im, dtype=np.float32)
            im_np = np.transpose(im_np, (2, 0, 1))  # channel, h, w
            for c in range(im_np.shape[0]):
                im_np[c] = (im_np[c] - np.mean(im_np[c])) / np.var(im_np[c])
                # im_np = (im_np - im_np.min()) / (im_np.max() - im_np.min())
            # save_path = os.path.join(split_root, str(i) + '.pkl')
            save_path = os.path.join(split_root, str(len(img_idx2path)) + '.pkl')
            cPickle.dump([im_np, caption, img_id, caption_id], open(save_path, 'wb'))
            img_idx2path[i] = save_path

        save_path = os.path.join(split_root, save_pkl[k])
        print("save image data to", save_path)
        cPickle.dump(img_idx2path, open(save_path, 'wb'))

        print(k, "total number of images:", len(img_idx2path))



if __name__ == '__main__':
    preprocess_pretrain()
    # remove_mimic
    # preprocess_pretrain()



