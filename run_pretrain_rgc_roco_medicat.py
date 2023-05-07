import torch
from modules.model import MVLBertForPretraining
from modules.config import MVLBertPretrainConfig
from modules.logger import setup_logger
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
import numpy as np
from PIL import Image
import _pickle as cPickle
import argparse, json, random
import time, os
from utils import print_obj
from run_pretrain import pretrain_MVLBert


class RGCROCOPretrainData(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config

        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
        self.max_length = self.config.max_length

        # RGC
        self.RGC_data_root = os.path.join('./dataset/RGC/', split)
        # self.data_path = os.path.join(self.data_root, split + '_img_idx2path.pkl')
        self.RGC_data_path = os.path.join(self.RGC_data_root, split + '_img_idx2path.pkl')
        self.RGC_img_idx2path = cPickle.load(open(self.RGC_data_path, 'rb'))
        self.RGC_img_num = len(self.RGC_img_idx2path)
        logger.info(f"load RGC data index information from: {self.RGC_data_path}")
        logger.info(f"RGC samples: {len(self.RGC_img_idx2path)}")

        # ROCO
        self.ROCO_data_root = './dataset/ROCO/data/train/radiology/'
        ROCO_caption_path = os.path.join(self.ROCO_data_root, 'captions.txt')
        logger.info(f"load ROCO captions from: {ROCO_caption_path}")
        ROCO_caption_file = open(ROCO_caption_path, 'r', encoding='utf-8')
        caption_data = ROCO_caption_file.readlines()
        self.ROCO_img_root = os.path.join(self.ROCO_data_root, 'images')
        img_names_real = os.listdir(self.ROCO_img_root)
        self.ROCO_processed_json = 'ROCO.json'
        self.ROCO_img_names_list = []
        self.ROCO_caption_list = []
        max_length_tmp = 0
        total_length = 0
        ROCO_json_path = os.path.join(self.ROCO_data_root, self.ROCO_processed_json)
        if not os.path.exists(ROCO_json_path):
            logger.info("First preprocess ROCO.")
            for data in caption_data:
                d = data.split('\t')
                img_name = d[0] + '.jpg'
                cap_len = len(d[1].split(' '))
                if cap_len > max_length_tmp:
                    max_length_tmp = cap_len
                total_length += cap_len
                img_path = 0
                if img_name in img_names_real:
                    try:
                        img_path = os.path.join(self.ROCO_img_root, img_name)
                        im = Image.open(img_path, 'r').convert('RGB')
                    except:
                        logger.warning(f"invalid image: {img_path}")
                        continue
                    self.ROCO_img_names_list.append(img_name)
                    self.ROCO_caption_list.append(d[1])
                    # if len(self.ROCO_img_names_list) % 10000 ==0 and len(self.ROCO_img_names_list) > 0:
                    #     print(f"already reading {len(self.ROCO_img_names_list)}")
            logger.info(f"ROCO max_length: {max_length_tmp}")
            logger.info(f"ROCO avg_length: {total_length / len(caption_data)}")
            json.dump([self.ROCO_img_names_list, self.ROCO_caption_list], open(ROCO_json_path, 'w'))
        else:
            logger.info("already preprecess. Load from " + ROCO_json_path)
            self.ROCO_img_names_list, self.ROCO_caption_list = json.load(open(ROCO_json_path, 'r'))


        self.ROCO_img_num = len(self.ROCO_img_names_list)
        logger.info(f"ROCO training sample number: {self.ROCO_img_num}")  # caption 65450, image 65404

        # MedICaT
        self.medicat_data_root = './dataset/medicat/release/'
        self.medicat_processed_json = 'medicat.json'
        json_path = os.path.join(self.medicat_data_root, self.medicat_processed_json)
        logger.info(f"load MedICaT Data from: {json_path}")
        self.medicat_data = json.load(open(json_path, 'r'))
        self.medicat_img_num = len(self.medicat_data)
        logger.info(f"total MedICaT training sample number: {self.medicat_img_num}")
        logger.info(f"total training sample number: {self.ROCO_img_num + self.RGC_img_num + self.medicat_img_num}")



    def __len__(self):
        return self.RGC_img_num + self.ROCO_img_num + self.medicat_img_num

    def get_data_by_idx(self, index):
        if index < self.RGC_img_num:
            img_path = self.RGC_img_idx2path[index]
            im_np, caption, img_id, cap_id = cPickle.load(open(img_path, 'rb'))
            return im_np, caption, img_id, cap_id
        elif index < self.RGC_img_num + self.ROCO_img_num:
            index -= self.RGC_img_num
            img_name = self.ROCO_img_names_list[index]
            caption = self.ROCO_caption_list[index]
            img_path = os.path.join(self.ROCO_img_root, img_name)
            image_size = (224, 224)
            im = Image.open(img_path, 'r').convert('RGB')
            im = im.resize(image_size)
            im_np = np.array(im, dtype=np.float32)
            im_np = np.transpose(im_np, (2, 0, 1))  # channel, h, w
            for c in range(im_np.shape[0]):
                im_np[c] = (im_np[c] - np.mean(im_np[c])) / np.var(im_np[c])

            # caption_tokens = self.tokenizer.tokenize(caption_with_end)
            return im_np, caption, index, index
        else:
            data = self.medicat_data[index - self.RGC_img_num - self.ROCO_img_num]
            id = index
            img_name = data['pdf_hash'] + '_' + data['fig_uri']
            img_path = os.path.join(self.medicat_data_root, 'figures', img_name)
            caption = data['s2_caption']
            image_size = (224, 224)
            im = Image.open(img_path, 'r').convert('RGB')
            im = im.resize(image_size)
            im_np = np.array(im, dtype=np.float32)
            im_np = np.transpose(im_np, (2, 0, 1))  # channel, h, w
            for c in range(im_np.shape[0]):
                im_np[c] = (im_np[c] - np.mean(im_np[c])) / np.var(im_np[c])

            # caption_tokens = self.tokenizer.tokenize(caption_with_end)
            return im_np, caption, index, index

    def __getitem__(self, index):
        im_np, caption, img_id, cap_id = self.get_data_by_idx(index)

        _p = random.random()
        # ITM
        ITM_label = 1
        if _p < 0.5 or not self.config.ITM_task:
            ITM_label = 1
        else:
            ITM_label = 0
            # ITM
            rand_index = random.randrange(0, self.__len__())
            rand_img_np, rand_img_caption, rand_img_id, rand_cap_id = self.get_data_by_idx(
                rand_index)

            while rand_index == index or cap_id == rand_cap_id:
                rand_index = random.randrange(0, self.__len__())
                rand_img_np, rand_img_caption, rand_img_id, rand_cap_id = self.get_data_by_idx(
                    rand_index)

            if random.random() < 0.5:
                # replace the image with another with different captions
                im_np = rand_img_np
            else:
                # or replace the captions with negative sample.
                caption = rand_img_caption

        # add [END] token
        caption_with_end = caption + ' ' + self.tokenizer.eos_token
        caption_tokens = self.tokenizer.tokenize(caption_with_end)
        if self.config.MLM_task and ITM_label == 1:
            caption_tokens, mlm_labels = self._random_mask_word(caption_tokens)
        else:
            mlm_labels = None

        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)

        if len(caption_ids) > self.max_length:
            # preserve [END]
            caption_ids = caption_ids[:(self.max_length - 1)] + [caption_ids[-1]]
            mlm_labels = mlm_labels[:(self.max_length-1)] + [mlm_labels[-1]] if mlm_labels is not None else None

        # set caption to max_caption_len
        caption_ids = np.array(caption_ids, dtype=np.int64)
        new_cap_ids = np.zeros(self.max_length, dtype=np.int64)
        new_cap_ids[:min(self.max_length, caption_ids.shape[0])] = caption_ids[:min(self.max_length, caption_ids.shape[0])]

        new_mlm_labels = np.ones(self.max_length, dtype=np.int64) * -100
        if mlm_labels is not None:
            new_mlm_labels[:min(self.max_length, caption_ids.shape[0])] = mlm_labels[:min(self.max_length, caption_ids.shape[0])]
        return torch.tensor(im_np), torch.tensor(new_cap_ids).long(), new_mlm_labels, ITM_label


    def _random_mask_word(self, tokens):
        token_len = len(tokens)
        output_tokens = [token for token in tokens]

        output_labels = [-100] * token_len
        # max mosked token is 10, each token with 20% to be masked.
        mask_prob = 0.2
        masked_token_num = min(10, max(1, round(token_len*mask_prob)))
        mask_idx_list = [i for i in range(token_len)]

        random.shuffle(mask_idx_list)
        mask_idx = mask_idx_list[:masked_token_num]
        for idx in mask_idx:
            _p = random.random()
            token = tokens[idx]
            if _p < 0.8:
                output_tokens[idx] = '[MASK]'
            # 10% randomly change token to random token
            elif _p < 0.9:
                output_tokens[idx] = random.choice(list(self.tokenizer.vocab.keys()))
            # -> rest 10% randomly keep current token, in which case the token will not be changed

            try:
                output_labels[idx] = self.tokenizer.vocab[token]
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_labels[idx] = self.tokenizer.vocab["[UNK]"]

        return output_tokens, output_labels



def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--batch', help='batch size for training and testing', default=32, type=int)
    parser.add_argument('--epochs', help='epochs', default=150, type=int)
    parser.add_argument('--lr', help='learning rate', default=None, type=float)
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear', 'resnet50', 'patch_resnet50', 'swintransformer',
                                 'visiontransformer', 'vit']
                        , type=str)
    parser.add_argument('--save_model_name', help='name of model to be saved', type=str)
    parser.add_argument('--max_length', help='max_length of caption', type=int, default=80)
    parser.add_argument('--use_cache', help='whether to save images in memory', action='store_true')
    parser.add_argument('--pretrained_path', help='continue pretraining from a pretrained model', type=str, default=None)
    parser.add_argument('--ITM', help='whether to use ITM as pretraining tasks', action='store_true')
    parser.add_argument('--NOT_MLM', help='whether to close MLM as pretraining tasks', action='store_true')
    parser.add_argument('--save_freq', help='frequency to save the model', action='store_true')
    args = parser.parse_args()
    return args

# python run_pretrain_roco_rgc.py --conv swintransformer --batch 32 --lr 4-e5  --ITM --save_model_name swin-roco+rgc-mlm-itm --epochs 801 --max_length 80


if __name__ == '__main__':
    args = parse_args()
    torch.cuda.set_device(args.device)
    config = MVLBertPretrainConfig.from_pretrained('bert-base-uncased')
    config.conv = args.conv
    config.max_length = args.max_length

    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
    config.update_special_tokens(tokenizer)
    if args.ITM:
        config.ITM_task = True

    if args.NOT_MLM:
        config.MLM_task = False

    if args.lr is not None:
        config.lr = args.lr

    global logger
    time_tmp = time.asctime(time.localtime(time.time()))
    logger = setup_logger(__name__, f"log", 0, f"{args.conv}-rgc+roco+medicat-{time_tmp.replace(':', '-')}.txt")

    print_obj(config)

    if args.pretrained_path is not None:
        print("load from checkpoints:", args.pretrained_path)
        model = MVLBertForPretraining.from_pretrained(args.pretrained_path, config=config)
    else:
        model = MVLBertForPretraining(config)

    pretrain_dataset = RGCROCOPretrainData(config)

    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch, num_workers=8, shuffle=True)
    logger.info(f"args: {args}")
    model_name = './checkpoints/' + args.save_model_name
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    pretrain_MVLBert(model, pretrain_dataloader, args.epochs, model_name=model_name, logger=logger, save_freq=args.save_freq)
