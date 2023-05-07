import torch
from modules.model import MVLBertForPretraining
from modules.config import MVLBertPretrainConfig
from modules.logger import setup_logger
import time
from torch.utils.data import DataLoader, Dataset
import os
import _pickle as cPickle
import random
from transformers import BertTokenizer
import numpy as np
from utils import print_obj
import argparse


class RGCPretrainData(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.data_root = os.path.join('./dataset/RGC/', split)
        # self.data_path = os.path.join(self.data_root, split + '_img_idx2path.pkl')
        self.data_path = os.path.join(self.data_root, split + '_img_idx2path.pkl')
        self.img_idx2path = cPickle.load(open(self.data_path, 'rb'))
        self.img_num = len(self.img_idx2path)
        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
        self.max_length = self.config.max_length
        print("load data index information from:", self.data_path)

        # read the images and captions into memory if use_cache is True
        self.use_cache = False

    def __len__(self):
        return self.img_num

    def save_data_in_cache(self):
        assert self.tokenizer.eos_token is not None, 'please add the eos token for the tokenizer.'
        # self.cached_images = []
        self.cached_images = np.zeros((self.img_num, 3, 224, 224), dtype=np.float32)
        self.cached_caps_tokens = []
        self.cached_caps = []
        self.cached_img_ids = []
        self.cached_cap_ids = []

        print("saving images in memory")
        for idx in range(self.img_num):
            img_path = self.img_idx2path[idx]
            im_np, caption, img_id, cap_id = cPickle.load(open(img_path, 'rb'))
            # self.cached_images.append(im_np)
            self.cached_images[idx, :, :, :] = im_np
            self.cached_caps.append(caption)
            caption_with_end = caption + ' ' + self.tokenizer.eos_token
            caption_tokens = self.tokenizer.tokenize(caption_with_end)
            self.cached_caps_tokens.append(caption_tokens)

            self.cached_img_ids.append(img_id)
            self.cached_cap_ids.append(cap_id)

        self.use_cache = True
        return


    def get_data_by_idx(self, index):
        if self.use_cache:
            return self.cached_images[index], self.cached_caps[index], self.cached_caps_tokens[index], \
                   self.cached_img_ids[index], self.cached_cap_ids[index]
        else:
            img_path = self.img_idx2path[index]
            im_np, caption, img_id, cap_id = cPickle.load(open(img_path, 'rb'))
            caption_with_end = caption + ' ' + self.tokenizer.eos_token
            caption_tokens = self.tokenizer.tokenize(caption_with_end)
            return im_np, caption, caption_tokens, img_id, cap_id


    def __getitem__(self, index):
        img_path = self.img_idx2path[index]
        im_np, caption, caption_tokens, img_id, cap_id = self.get_data_by_idx(index)

        _p = random.random()
        # ITM
        ITM_label = 1
        if _p < 0.5 or not self.config.ITM_task:
            ITM_label = 1
        else:
            ITM_label = 0
            # ITM
            rand_index = random.randrange(0, self.img_num)
            rand_img_np, rand_img_caption, rand_img_caption_tokens, rand_img_id, rand_cap_id = self.get_data_by_idx(
                rand_index)

            while rand_index == index or cap_id == rand_cap_id:
                rand_index = random.randrange(0, self.img_num)
                rand_img_np, rand_img_caption, rand_img_caption_tokens, rand_img_id, rand_cap_id = self.get_data_by_idx(
                    rand_index)

            if random.random() < 0.5:
                # replace the image with another with different captions
                im_np = rand_img_np
            else:
                # or replace the captions with negative sample.
                caption_tokens = rand_img_caption_tokens

        # add [END] token
        # caption_tokens = self.basictokenizer.tokenize(caption)
        # caption_tokens = caption_tokens + [self.tokenizer.eos_token]
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



def pretrain_MVLBert(model, pretrain_dataloader, epochs=500, model_name='./checkpoints/bert-base', optim=None, save_freq=100, logger=None):
    # bert-large: lr=1e-5, base: lr=4*1e-5(remain test)
    model.cuda()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=model.config.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0001) if optim is None else optim
    optimizer.zero_grad()

    model.train()

    for epoch in range(epochs):
        t = time.time()
        n = len(pretrain_dataloader)
        total_loss = 0

        for i_, (image, caption_masked, caption_label, ITM_label) in enumerate(pretrain_dataloader):
            image = image.cuda()
            caption_masked = caption_masked.cuda()
            caption_label = caption_label.cuda()
            ITM_label = ITM_label.cuda()
            loss = model(image, caption_masked, caption_label, ITM_label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * image.shape[0]
        if logger is not None:
            logger.info(f"epoch: {epoch}, using time: {time.time() - t}, loss: {total_loss / n}")
        else:
            print("epoch:", epoch, "using time:", time.time() - t, "loss:", total_loss / n)
        model.save_pretrained(model_name)
        if epoch % save_freq == 0 and epoch > 0:
            model.save_pretrained(model_name+ '-' +str(epoch))

    return


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--batch', help='batch size for training and testing', default=32, type=int)
    parser.add_argument('--epochs', help='epochs', default=605, type=int)
    parser.add_argument('--lr', help='learning rate', default=None, type=float)
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear', 'resnet50', 'patch_resnet50', 'swintransformer', 'visiontransformer', 'vit']
                        , type=str)
    parser.add_argument('--save_model_name', help='name of model to be saved', type=str, default='resnet101-bert-base')
    parser.add_argument('--max_length', help='max_length of caption', type=int, default=80)
    parser.add_argument('--use_cache', help='whether to save images in memory', action='store_true')
    parser.add_argument('--pretrained_path', help='continue pretraining from a pretrained model', type=str, default=None)
    parser.add_argument('--ITM', help='whether to use ITM as pretraining tasks', action='store_true')
    parser.add_argument('--NOT_MLM', help='whether to close MLM as pretraining tasks', action='store_true')
    parser.add_argument('--save_freq', help='frequency of save',  default=100, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    # os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.cuda.set_device(args.device)
    config = MVLBertPretrainConfig.from_pretrained('bert-base-uncased')
    config.conv = args.conv
    config.max_length = args.max_length

    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
    tokenizer.add_special_tokens({'eos_token': '[END]'})
    config.update_special_tokens(tokenizer)
    if args.ITM:
        config.ITM_task = True

    if args.NOT_MLM:
        config.MLM_task = False

    if args.lr is not None:
        config.lr = args.lr
    print_obj(config)

    if args.pretrained_path is not None:
        print("load from checkpoints:", args.pretrained_path)
        model = MVLBertForPretraining.from_pretrained(args.pretrained_path, config=config)
    else:
        model = MVLBertForPretraining(config)
    # model = model.from_pretrained('./checkpoints/' + args.save_model_name, config=config)

    pretrain_dataset = RGCPretrainData(config)

    global logger
    time_tmp = time.asctime(time.localtime(time.time()))
    logger = setup_logger(__name__, f"log", 0, f"{args.conv}-RGC-{time_tmp.replace(':', '-')}.txt")

    if args.use_cache:
        pretrain_dataset.save_data_in_cache()

    pretrain_dataloader = DataLoader(pretrain_dataset, batch_size=args.batch, num_workers=8, shuffle=True)
    print("args:", args)
    model_name = './checkpoints/' + args.save_model_name
    if not os.path.exists(model_name):
        os.mkdir(model_name)
    pretrain_MVLBert(model, pretrain_dataloader, args.epochs, model_name=model_name, save_freq=args.save_freq, logger=logger)

