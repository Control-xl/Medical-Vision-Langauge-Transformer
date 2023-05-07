import torch
import time
import sys
import numpy as np
import argparse
import utils
import os
import torch.nn.functional as F
from transformers import BertTokenizer
import _pickle as cPickle
import random
from torch.utils.data import DataLoader, Dataset
from modules.model import MVLBertForRetrieval
from modules.config import MVLBertRetrieval
from modules.logger import setup_logger
import json

class RetrievalPretrainDataset(Dataset):
    # given a caption, find the corresponding image
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.split = split
        self.data_root = os.path.join('./dataset/RGC/', split)
        assert split in ['train', 'test'], 'Dataset has to be train set or test set'
        self.data_path = os.path.join(self.data_root, split + '_img_idx2path' + '.pkl')

        logger.info("load data index information from: {}".format(self.data_path))
        self.img_idx2path = cPickle.load(open(self.data_path, 'rb'))
        self.img_num = len(self.img_idx2path)
        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
        self.max_caption_len = self.config.max_length
        self.ITM_prob = 0.5
        self.use_cache = False

    def save_data_in_cache(self):
        assert self.tokenizer.eos_token is not None, 'please add the eos token for the tokenizer.'
        # self.cached_images = []
        self.cached_images = np.zeros((self.img_num, 3, 224, 224), dtype=np.float32)
        self.cached_caps_tokens = []
        self.cached_caps = []
        self.cached_img_ids = []
        self.cached_cap_ids = []

        logger.info("saving images in memory for {} set".format(self.split))
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


    def __len__(self):
        if self.split == 'train':
            return self.img_num
        return self.img_num**2

    def __getitem__(self, index):
        if self.split == 'train':
            im_np_gt, caption_gt, caption_tokens_gt, img_id, cap_id = self.get_data_by_idx(index)
            # ITM
            rand_index = random.randrange(0, self.img_num)
            rand_img_np, rand_img_caption, rand_img_caption_tokens, rand_img_id, rand_cap_id = self.get_data_by_idx(rand_index)

            while rand_index == index or cap_id == rand_cap_id:
                rand_index = random.randrange(0, self.img_num)
                rand_img_np, rand_img_caption, rand_img_caption_tokens, rand_img_id, rand_cap_id = self.get_data_by_idx(rand_index)

            if random.random() < 0.5:
                # replace the image with another with different captions
                im_np_neg = rand_img_np
                caption_tokens_neg = caption_tokens_gt
            else:
                # or replace the captions with negative sample.
                im_np_neg = im_np_gt
                caption_tokens_neg = rand_img_caption_tokens

            caption_ids_gt = self.tokenizer.convert_tokens_to_ids(caption_tokens_gt)
            caption_ids_neg = self.tokenizer.convert_tokens_to_ids(caption_tokens_neg)

            # preserve [END]
            if len(caption_ids_gt) > self.max_caption_len:
                caption_ids_gt = caption_ids_gt[:(self.max_caption_len - 1)] + [caption_ids_gt[-1]]
            if len(caption_ids_neg) > self.max_caption_len:
                caption_ids_neg = caption_ids_neg[:(self.max_caption_len - 1)] + [caption_ids_neg[-1]]

            # set caption to max_caption_len
            caption_ids_gt = np.array(caption_ids_gt, dtype=np.int64)
            caption_ids_neg = np.array(caption_ids_neg, dtype=np.int64)

            new_cap_ids_gt = np.zeros(self.max_caption_len, dtype=np.int64)
            new_cap_ids_gt[:min(self.max_caption_len, caption_ids_gt.shape[0])] = caption_ids_gt[:min(self.max_caption_len,
                                                                                             caption_ids_gt.shape[0])]
            new_cap_ids_neg = np.zeros(self.max_caption_len, dtype=np.int64)
            new_cap_ids_neg[:min(self.max_caption_len, caption_ids_neg.shape[0])] = caption_ids_neg[:min(self.max_caption_len,
                                                                                       caption_ids_neg.shape[0])]
            ITM_label_1 = torch.tensor(1, dtype=torch.int64)
            ITM_label_0 = torch.tensor(0, dtype=torch.int64)
            # if ITM_label == 1:
            #     ITM_label_binary = np.array([0,1], dtype=np.float32)
            # else:
            #     ITM_label_binary = np.array([1,0], dtype=np.float32)
            return (im_np_gt, new_cap_ids_gt, ITM_label_1), (im_np_neg, new_cap_ids_neg, ITM_label_0)

        elif self.split == 'test':
            # img_idx = index // (self.num_captions_per_img * len(self.img_keys))  # 要获取的图片
            # cap_idx = index % (self.num_captions_per_img * len(self.img_keys))    #  要获取的cap的行号，与上面图片不一定是match的
            # img_idx1 = cap_idx // self.num_captions_per_img # 获取cap_idx行号所对应的图片的idx，用img_idx1和img_idx是否相等，来获取label是1还是0
            # cap_idx1 = cap_idx % self.num_captions_per_img # 要输入的cap
            # return img_idx, [self.img_keys[img_idx1], cap_idx1]

            img_idx = index // self.img_num
            cap_idx = index % self.img_num

            img_np1, caption1, cap_tokens1, img_id1, cap_id1 = self.get_data_by_idx(img_idx)
            img_np2, caption2, cap_tokens2, img_id2, cap_id2 = self.get_data_by_idx(cap_idx)
            # different images may have same caption, so cap_id is used to judge whether image and caption with different idx are matched
            label = 1 if img_idx == cap_idx or cap_id1 == cap_id2 else 0
            # we use img_np1 and cap_tokens2 as input
            cap_ids = self.tokenizer.convert_tokens_to_ids(cap_tokens2)
            cap_ids = np.array(cap_ids, dtype=np.int64)
            new_cap_ids = np.zeros(self.max_caption_len, dtype=np.int64)
            new_cap_ids[:min(cap_ids.shape[0], self.max_caption_len)] = cap_ids[:min(cap_ids.shape[0], self.max_caption_len)]
            return img_np1, new_cap_ids, label


def trainRetrieval(model, epochs, train_data_loader, save_path='./checkpoints/retrieval.model', save_freq=50):
    # optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
    #                               lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001, amsgrad=False)
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #                               lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-7)

    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()
        t = time.time()
        n = len(train_data_loader)
        for i_, (gt_sample, neg_sample) in enumerate(train_data_loader):
            image = torch.cat([gt_sample[0], neg_sample[0]], dim=0)
            caption = torch.cat([gt_sample[1], neg_sample[1]], dim=0)
            label = torch.cat([gt_sample[2], neg_sample[2]], dim=0)

            image = image.cuda()
            caption = caption.cuda()
            label = label.cuda()
            logits = model(image, caption, image_text_label=label)
            if len(label.shape) == 2:
                # binary cross_entropy, label will be [batch, prob_of_each_type]
                loss = F.binary_cross_entropy_with_logits(logits, label).mean()
            elif len(label.shape) == 1:
                # cross_entropy, label will be [batch,]
                loss = F.cross_entropy(logits, label)
            else:
                assert False, 'wrong label dimentsion!'

            total_loss += loss.item() * image.shape[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch+1) % save_freq == 0:
            _save_path = save_path + '-' + str(epoch)
            torch.save(model, _save_path)
            logger.info(f'save to {_save_path}')
        logger.info("epoch: {}, using time: {},loss: {}".format(epoch, time.time() - t, total_loss / n) )
    return


def testRetrieval(model, test_data_loader, output_file='./results/retrieval/test_result.pt'):
    model.eval()
    results = {}
    labels = {}
    logger.info("start testing...")

    with torch.no_grad():
        for img, caption, label in test_data_loader:
            img = img.cuda()
            caption = caption.cuda()
            prob = model(img, caption)

            result = prob[:, 1]
            result = [_.to(torch.device("cpu")) for _ in result]

            for res, lab in zip(result, label):
                results[len(results)] = res.item()
                labels[len(labels)] = lab.item()
            # results.update({idx.item(): res.item() for idx, res in zip(indexs, result)})
            # for r in result:
            #     results[len(results)] = r.item()
        torch.save([results, labels], output_file)
    logger.info("Prediction results and labels saved to {}".format(output_file))

    model.train(True)
    return


def compute_ranks(dataset, results):
    # results: [pred_result, gt_labels]
    labels = np.array([results[1][i] for i in range(len(dataset))])

    similarities = np.array([results[0][i] for i in range(len(dataset))])
    # num_captions_per_img = len(dataset.img_keys) * dataset.num_captions_per_img
    num_captions_per_img = dataset.img_num
    labels = np.reshape(labels, [-1, num_captions_per_img]) # (img_num, img_num)
    similarities = np.reshape(similarities, [-1, num_captions_per_img])  # (img_num, img_num)
    i2t_ranks, t2i_ranks = [], []
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1] # max prob
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        i2t_ranks.append(rank)

    labels = np.swapaxes(labels, 0, 1)
    similarities = np.swapaxes(similarities, 0, 1)
    for lab, sim in zip(labels, similarities):
        inds = np.argsort(sim)[::-1]
        rank = num_captions_per_img
        for r, ind in enumerate(inds):
            if lab[ind] == 1:
                rank = r
                break
        t2i_ranks.append(rank)
    return i2t_ranks, t2i_ranks




# def compute_ranks(dataset, results):
#     labels = np.array([dataset.get_label(i) for i in range(len(dataset))])
#     similarities = np.array([results[i] for i in range(len(dataset))])
#     num_captions_per_img = len(dataset.img_keys) # * dataset.num_captions_per_img
#     labels = np.reshape(labels, [-1, num_captions_per_img])
#     similarities = np.reshape(similarities, [-1, num_captions_per_img])
#     i2t_ranks, t2i_ranks = [], []
#     for lab, sim in zip(labels, similarities):
#         inds = np.argsort(sim)[::-1]
#         rank = num_captions_per_img
#         for r, ind in enumerate(inds):
#             if lab[ind] == 1:
#                 rank = r
#                 break
#         i2t_ranks.append(rank)
#     if not dataset.has_caption_indexs:
#         labels = np.swapaxes(labels, 0, 1)
#         similarities = np.swapaxes(similarities, 0, 1)
#         for lab, sim in zip(labels, similarities):
#             inds = np.argsort(sim)[::-1]
#             rank = num_captions_per_img
#             for r, ind in enumerate(inds):
#                 if lab[ind] == 1:
#                     rank = r
#                     break
#             t2i_ranks.append(rank)
#     return i2t_ranks, t2i_ranks


def evaluate(eval_dataset, test_results):
    i2t_ranks, t2i_ranks = compute_ranks(eval_dataset, test_results)
    rank = [1, 5, 10]
    i2t_accs = [sum([_ < r for _ in i2t_ranks]) / len(i2t_ranks) for r in rank]
    logger.info("I2T Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                i2t_accs[0], i2t_accs[1], i2t_accs[2]))
    eval_result = {"i2t_retrieval": {"R@1": i2t_accs[0], "R@5": i2t_accs[1], "R@10": i2t_accs[2]}}
    if t2i_ranks:
        t2i_accs = [sum([_ < r for _ in t2i_ranks]) / len(t2i_ranks) for r in rank]
        logger.info("T2I Retrieval: {:.4f} @ R1, {:.4f} @ R5, {:.4f} @ R10".format(
                    t2i_accs[0], t2i_accs[1], t2i_accs[2]))
        eval_result["t2i_retrieval"] = {"R@1": t2i_accs[0], "R@5": t2i_accs[1], "R@10": t2i_accs[2]}
    return eval_result



def RetrievalTask(args):
    config = MVLBertRetrieval.from_pretrained('bert-base-uncased')
    config.conv = args.conv
    config.lr = args.lr
    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
    config.update_special_tokens(tokenizer)
    utils.print_obj(config)
    print(args)
    epochs = args.epochs

    batch_size = args.batch

    model_name = args.pretrained_path
    if args.pretrained:
        model = MVLBertForRetrieval.from_pretrained(model_name)
    else:
        model = MVLBertForRetrieval(config)

    # load the checkpoint if specified
    if args.ckpt_path is not None:
        model = torch.load(args.ckpt_path)
    model.cuda()

    train_dataset = RetrievalPretrainDataset(config, 'train')
    test_dataset = RetrievalPretrainDataset(config, 'test')
    logger.info("total number of train samples: {}, test samples: {}".format(len(train_dataset), len(test_dataset)))
    time_tmp = time.asctime(time.localtime(time.time())).replace(':', '-')
    if args.do_train:
        # torch.cuda.set_device(args.device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        if args.use_cache:
            train_dataset.save_data_in_cache()

        logger.info("start training...")
        model.train()
        trainRetrieval(model, epochs=epochs, train_data_loader=train_dataloader,
                       save_path=f'./checkpoints/rgc_retrieval_model_{args.conv}'+ (f'_{os.path.split(args.pretrained_path)[-1]}_' if args.pretrained else '') + time_tmp, save_freq=args.save_freq)


    output_file = f'./results/retrieval/RGC_{args.conv}_test_result_{time_tmp}.pt' if args.output_file is None else args.output_file
    if args.do_test:
        logger.info("start testing")
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        if args.use_cache:
            test_dataset.save_data_in_cache()
        testRetrieval(model, test_dataloader, output_file=output_file)


    if args.do_rank:
        logger.info("Start ranking the result.")
        results, labels = torch.load(output_file)
        eval_result = evaluate(test_dataset, [results, labels])
        result_file = os.path.splitext(output_file)[0] + '_test.json'
        with open(result_file, 'w') as f:
            json.dump(eval_result, f)
        logger.info("Evaluation results saved to {}.".format(result_file))
    return


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-6, type=float)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--use_cache', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_path', default='./checkpoints/swin-bert-base', type=str)
    parser.add_argument('--batch', help='batch', default=32, type=int)
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear', 'resnet50', 'swintransformer', 'vit'], type=str, default='resnet101')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_rank', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='path to recover ckpt')
    parser.add_argument('--save_freq', default=20, type=int)
    parser.add_argument('--output_file', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    global logger
    # if not os.path.exists(args.output_dir):
    #     os.mkdir(args.output_dir)
    logger = setup_logger("rgc_retrieval", 'log', 0, 'rgc-retrieval.txt')

    RetrievalTask(args)