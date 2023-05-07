import torch
import time
import sys
import numpy as np
import argparse
import utils
import os
import torch.nn.functional as F
from torchvision import transforms
from transformers import BertTokenizer
import _pickle as cPickle
import random
from torch.utils.data import DataLoader, Dataset
from modules.model import MVLBertForRetrieval
from modules.config import MVLBertRetrieval
from modules.logger import setup_logger
import json, re
from PIL import Image


class CXRDatasetForRetrieval(Dataset):
    # given a caption, find the corresponding image
    def __init__(self, config, args, split='train'):
        super().__init__()
        self.config = config
        self.split = split
        self.args = args
        self.data_root = os.path.join('./dataset/iu_xray/')

        self.data_path_cleaned = os.path.join(self.data_root, 'annotation.json')

        logger.info("load data index information from: {}".format(self.data_path_cleaned))
        self.data = json.load(open(self.data_path_cleaned, 'r'))
        self.data = self.data[split]

        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
        self.max_caption_len = self.config.max_length
        self.ITM_prob = 0.5
        self.use_cache = False
        self.img_num = len(self.data)
        logger.info(f"{self.split}: {self.img_num}")

        for i in range(len(self.data)):
            caption = self.data[i]['report']
            caption = self.clean_report_iu_xray(caption)
            self.data[i]['report'] = caption
            caption += ' [END]'
            caption_tokens = self.tokenizer.tokenize(caption)
            self.data[i]['tokens'] = caption_tokens

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_data_by_idx(self, index):

        image_path = self.data[index]['image_path']
        image_1 = Image.open(os.path.join(self.data_root, 'images', image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.data_root, 'images', image_path[1])).convert('RGB')


        if self.transform is not None and not self.args.pretrained:
            image1 = self.transform(image_1)
            image2 = self.transform(image_2)
        else:
            image_size = (224, 224)
            image1 = image_1.resize(image_size)
            image1 = np.array(image1, dtype=np.float32)
            image1 = np.transpose(image1, (2, 0, 1))  # channel, h, w
            for c in range(image1.shape[0]):
                image1[c] = (image1[c] - np.mean(image1[c])) / np.var(image1[c])

            image2 = image_2.resize(image_size)
            image2 = np.array(image2, dtype=np.float32)
            image2 = np.transpose(image2, (2, 0, 1))  # channel, h, w
            for c in range(image2.shape[0]):
                image2[c] = (image2[c] - np.mean(image2[c])) / np.var(image2[c])

            image1 = torch.from_numpy(image1)
            image2 = torch.from_numpy(image2)

        image = torch.stack((image1, image2), 0)  # (2, channel, h, w)

        report = self.data[index]['report']
        # im_np, caption, img_id, cap_id = cPickle.load(open(img_path, 'rb'))
        # caption_with_end = report + ' ' + self.tokenizer.eos_token
        # caption_tokens = self.tokenizer.tokenize(caption_with_end)
        caption_tokens = self.data[index]['tokens']
        return image, report, caption_tokens, index, index


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

            # if random.random() < 0.5:
            #     # replace the image with another with different captions
            im_np_neg = rand_img_np
            caption_tokens_neg = caption_tokens_gt
            # else:
            #     # or replace the captions with negative sample.
            #     im_np_neg = im_np_gt
            #     caption_tokens_neg = rand_img_caption_tokens

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
            img_idx = index // self.img_num
            cap_idx = index % self.img_num

            img_np1, caption1, cap_tokens1, img_id1, cap_id1 = self.get_data_by_idx(img_idx)
            img_np2, caption2, cap_tokens2, img_id2, cap_id2 = self.get_data_by_idx(cap_idx)
            # different images may have same caption, so cap_id is used to judge whether image and caption with different idx are matched
            label = 1 if img_idx == cap_idx else 0
            # we use img_np1 and cap_tokens2 as input
            cap_ids = self.tokenizer.convert_tokens_to_ids(cap_tokens2)
            cap_ids = np.array(cap_ids, dtype=np.int64)
            new_cap_ids = np.zeros(self.max_caption_len, dtype=np.int64)
            new_cap_ids[:min(cap_ids.shape[0], self.max_caption_len)] = cap_ids[:min(cap_ids.shape[0], self.max_caption_len)]
            return img_np1, new_cap_ids, label


def trainRetrieval(model, epochs, train_data_loader, save_path='./checkpoints/report_retrieval.model', save_freq=30):
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

            total_loss += loss.item()
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
        logger.info(f"load models from {args.ckpt_path}")
        model = torch.load(args.ckpt_path)
    model.cuda()

    train_dataset = CXRDatasetForRetrieval(config, args, 'train')
    test_dataset = CXRDatasetForRetrieval(config, args, 'test')
    logger.info("total number of train samples: {}, test samples: {}".format(len(train_dataset), len(test_dataset)))
    time_tmp = time.asctime(time.localtime(time.time())).replace(':', '-')
    if args.do_train:
        # torch.cuda.set_device(args.device)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        logger.info("start training...")
        model.train()

        trainRetrieval(model, epochs=epochs, train_data_loader=train_dataloader,
                       save_path=f"./checkpoints/IUXray_report_retrieval_model_{args.conv}-{'pretrained' if args.pretrained else ''}-{time_tmp}",  save_freq=args.save_freq)

    output_file = f'./results/retrieval/IUXray_report_test_result_{time_tmp}.pt'
    if args.do_test:
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
        testRetrieval(model, test_dataloader, output_file=output_file)


    if args.do_rank:
        logger.info("Start ranking the result.")
        results, labels = torch.load(output_file)
        eval_result = evaluate(test_dataset, [results, labels])
        result_file = os.path.splitext(output_file)[0] + '.test.json'
        with open(result_file, 'w') as f:
            json.dump(eval_result, f)
        logger.info("Evaluation results saved to {}.".format(result_file))
    return


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--lr', help='learning rate', default=1e-6, type=float)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--pretrained_path', default='./checkpoints/swin-bert-base', type=str)
    parser.add_argument('--batch', help='batch', default=32, type=int)
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear', 'swintransformer', 'vit'], type=str, default='resnet101')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_rank', action='store_true')
    parser.add_argument('--ckpt_path', default=None, type=str, help='path to recover ckpt')
    parser.add_argument('--save_freq', default=20, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    global logger
    time_tmp = time.asctime(time.localtime(time.time())).replace(':', '-')
    logger = setup_logger("IUXray-CXR_retrieval", 'log', 0, f'{args.conv}-retrieval-iuxray-{time_tmp}.txt')

    RetrievalTask(args)