import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
import argparse
from pycocoevalcap.eval import MVLBertEvalCap
import time
from modules.model import MVLBertForImageCaption
from modules.config import MVLBertConfigForImageCaption
from transformers import BertTokenizer
import utils
import json
from PIL import Image
import numpy as np
import random
from torchvision import transforms
import re


class BaseDataset(Dataset):
    def __init__(self,split='train'):
        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
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
        return



    def _random_mask_word(self, tokens):
        token_len = len(tokens)
        output_tokens = [token for token in tokens]

        output_labels = [-100] * token_len
        # max number of masked token is 10, each token with 20% to be masked.
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
                print(token, 'not in the vocab')
                output_labels[idx] = self.tokenizer.vocab["[UNK]"]

        return output_tokens, output_labels

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class IuxrayMultiImageDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super(IuxrayMultiImageDataset, self).__init__(split=split)
        self.args = args
        self.split = split
        self.image_dir = 'dataset/iu_xray/images/'
        self.ann_path = 'dataset/iu_xray/annotation.json'
        self.max_length = 80

        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            caption = self.examples[i]['report']
            caption = self.clean_report_iu_xray(caption)
            self.examples[i]['report'] = caption.lower()
            caption += ' [END]'
            caption_tokens = self.tokenizer.tokenize(caption)
            # self.examples[i]['ids'] = tokenizer()[:self.max_seq_length]
            self.examples[i]['tokens'] = caption_tokens
            # self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

    def __len__(self):
        return len(self.examples)

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

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        if 'image' not in example.keys():
            image_path = example['image_path']
            image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
            image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
            if self.args.cache_image:
                self.examples[idx]['image'] = (image_1, image_2)
        else:
            image_1, image_2 = example['image']

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
        # print("image.shape:", image.shape)
        caption = example['report']
        caption_tokens = example['tokens']

        if self.split == 'train':
            # caption_tokens, mlm_labels = self._random_mask_whole_word(caption_tokens)
            caption_tokens, mlm_labels = self._random_mask_word(caption_tokens)
        else:
            mlm_labels = None

        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)

        if len(caption_ids) > self.max_length:
            # preserve [END]
            caption_ids = caption_ids[:(self.max_length - 1)] + [caption_ids[-1]]
            mlm_labels = mlm_labels[:(self.max_length - 1)] + [mlm_labels[-1]] if mlm_labels is not None else None

        # set caption to max_caption_len
        caption_ids = np.array(caption_ids, dtype=np.int64)
        new_cap_ids = np.zeros(self.max_length, dtype=np.int64)
        new_cap_ids[:min(self.max_length, caption_ids.shape[0])] = caption_ids[:min(self.max_length,
                                                                                    caption_ids.shape[0])]

        new_mlm_labels = np.ones(self.max_length, dtype=np.int64) * -100
        if mlm_labels is not None:
            new_mlm_labels[:min(self.max_length, caption_ids.shape[0])] = mlm_labels[:min(self.max_length,
                                                                                          caption_ids.shape[0])]

        # sample = (image_id, image, report_ids, report_masks, seq_length)
        sample = (image, torch.tensor(new_cap_ids).long(), caption, new_mlm_labels, image_id)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super(MimiccxrSingleImageDataset, self).__init__(split=split)
        self.args = args
        self.split = split
        self.image_dir = 'dataset/mimic_cxr/images/'
        self.ann_path = 'dataset/mimic_cxr/annotation.json'
        self.max_length = 150
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        print(self.split, len(self.examples))
        for i in range(len(self.examples)):
            caption = self.examples[i]['report']
            # caption = self.clean_report_iu_xray(caption)
            caption = self.clean_report_mimic_cxr(caption)
            self.examples[i]['report'] = caption
            caption += ' [END]'
            caption_tokens = self.tokenizer.tokenize(caption)
            self.examples[i]['tokens'] = caption_tokens

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']

        image_path = example['image_path']
        image = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')

        if self.transform is not None and not self.args.pretrained:
            image = self.transform(image)
        else:
            image_size = (224, 224)
            image = image.resize(image_size)
            image = np.array(image, dtype=np.float32)
            image = np.transpose(image, (2, 0, 1))  # channel, h, w
            for c in range(image.shape[0]):
                image[c] = (image[c] - np.mean(image[c])) / np.var(image[c])

        # print("image.shape:", image.shape)
        caption = example['report']
        caption_tokens = example['tokens']

        if self.split == 'train' and args.learning_strategy=='unilm':
            # caption_tokens, mlm_labels = self._random_mask_whole_word(caption_tokens)
            caption_tokens, mlm_labels = self._random_mask_word(caption_tokens)
        elif self.split == 'train':
            mlm_labels = self.tokenizer.convert_tokens_to_ids(caption_tokens)
        else:
            mlm_labels = None

        caption_ids = self.tokenizer.convert_tokens_to_ids(caption_tokens)

        if len(caption_ids) > self.max_length:
            # preserve [END]
            caption_ids = caption_ids[:(self.max_length - 1)] + [caption_ids[-1]]
            mlm_labels = mlm_labels[:(self.max_length - 1)] + [mlm_labels[-1]] if mlm_labels is not None else None

        # set caption to max_caption_len
        caption_ids = np.array(caption_ids, dtype=np.int64)
        new_cap_ids = np.zeros(self.max_length, dtype=np.int64)
        new_cap_ids[:min(self.max_length, caption_ids.shape[0])] = caption_ids[:min(self.max_length,
                                                                                    caption_ids.shape[0])]

        new_mlm_labels = np.ones(self.max_length, dtype=np.int64) * -100
        if mlm_labels is not None:
            new_mlm_labels[:min(self.max_length, caption_ids.shape[0])] = mlm_labels[:min(self.max_length,
                                                                                          caption_ids.shape[0])]

        # sample = (image_id, image, report_ids, report_masks, seq_length)
        sample = (image, torch.tensor(new_cap_ids).long(), caption, new_mlm_labels, image_id)
        return sample


    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ').replace('--', ' -- ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report



def compute_scores(gts, res):
    """
    modified from R2GEN
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)

    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider

    for key in gts.keys():
        # list
        gts[key][0] = gts[key][0].replace('.', ' .')
        res[key][0] = res[key][0].replace('.', ' .')

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def test(model, test_data_loader, learning_strategy, mode='beam', dataset='mimic', output_file_name=None):
    print("start testing...")
    gt = {}
    res = {}
    cnt = 0
    model.cuda()
    model.eval()
    print("start preprocessing generated results.")
    print("search mode is", mode)

    save_res = []
    with torch.no_grad():
        for index, (image, caption_token_ids, caption_gt, _, datapath) in enumerate(test_data_loader):
            image = image.cuda()
            caption_token_ids = caption_token_ids.cuda()
            if mode == 'beam':
                output_tokens_ids = model(image, caption_token_ids, learning_strategy=learning_strategy, num_beams=5)
            else:
                output_tokens_ids, _ = model(image, caption_token_ids, learning_strategy=learning_strategy, num_beams=1)

            for batch_idx in range(output_tokens_ids.shape[0]):
                cur_tokens_ids = output_tokens_ids[batch_idx]
                output_buf = model.tokenizer.convert_ids_to_tokens(cur_tokens_ids)

                output_tokens = []
                for token in output_buf:
                    if token in ("[SEP]", "[PAD]", "[END]"):
                        break
                    output_tokens.append(token)
                # output_sequence = ' '.join(detokenize(output_tokens))
                output_sequence = model.tokenizer.convert_tokens_to_string(output_tokens)
                output_sequence = output_sequence.replace(' - ', '-')
                gt[cnt] = caption_gt[batch_idx]
                res[cnt] = output_sequence
                cnt += 1
                save_res.append({'image':datapath[batch_idx], 'pred': output_sequence, 'gt':caption_gt[batch_idx]})
        if output_file_name is None:
            output_file = './results/' + dataset + '-' +  mode + '.json'
        else:
            output_file = './results/' + output_file_name + '-' + mode + '.json'

        print("results saved into", output_file)
        json.dump(save_res, open(output_file, 'w'))
        # for i in range(len(gt)):
        #     if i % 300 == 0:
        #         print("gt:", gt[i])
        #         print("res:", res[i])

        MVlBertEval = MVLBertEvalCap(gt, res)
        MVlBertEval.evaluate()
        # create output dictionary
        out = {}
        for metric, score in MVlBertEval.eval.items():
            out[metric] = score
        # from R2GEN
        gt_ = {i: [cap] for i, cap in gt.items()}
        res_ = {i: [cap] for i, cap in res.items()}
        log = {}
        test_met = compute_scores(gt_, res_)
        log.update(**{'test_' + k: v for k, v in test_met.items()})
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

    model.train()
    return out


def imageCaptionTaskUniLM(args):
    torch.cuda.set_device(args.device)
    config = MVLBertConfigForImageCaption.from_pretrained('bert-base-uncased')
    if args.dataset == 'mimic':
        config.max_length = 150
    else:
        config.max_length = 100
    if args.max_length is not None:
        config.max_length = args.max_length
    else:
        args.max_length = config.max_length

    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')

    config.update_special_tokens(tokenizer)
    config.conv = args.conv
    utils.print_obj(config)
    print("args:", args)
    valid_dataset = None
    # if args.dataset.lower() == 'iucxr':
    #     train_dataset = IUCXR(args, tokenizer=tokenizer, split='train')
    #     test_dataset = IUCXR(args, tokenizer=tokenizer, split='test')

    print("loading dataset and tokenizing sentences...")
    if args.dataset == 'mimic':
        train_dataset = MimiccxrSingleImageDataset(args, split='train')
        valid_dataset = MimiccxrSingleImageDataset(args, split='val')
        test_dataset = MimiccxrSingleImageDataset(args, split='test')
    else:
        train_dataset = IuxrayMultiImageDataset(args, split='train')
        valid_dataset = IuxrayMultiImageDataset(args, split='val')
        test_dataset = IuxrayMultiImageDataset(args, split='test')


    train_data_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=min(args.batch, 16), shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=min(args.batch, 16), shuffle=False)
    valid_data_loader = DataLoader(valid_dataset if valid_dataset is not None else test_dataset,
                                   batch_size=args.batch, num_workers=min(args.batch, 16), shuffle=False)

    epochs = args.epochs


    model = MVLBertForImageCaption(config, tokenizer=tokenizer)

    if args.pretrained:
        pretrained_path = args.pretrained_path # './checkpoints/image-caption.model'
        print("load pretrained model from", pretrained_path)
        # model_ckpt = torch.load(ckpt_path)
        # model.load_state_dict(model_ckpt.state_dict())
        model = MVLBertForImageCaption.from_pretrained(pretrained_path, config=config, tokenizer=tokenizer)
        # print("testing caption model...")
        # with torch.no_grad():
        #     eval_results = test(model, test_data_loader, mask_learning=args.mask_learning, mode='greedy', dataset=args.dataset)

    if args.ckpt_path is not None:
        model = torch.load(args.ckpt_path)
        print("config overload.")
        utils.print_obj(model.config)
    model.tokenizer = tokenizer
    model = model.cuda()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
    print("totol training samples:", len(train_dataset), "total testing samples:", len(test_dataset))

    if args.do_eval:
        search_mode = 'beam' if args.beam_search else 'greedy'
        eval_results = test(model, test_data_loader, learning_strategy=args.learning_strategy, mode=search_mode,
                            dataset=args.dataset)
    print("model.config.conv:", model.config.conv)

    time_tmp = time.asctime(time.localtime(time.time())).replace(':', '-')
    # model_path = './checkpoints/report_generation-' + ('pretrained' if args.pretrained else '') + args.conv + '-' + \
    #              args.dataset + '-' + args.learning_strategy
    model_path = f"./checkpoints/report_generation-{'pretrained' if args.pretrained else ''}-{args.conv}-{args.dataset}-{args.learning_strategy}-{time_tmp}"

    for epoch in range(epochs):
        t = time.time()
        n = len(train_data_loader)
        total_loss = 0
        for _step, (image, caption_ids, gt_caption, mlm_labels, _) in enumerate(train_data_loader):
            # for IU-Xray in R2Gen, image.shape will be [batch, 2, 3, 224,224]
            # for others, it will be [batch, 3, 224, 224]
            image = image.cuda()
            caption_ids = caption_ids.cuda()
            mlm_labels = mlm_labels.cuda()

            MLM_logits = model(image, caption_ids, learning_strategy=args.learning_strategy, num_beams=0)  # batch, vocab_size, seq_len

            loss= F.cross_entropy(MLM_logits, mlm_labels, ignore_index=-100)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cur_loss = loss.item() * image.shape[0]
            total_loss += cur_loss

        print("epoch:", epoch, "using time:", time.time() - t, "loss:", total_loss / n)
        # torch.save(model, './checkpoints/report_generation.model')
        test_freq = args.test_freq
        if epoch > args.epochs/2:
            test_freq = int(args.test_freq/2)

        if (epoch + 1) % test_freq == 0:
            search_mode = 'beam' if args.beam_search else 'greedy'
            # eval_results = test(model, test_data_loader, learning_strategy=args.learning_strategy, mode=search_mode, dataset=args.dataset, output_file_name=f'report_generation-{args.dataset}-{args.learning_strategy}-{time_tmp}')
            # print("###############beam search:")
            eval_results = test(model, test_data_loader, learning_strategy=args.learning_strategy, mode=search_mode, dataset=args.dataset, output_file_name=f'report_generation-{args.dataset}-{args.learning_strategy}-{time_tmp}')

        if (epoch + 1) % (test_freq*2) == 0:
            save_path = model_path + '-' + str(epoch + 1)
            print("save to", save_path)
            torch.save(model, save_path)

# python run_report_generation.py --conv swintransformer --test_frq 5 --pretrained --pretrained_path ./checkpoints/swin-roco-rgc-medicat --epochs 300

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--batch', help='batch size for training and testing', default=32, type=int)
    parser.add_argument('--epochs', help='epochs', default=200, type=int)
    parser.add_argument('--max_length', help='max token length', default=None, type=int)
    parser.add_argument('--beam_search', help='whether to use beam search during inference', action='store_true')
    parser.add_argument('--cache_image', help='whether to save images in the memory', action='store_true')
    parser.add_argument('--dataset', help='dataset to run image caption', choices=['iu_xray', 'mimic', None], type=str, default='mimic')
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear',  'resnet50', 'swintransformer'], type=str, default='resnet101')
    parser.add_argument('--test_freq', help='frequency to print test results', type=int, default=10)
    parser.add_argument('--pretrained', help='whether to use pretrained model', action='store_true')
    parser.add_argument('--pretrained_path', help='ckpt path of model to load', type=str, default='./checkpoints/image-caption.model')
    parser.add_argument('--ckpt_path', help='recover model from ckpt', type=str, default=None)
    parser.add_argument('--learning_strategy', help='train the captioning model with UniLM, '
                                                    'or just learn to generate the whole sequence with teacher force',
                        type=str, choices=['unilm', 'normal'], default='unilm')
    parser.add_argument('--do_eval', help='whether evaluate the model before training', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse_args()
    imageCaptionTaskUniLM(args)


    # test_dataloader(args)
