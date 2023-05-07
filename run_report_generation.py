import argparse
import os
import _pickle as cPickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from modules.config import MVLBertConfigForImageCaption
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertTokenizer
from modules.model import MVLBertForImageCaption
from pycocoevalcap.eval import MVLBertEvalCap
import utils
import random
import json


DATASET_NAME = 'RGC'
SEED = 0

class ImageCaptionDataset(Dataset):
    # given a caption, find the corresponding image
    def __init__(self, config, split='train'):
        super().__init__()
        self.config = config
        self.split = split

        self.data_root = os.path.join('./dataset/RGC/', split)
        assert split in ['train', 'test'], 'Dataset has to be train set or test set'

        self.data_path = os.path.join(self.data_root, split + '_img_idx2path' + '.pkl')
        print("load data index information from:", self.data_path)

        # self.img_idx2path, self.img_idx2original_img_path = cPickle.load(open(self.data_path, 'rb'))
        self.img_idx2path = cPickle.load(open(self.data_path, 'rb'))
        self.img_num = len(self.img_idx2path)
        self.tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')

        self.max_length = self.config.max_length # max token length for caption after tokenization
        self.mask_word = True if self.split == 'train' else False

        self.use_cache = False # read the images and captions into memory if use_cache is True



    def save_data_in_cache(self):
        assert self.tokenizer.eos_token is not None, 'please add the eos token for the tokenizer.'
        # self.cached_images = []
        self.cached_images = np.zeros((self.img_num, 3, 224, 224), dtype=np.float32)
        self.cached_caps_tokens = []
        self.cached_caps = []
        print("saving images in memory for", self.split, "set")
        for idx in range(self.img_num):
            data_path = self.img_idx2path[idx]
            im_np, caption, _, _ = cPickle.load(open(data_path, 'rb'))
            # self.cached_images.append(im_np)
            self.cached_images[idx, :, :, :] = im_np
            self.cached_caps.append(caption)

            caption_with_end = caption + ' ' + self.tokenizer.eos_token
            caption_tokens = self.tokenizer.tokenize(caption_with_end)
            self.cached_caps_tokens.append(caption_tokens)
        self.use_cache = True
        return


    def __len__(self):
        return self.img_num


    def __getitem__(self, index):

        img_path = self.img_idx2original_img_path[index]
        data_path = self.img_idx2path[index]
        if self.use_cache:
            im_np = self.cached_images[index]
            caption = self.cached_caps[index]
            caption_tokens = self.cached_caps_tokens[index]
        else:
            im_np, caption, _, _ = cPickle.load(open(data_path, 'rb'))
            caption_with_end = caption + ' ' + self.tokenizer.eos_token
            caption_tokens = self.tokenizer.tokenize(caption_with_end)


        if self.mask_word:
            # caption_tokens, mlm_labels = self._random_mask_whole_word(caption_tokens)
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
        return torch.tensor(im_np), torch.tensor(new_cap_ids).long(), caption.lower(), new_mlm_labels, img_path

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


    def _random_mask_whole_word(self, tokens):
        output_tokens = []
        output_labels = []
        for token in tokens:
            sub_tokens = self.tokenizer.tokenize(token)
            _p = random.random()

            # mask token with 15% prob
            if _p < 0.15:
                _p /= 0.15
                # 80% randomly change token to [MASK]
                if _p < 0.8:
                    for sub_token in sub_tokens:
                        output_tokens.append("[MASK]")
                # 10% randomly change token to random token
                elif _p < 0.9:
                    for sub_token in sub_tokens:
                        output_tokens.append(random.choice(list(self.tokenizer.vocab.keys())))
                # -> rest 10% randomly keep current token
                else:
                    # append current token to output (we will predict these later)
                    for sub_token in sub_tokens:
                        output_tokens.append(sub_token)

                for sub_token in sub_tokens:
                    try:
                        output_labels.append(self.tokenizer.vocab[sub_token])
                    except KeyError:
                        # For unknown words (should not occur with BPE vocab)
                        output_labels.append(self.tokenizer.vocab["[UNK]"])
                        # logging.warning("Cannot find sub_token '{}' in vocab. Using [UNK] insetad".format(sub_token))
            else:
                for sub_token in sub_tokens:
                    # no masking token (will be ignored by loss function later)
                    output_tokens.append(sub_token)
                    output_labels.append(-100)
        return output_tokens, output_labels

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list



def test(model, test_data_loader, args, mode='beam'):
    print("start testing...")
    gt = {}
    res = {}
    cnt = 0
    model.cuda()
    model.eval()
    print("start preprocessing generated results.")
    print("search mode is", mode)
    save_res = []

    for index, (image, caption_token_ids, caption_gt, _, datapath) in enumerate(test_data_loader):
        image = image.cuda()
        caption_token_ids = caption_token_ids.cuda()
        if mode == 'beam':
            output_tokens_ids = model(image, caption_token_ids, learning_strategy='unilm', num_beams=5)
        else:
            output_tokens_ids, _ = model(image, caption_token_ids, learning_strategy='unilm', num_beams=1)
        # if (index +1) % 30 ==0:
        #     print('output_tokens_ids.shape', output_tokens_ids.shape)
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
            save_res.append({'image': datapath[batch_idx], 'pred': output_sequence, 'gt': caption_gt[batch_idx]})

    output_file = './results/'+ DATASET_NAME + '_' + args.conv+ '_' + str(SEED) + '.json'
    print("results saved into", output_file)
    json.dump(save_res, open(output_file, 'w'))
    for i in range(len(gt)):
        if i % 500 == 0:
            print("gt:", gt[i])
            print("res:", res[i])

    MVlBertEval = MVLBertEvalCap(gt, res)
    MVlBertEval.evaluate()
    # create output dictionary
    out = {}
    for metric, score in MVlBertEval.eval.items():
        out[metric] = score

    # imgToEval = MVlBertEval.imgToEval
    # for p in preds_filt:
    #     image_id, caption = p['image_id'], p['caption']
    #     imgToEval[image_id]['caption'] = caption
    #
    # with open(cache_path, 'w') as outfile:
    #     json.dump({'overall': out, 'imgToEval': imgToEval}, outfile)
    return out

class CrossEntropyForImageCaption(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.LogSoftmax(dim=-1)

    def forward(self, logits, tgt):
        props = self.m(logits)
        mask = (tgt > 0).float()
        print(tgt)
        tgt_props = props.gather(1, tgt.unsqueeze(1)).squeeze()
        return -(tgt_props * mask).sum() / mask.sum()

        # tgt_props = props.gather(2, tgt.unsqueeze(2)).squeeze()
        # mask = (tgt > 0).float()
        # return -(tgt_props * mask).sum() / mask.sum()


# def imageCaptionTask(args):
#
#     torch.cuda.set_device(args.device)
#     config = MVLBertConfigForImageCaption.from_pretrained('bert-base-uncased')
#     config.seq2seq_mask = False
#     # config.max_length = 5
#     tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
#     tokenizer.add_special_tokens({'eos_token': '[END]'})
#     config.update_special_tokens(tokenizer)
#     config.seq2seq_mask = False
#     utils.print_obj(config)
#
#     print("args:", args)
#     train_dataset = ImageCaptionDataset(config, 'train', sub_data=args.dataset)
#     test_dataset = ImageCaptionDataset(config, 'test', sub_data=args.dataset)
#     # update tokenizer
#     train_dataset.tokenizer = tokenizer
#     test_dataset.tokenizer = tokenizer
#
#     train_data_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=8, shuffle=True)
#     test_data_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=8, shuffle=False)
#     # test_data_loader =  DataLoader(train_dataset, batch_size=args.batch, num_workers=8, shuffle=False)
#
#     epochs = args.epochs
#     eos_token_id = config.eos_token_id
#     pad_token_id = config.pad_token_id
#
#     model_path = './checkpoints/imagecaption.model0-240'
#     if args.pretrained:
#         print("load pretrained model from", model_path)
#         model = torch.load(model_path)
#
#         print("testing caption model...")
#         with torch.no_grad():
#             # eval_results = test(model, test_data_loader)
#             eval_results = test(model, test_data_loader, 'greedy')
#         # for k, v in eval_results.items():
#         #     print(k, v)
#     else:
#         model = MVLBertForImageCaption(config)
#
#     model.tokenizer = tokenizer
#     model = model.cuda()
#
#
#     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
#                                   lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
#     print("totol training samples:", len(train_dataset), "total testing samples:",len(test_dataset))
#
#     if args.scst:
#         rl_crit = RewardCriterion()
#         print("using scst")
#     else:
#         print("using sum of log probability")
#         CE = CrossEntropyForImageCaption()
#
#     for epoch in range(epochs):
#         t = time.time()
#         n = len(train_data_loader)
#         total_loss = 0
#         for _step, (image, caption_ids, _, _) in enumerate(train_data_loader):
#             # print(_step)
#             model.eval()
#             image = image.cuda()
#             caption_ids = caption_ids.cuda()
#             if args.scst:
#                 greedy_res = caption_ids.new(caption_ids.size(0), caption_ids.size(1)).fill_(0)
#                 gen_result = caption_ids.new(caption_ids.size(0), caption_ids.size(1)).fill_(0)
#
#                 with torch.no_grad():
#                     greedy_res_raw, _, _ = model(image, caption_ids, num_beams=1, sample_mode='greedy')
#                     for b in range(greedy_res_raw.size(0)):
#                         for idx in range(greedy_res_raw.size(1)):
#                             if greedy_res_raw[b][idx] not in [eos_token_id, pad_token_id]:
#                                 greedy_res[b][idx] = greedy_res_raw[b][idx]
#                             else:
#                                 if greedy_res_raw[b][idx] == eos_token_id:
#                                     greedy_res[b][idx] = eos_token_id
#                                 break
#                 model.train()
#
#                 gen_result_raw, sample_logprobs = model(image, caption_ids, num_beams=1, sample_mode='sample')
#                 for b in range(gen_result_raw.size(0)):
#                     for idx in range(gen_result_raw.size(1)):
#                         if gen_result_raw[b][idx] not in [eos_token_id, pad_token_id]:
#                             gen_result[b][idx] = gen_result_raw[b][idx]
#                         else:
#                             if gen_result_raw[b][idx] == eos_token_id:
#                                 gen_result[b][idx] = eos_token_id
#                             break
#
#                 gt_ids = caption_ids
#                 reward = get_self_critical_reward(greedy_res, gt_ids, gen_result, gt_ids.size(0))
#                 reward = torch.from_numpy(reward).float().to(gen_result.device)
#                 mean_reward = reward.mean()
#                 loss = rl_crit(sample_logprobs, gen_result.data, reward)
#             else:
#                 # prob_all_tokens (batch, vocab_size, generated_seq_len)
#                 _, _, logits_all_tokens = model(image, caption_ids, num_beams=1, sample_mode='greedy')
#                 print("caption_ids.shape:", caption_ids.shape)
#                 print("logits_all_tokens.shape:", logits_all_tokens.shape)
#                 # convert to (batch, seq_len, vocab_size) so that it can be used in
#                 # logits_all_tokens = torch.transpose(logits_all_tokens, 1, 2)
#                 loss = CE(logits_all_tokens, caption_ids)
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#
#             total_loss += loss.item() * image.shape[0]
#         print("epoch:", epoch, "using time:", time.time()-t, "loss:", total_loss/n)
#
#         if (epoch + 1) % 20 == 0:
#             save_path = model_path + str(args.device) + '-' + str(epoch+1)
#             print("save to", save_path)
#             torch.save(model, save_path)
#             with torch.no_grad():
#                 eval_results = test(model, test_data_loader)
#             for k,v in eval_results.items():
#                 print(k, v)


def imageCaptionTaskUniLM(args):
    torch.cuda.set_device(args.device)
    # torch.cuda.manual_seed(args.seed)
    # global SEED
    # SEED = args.seed
    config = MVLBertConfigForImageCaption.from_pretrained('bert-base-uncased')

    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')

    config.conv = args.conv
    config.lr = args.lr
    config.update_special_tokens(tokenizer)

    if args.dropout:
        config.hidden_dropout_prob = 0.1
        config.attention_probs_dropout_prob = 0.1
    else:
        config.hidden_dropout_prob = 0.0
        config.attention_probs_dropout_prob = 0.0
    utils.print_obj(config)

    print("args:", args)
    train_dataset = ImageCaptionDataset(config, 'train')
    test_dataset = ImageCaptionDataset(config, 'test')

    train_data_loader = DataLoader(train_dataset, batch_size=args.batch, num_workers=8, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch, num_workers=8, shuffle=False)

    if args.use_cache:
        train_dataset.save_data_in_cache()
        test_dataset.save_data_in_cache()

    epochs = args.epochs

    time_tmp = time.asctime(time.localtime(time.time())).replace(':', '-')
    model_path = f"./checkpoints/RGC_report_generation-{'pretrained' if args.pretrained else ''}-{args.conv}-{DATASET_NAME}-{time_tmp}"


    if args.pretrained:
        model_path += '-pretrained-'
        ckpt_path = args.pretrained_path
        print("load pretrained model from", ckpt_path) #./checkpoints/imagecaption.model0unilm-medpix-260
        model = MVLBertForImageCaption.from_pretrained(ckpt_path, config=config, tokenizer=tokenizer)
        model.tokenizer = tokenizer
        model = model.cuda()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
        # print("testing caption model...")
        # with torch.no_grad():
        #     # eval_results = test(model, test_data_loader ,args,)
        #     eval_results = test(model, test_data_loader, args, 'greedy')
    else:
        model = MVLBertForImageCaption(config, tokenizer=tokenizer)
        model = model.cuda()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)

    print("totol training samples:", len(train_dataset), "total testing samples:", len(test_dataset))

    for epoch in range(epochs):
        # if epoch > 100 and args.pretraiend:
        #     for p in model.parameters():
        #         p.requires_grad = True
        #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
        #                                   lr=model.config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0001)
        t = time.time()
        n = len(train_data_loader)
        total_loss = 0
        for _step, (image, caption_ids, gt_caption, mlm_labels, _) in enumerate(train_data_loader):
            model.train()
            image = image.cuda()
            caption_ids = caption_ids.cuda()
            mlm_labels = mlm_labels.cuda()
            # batch, vocab_size, seq_len
            MLM_logits = model(image, caption_ids, num_beams=0, learning_strategy='unilm')

            # for debug
            # batch_predicted_tokens = torch.argmax(MLM_logits, dim=1) # batch, seq_len
            # batch_size = mlm_labels.shape[0]
            # for i in range(2):
            #     print("caption:", gt_caption[i])
            #     gt_caption_toknens = tokenizer.tokenize(gt_caption[i])
            #     print("tokenized caption:", gt_caption_toknens)
            #     print("masked gt label:")
            #     print(caption_ids[i])
            #     print("gt label:")
            #     print(tokenizer.convert_tokens_to_ids(gt_caption_toknens))
            #     label = mlm_labels[i]
            #     predicted_tokens = batch_predicted_tokens[i]
            #     print("label:")
            #     print(label)
            #     print("predict:")
            #     print(predicted_tokens)
            #     for idx in range(len(label)):
            #         if label[idx] != -100:
            #             print("at position", idx)
            #             print("gt label:", label[idx].item(), "predicted label:", predicted_tokens[idx].item())
            #             print("gt token:", tokenizer.convert_ids_to_tokens(label[idx].item()), '-',
            #                   "predicted token:", tokenizer.convert_ids_to_tokens(predicted_tokens[idx].item()))
            #     print('---------------------')

            loss = F.cross_entropy(MLM_logits, mlm_labels, ignore_index=-100)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cur_loss = loss.item()
            total_loss += cur_loss

        print("epoch:", epoch, "using time:", time.time() - t, "loss:", total_loss / n)
        test_frq = args.test_frq

        if (epoch + 1) % test_frq == 0:
            search_mode = 'beam' if args.beam_search else 'greedy'
            with torch.no_grad():
                eval_results = test(model, test_data_loader, args, search_mode)

            if (epoch + 1) % (2*test_frq) == 0:
                save_path = model_path + str(args.seed) + 'unilm-' + str(args.conv) + '-' + str(epoch + 1)
                print("save to", save_path)
                torch.save(model, save_path)
            #     print("attention_score dropout:", model.MVLBert.encoder.layer[0].attention.self.dropout.p,
            #           "output_layer dropout:", model.MVLBert.encoder.layer[0].attention.output.dropout.p)
            # # for k,v in eval_results.items():
            # #     print(k, v)

# python run_report_generation.py --conv vit --pretrained --pretrained_path ./checkpoints/vit-rgc-roco-medicat --epochs 400 --beam_search --batch 32 --test_freq 10

def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--batch', help='batch size for training and testing', default=32, type=int)
    parser.add_argument('--dataset', help='sub dataset to run', choices=['medpix', 'mimic', None], default=None, type=str)
    parser.add_argument('--epochs', help='epochs', default=400, type=int)
    parser.add_argument('--beam_search', help='whether to use beam search during inference', action='store_true')
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear',  'resnet50', 'swintransformer','vit'], type=str, default='resnet101')
    parser.add_argument('--pretrained', help='whether to use pretrained model', action='store_true')
    parser.add_argument('--pretrained_path', default='./checkpoints/resnet101-bert-base', type=str)
    parser.add_argument('--test_freq', default=5, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--use_cache', help='save all training data in the memory' ,action='store_true')
    parser.add_argument('--dropout', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    imageCaptionTaskUniLM(args)
    # if args.unilm:
    #     imageCaptionTaskUniLM(args)
    # else:
    #     imageCaptionTask(args)
    # SLAKETask(args)
    # RetrievalTask(args )