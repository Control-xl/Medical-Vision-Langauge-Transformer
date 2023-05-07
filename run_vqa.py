import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import time
from modules.config import MVLBertConfigforVQA
from modules.model import MVLBertForVQA
import argparse
import utils
import _pickle as cPickle
from transformers import BertTokenizer
import os
import json
from torch.optim.lr_scheduler import MultiStepLR

# os.environ["TOKENIZERS_PARALLELISM"] = "frue"
class MedVQADataset(Dataset):
    def __init__(self, dataset, config, split):
        super().__init__()
        self.dataset = dataset
        self.config = config
        assert dataset in ['SLAKE', 'VQA-RAD']
        self.root = './dataset/' + dataset
        # self.anno_file = 'question_' + split + '.json'
        self.image_data_path = os.path.join('./dataset/', dataset, dataset +'_image_data.pkl')
        self.text_file =  os.path.join('./dataset/', dataset, dataset + '_text_data.pkl' )
        print("read image data from", self.image_data_path)
        self.img_id2idx, self.idx2img_id, self.img_list_in_np = cPickle.load(open(self.image_data_path, 'rb'))
        self.entries, self.ans2label, self.label2ans = cPickle.load(open(self.text_file, 'rb'))

        self.entries = self.entries[split]
        self.max_len = 23
        if dataset == 'VQA-RAD':
            self.max_len = 30

        # self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


    def __getitem__(self, index):
        img_id = self.entries[index]['img_id']
        img_idx = self.img_id2idx[img_id]
        v = self.img_list_in_np[img_idx]
        q = self.entries[index]['q_ids']
        _a = self.entries[index]['label']
        # for unanswerable questions, set to ignore_index in nn.CELoss
        if _a == None:
            a = -100
        else:
            a = _a
        question_type = self.entries[index]['answer_type']
        return v, q, a, question_type, img_id, self.entries[index]['question']

    def __len__(self):
        return len(self.entries)

    def tokenize(self, tokenizer):
        max_q_len = 0
        assert tokenizer.eos_token == '[END]', 'tokenizer.eos_token must be [END]!'
        for entry in self.entries:
            question = entry['question'] + ' [END]'
            q_tokens = tokenizer.tokenize(question)
            q_ids = tokenizer.convert_tokens_to_ids(q_tokens)
            entry['q_ids'] = q_ids
            if len(q_ids) > max_q_len:
                max_q_len = len(q_ids)
        print("max question length in dataset:", max_q_len, "setting:", self.max_len)
        for entry in self.entries:
            q_ids = entry['q_ids']
            q_ids = np.array(q_ids, dtype=np.int64)
            q_new = np.zeros(self.max_len, dtype=np.int64)
            q_new[:min(q_ids.shape[0], self.max_len)] = q_ids[:min(q_ids.shape[0], self.max_len)]
            entry['q_ids'] = q_new




def trainVQA(model, train_data_loader, valid_data_loader, epochs, save_path='./checkpoints/a.model', test_data_loader=None):
    # optim = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.05, momentum=0.9, weight_decay=0.0005) #0.01->0.05
    # bert-large: lr=1e-5, base: lr=4*1e-5(remain test)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=model.config.lr, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0001)
    optim.zero_grad()
    best_acc = -1.0
    loss_F = nn.CrossEntropyLoss()
    scheduler = None

    for epoch in range(epochs):
        if test_data_loader is not None: #  and epoch % 2 == 0
            acc, open_acc, close_acc, predict_list = testVQA(model, test_data_loader)
            print("\n#############################")
            print("test acc:%0.3f, open acc:%0.3f, close acc:%0.3f"%(acc, open_acc, close_acc))
            print("###############################\n")
        total_loss = 0
        t = time.time()
        n = len(train_data_loader)
        for i_, (v, q, a, _, _, _) in enumerate(train_data_loader):
            # if type(v) == list:
            #     for i in range(len(v)):
            #         v[i] = v[i].cuda()
            # else:
            v = v.cuda()
            q = q.cuda()
            a = a.cuda()
            prob, logits = model(v, q, a)
            loss = loss_F(logits, a)
            loss.backward()
            optim.step()
            total_loss += loss.item() * q.size(0)
            optim.zero_grad()
        if valid_data_loader is not None:
            acc, open_acc, close_acc, _ = testVQA(model, valid_data_loader)
            if acc >= best_acc:
                best_acc = acc
                torch.save(model, save_path)
            print("valid acc:%0.3f, open acc:%0.3f, close acc:%0.3f" % (acc, open_acc, close_acc))
        print("epoch:", epoch, "using time:", time.time()-t, "loss:", total_loss/n)
        if scheduler is not None:
            scheduler.step()




def create_predict_result(dataset, img_id, question, label, predict, correct, answer_type):

    entry = {
        'question': question,
        'label': dataset.label2ans[label],
        'predict': dataset.label2ans[predict],
        'correct': correct,
        'img_id': img_id,
        'answer_type': answer_type
    }
    return entry



def testVQA(model, data_loader, output_res=False):
    model.eval()
    total = 0
    correct = 0
    open_cor = 0
    open_tot = 0
    close_cor = 0
    close_tot = 0
    predict_list = []

    with torch.no_grad():
        for i_, (v, q, a, answer_type, img_id, question_text) in enumerate(data_loader):
            v = v.cuda()
            q = q.cuda()
            a = a.numpy()
            res, logits,  = model(v, q, a)
            total += res.shape[0]
            res = torch.argmax(res, dim=1).cpu().numpy()
            q = q.cpu().numpy()
            # print(a.shape)
            # print(res.shape)
            predict_res = (res == a)

            for i in range(len(predict_res)):
                # OPEN-ENDED acc and CLOSED-ENDED acc
                if answer_type[i] == 'OPEN':
                    open_tot += 1
                    if predict_res[i]:
                        open_cor += 1
                else:
                    close_tot += 1
                    if predict_res[i]:
                        close_cor += 1
                # print(i, q[i].shape)
                if output_res:
                    # print(data_loader.dataset.idx2img_id)
                    entry = create_predict_result(data_loader.dataset, img_id[i] if isinstance(img_id[i], str) else img_id[i].item(),
                         question_text[i], a[i].item(),res[i].item(), predict_res[i].item(), answer_type[i])
                    predict_list.append(entry)
                    # print(entry)
                    # print(question_text[i])
                    # predict_list['q'].append(q[i])
                    #
                    # predict_list['a'].append(a[i])
                    # predict_list['p'].append(res[i])
                    # predict_list['t'].append(answer_type[i])
                    # if type(img_id[i]) == str:
                    #     predict_list['i'].append(img_id[i])
                    # else:
                    #     predict_list['i'].append(img_id[i].cpu().numpy())
            correct += np.sum(predict_res)
    model.train(True)

    return correct/total, open_cor/open_tot, close_cor/close_tot, predict_list


def MedVQATask(args):
    assert args.dataset in ['VQA-RAD', 'SLAKE'], 'task should be either VQA-RAD or SLAKE'
    torch.cuda.set_device(args.device)
    model_name = args.pretrained_path

    if args.pretrained:
        config = MVLBertConfigforVQA.from_pretrained(model_name)
    else:
        config = MVLBertConfigforVQA.from_pretrained('bert-base-uncased')
        config.conv = args.conv


    tokenizer = BertTokenizer.from_pretrained('./dataset/bert-base-uncased')
    tokenizer.add_special_tokens({'eos_token': '[END]'})
    config.update_special_tokens(tokenizer)
    print("loading dataset")

    train_dataset = MedVQADataset(args.dataset, config, 'train')
    valid_dataset = MedVQADataset(args.dataset, config, 'valid') if args.dataset == 'SLAKE' else None
    test_dataset = MedVQADataset(args.dataset, config, 'test')

    config.result_num = len(train_dataset.ans2label)
    config.lr = args.lr
    utils.print_obj(config)

    # train_dataset.tokenizer = tokenizer
    train_dataset.tokenize(tokenizer)
    if valid_dataset is not None:
        # valid_dataset.tokenizer = tokenizer
        valid_dataset.tokenize(tokenizer)
    # test_dataset.tokenizer = tokenizer
    test_dataset.tokenize(tokenizer)

    if valid_dataset is None:
        valid_dataset = test_dataset

    batch_size = args.batch
    batch_size_small = min(16, args.batch)
    print("loading Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size_small, shuffle=False, num_workers=8) if valid_dataset is not None else None
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size_small, shuffle=False, num_workers=1)
    #
    print("training samples in trainset:", len(train_dataset))
    print("training samples in trainset:", len(valid_dataset))
    print("training samples in testset:", len(test_dataset))
    epoch = args.epochs
    total_model_num = args.total_round
    # result of model which achieve the best performance in validation set
    vqa_res = np.zeros(total_model_num)
    vqa_open = np.zeros(total_model_num)
    vqa_close = np.zeros(total_model_num)
    # result of model of the last epoch
    vqa_final_res = np.zeros(total_model_num)
    vqa_final_open = np.zeros(total_model_num)
    vqa_final_close = np.zeros(total_model_num)

    for i in range(total_model_num):
        print(i, ": ")
        torch.manual_seed(i)
        if args.pretrained:
            if model_name.startswith('resnet101'):
                config.vocab_size = 30523
            model = MVLBertForVQA.from_pretrained(model_name, config=config)
        else:
            model = MVLBertForVQA(config)

        params = sum(p.numel() for p in model.parameters())
        print(f'parameters:{params / 1000000.}')

        if args.recover_path is not None:
            model = torch.load(args.recover_path)
            model.cuda()
            acc, open_acc, close_acc, predict_list = testVQA(model, test_data_loader, output_res=True)
            print("vqa results:", acc)
            print("vqa open results:", open_acc)
            print("vqa close results:", close_acc)
            results_path = './results/Med-VQA'
            if not os.path.exists(results_path):
                os.mkdir(results_path)
            # results_file_path = os.path.join(results_path, args.dataset + '-' + args.conv + '-' + str(i) + '.json')
            results_file_path = os.path.join(results_path, os.path.split(args.recover_path)[-1] + '.json')
            json.dump(predict_list, open(results_file_path, 'w', encoding='utf-8'))
            return

        model.train()
        model.cuda()

        vqa_save_path = f"./checkpoints/vqa_model{'_pretrained_' + args.pretrained_path.split('/')[-1] if args.pretrained else ''}_{config.conv}_{i}_{str(args.dataset)}.pkl"
            # "./checkpoints/vqa_model" + ('_pretrained_' if args.pretrained else '') + '_' + args.pretrained_path.split('/')[-1] + '_' + config.conv + '_' + str(i) + '_' + str(args.dataset) + ".pkl"
        final_model_path = "./checkpoints/vqa_model_final" + ('_pretrained_' if args.pretrained else '') + str(i) + '_' + str(args.dataset) + ".pkl"

        _test_data_loader = test_data_loader
        # if args.dataset == 'VQA-RAD':
        #     _test_data_loader = None
        if not args.not_train:
            print("train for vqa:")
            trainVQA(model, train_data_loader, valid_data_loader, epoch, vqa_save_path, _test_data_loader)

        # torch.save(model, final_model_path)
        # model = torch.load(final_model_path)
        acc, open_acc, close_acc, predict_list = testVQA(model, test_data_loader, output_res=True)
        vqa_final_res[i] = acc
        vqa_final_open[i] = open_acc
        vqa_final_close[i] = close_acc

        if valid_dataset is not None:
            model = torch.load(vqa_save_path)
            acc, open_acc, close_acc, predict_list = testVQA(model, test_data_loader, output_res=True)

        # san_predict = trans_idx2word(san_predict, KVQA_train_dataset)
        # save_result(san_predict, 'san' + str(i))
            print("vqa_save_path:", vqa_save_path)
            print("pick the best in valid set: test acc:%0.3f, open acc:%0.3f, close acc:%0.3f" % (acc, open_acc, close_acc))
            vqa_res[i] = acc
            vqa_open[i] = open_acc
            vqa_close[i] = close_acc
            results_path = './results/Med-VQA'
            if not os.path.exists(results_path):
                os.mkdir(results_path)
            results_file_path = os.path.join(results_path, args.dataset + '-' + args.conv  + '-' + str(i) + '.json')
            # json.dump(predict_list, open(results_file_path, 'w', encoding='utf-8'))

        # print("vqa results:", vqa_res, "var:", np.var(vqa_res), "std:", np.std(vqa_res), "avg:",np.average(vqa_res))
        print("vqa results:", vqa_res, "avg:", np.average(vqa_res))
        print("vqa open results:", vqa_open, 'avg:', np.average(vqa_open))
        print("vqa close results:", vqa_close, 'avg:', np.average(vqa_close))
        print("vqa final results:", vqa_final_res, "avg:", np.average(vqa_final_res))
        print("vqa final open results:", vqa_final_open, 'avg:', np.average(vqa_final_open))
        print("vqa final close results:", vqa_final_close, 'avg:', np.average(vqa_final_close))

    print(args.dataset, ' Done!')


def parse_args():
    parser = argparse.ArgumentParser('Medical VLBert')
    parser.add_argument('--device', help='gpu number to train the model', default=0, type=int)
    parser.add_argument('--lr', help='learning rate', default=2e-5, type=float)
    parser.add_argument('--batch', help='batch', default=64, type=int)
    parser.add_argument('--dataset', help='VQA-RAD or SLAKE', choices=['VQA-RAD', 'SLAKE'], type=str)
    parser.add_argument('--conv', help='convolutional layer for images',
                        choices=['resnet101', 'linear', 'resnet50', 'swintransformer', 'vit','visiontransformer' ], type=str, required=True)
    parser.add_argument('--pretrained', help='whether to use pretrained model', action='store_true')
    parser.add_argument('--pretrained_path', default='./checkpoints/resnet50-bert-base')
    parser.add_argument('--recover_path', default=None)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--total_round', default=10, type=int)
    parser.add_argument('--not_train', action='store_true')
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    print(args)
    MedVQATask(args)