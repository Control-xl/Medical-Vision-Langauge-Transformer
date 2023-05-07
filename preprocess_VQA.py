import _pickle as cPickle
import os
from PIL import Image
import numpy as np
import json
import re
import argparse
def create_entry(question, label, img_id, answer_type):
    entry = {
        'question': question,
        'label': label,
        'img_id': img_id,
        'answer_type': answer_type
    }
    return entry

dataset_root = './dataset'
image_size = (224, 224)

def preprocessImage(dataset_name):
    data_path = os.path.join(dataset_root, dataset_name)
    img_id2idx = {}
    idx2img_id = []
    img_list_in_np = []
    # image_size = (640, 480)
    print("image_size:", image_size)
    if dataset_name == 'SLAKE':
        img_path = os.path.join(data_path, 'imgs')
        for dir in os.listdir(img_path):
            # 'xmlab' + id
            _, end = re.search('xmlab', dir).span()
            img_id = int(dir[end:])
            tmp_path = os.path.join(img_path, dir)
            for file_name in os.listdir(tmp_path):
                if file_name.endswith(".jpg"):
                    file_path = os.path.join(tmp_path, file_name)
                    im = Image.open(file_path, 'r') #1
                    im = im.resize(image_size) # w, h
                    im_np = np.array(im, dtype=np.float32)
                    im_np = np.transpose(im_np, (2, 0, 1))  # channel, h, w
                    for c in range(im_np.shape[0]):
                        im_np[c] = (im_np[c] - np.mean(im_np[c])) / np.var(im_np[c])

                    if img_id in img_id2idx.keys():
                        print("ALready preprocessed image:", img_id)
                        print(file_path)
                        continue
                    # print(output.shape)
                    img_id2idx[img_id] = len(img_id2idx)
                    idx2img_id.append(img_id)
                    img_list_in_np.append(im_np)
        print("total imgs:", len(img_list_in_np), "with each:", img_list_in_np[0].shape)
        # image_data_path = './data/SLAKE_image_data.pkl'
        # cPickle.dump([img_id2idx, idx2img_id, img_list_in_np], open(image_data_path, 'wb'))
    elif dataset_name == 'VQA-RAD':
        img_path = os.path.join(data_path, 'VQA_RAD Image Folder')
        for img_name in os.listdir(img_path):
            file_path = os.path.join(img_path, img_name)
            im = Image.open(file_path, 'r')  # 1
            im = im.resize(image_size)  # w, h
            im_np = np.array(im, dtype=np.float32)
            im_np = np.transpose(im_np, (2, 0, 1))  # channel, h, w
            # print(np.max(im_np), np.min(im_np), np.mean(im_np), np.var(im_np))
            for c in range(im_np.shape[0]):
                im_np[c] = (im_np[c] - np.mean(im_np[c])) / np.var(im_np[c])

            if img_name in img_id2idx.keys():
                print("ALready preprocessed image:", img_name)
                print(file_path)
                continue
            img_id2idx[img_name] = len(img_id2idx)
            idx2img_id.append(img_name)
            img_list_in_np.append(im_np)
        print("total imgs:", len(img_list_in_np), "with each:", img_list_in_np[0].shape)
    image_data_path = os.path.join('./dataset/', dataset_name, dataset_name +'_image_data.pkl')
    cPickle.dump([img_id2idx, idx2img_id, img_list_in_np], open(image_data_path, 'wb'))
    # # patches
    # image_patch_data_path = './data/' + dataset_name + '_image_patch_data.pkl'
    # image_patch_data = []
    # # the patches is shape of (patch_num, channel, patch_h, patch_w)
    # # in order to feed them into resnet, we should flatten all the patches into shape of (channel, patch_h, patch_w)
    # for img in img_list_in_np:
    #     patches_to_add = []
    #     img_patches_all = split_into_patches(img) # list of patches
    #     for patches in img_patches_all:
    #         patch_num = patches.shape[0]
    #         for i in range(patch_num):
    #             patches_to_add.append(patches[i])
    #
    #     image_patch_data.append(patches_to_add)
    #
    # cPickle.dump(image_patch_data, open(image_patch_data_path, 'wb'))



def preprocessText(dataset_name):
    # preprocess question answer
    # use idx to represent answer
    data_path = os.path.join(dataset_root, dataset_name)
    if dataset_name == 'SLAKE':

        answer2label_path = './dataset/SLAKE/combine/en_ans2label.pkl'
        answer2label = cPickle.load(open(answer2label_path, 'rb'))
        label2answer_path = './dataset/SLAKE/combine/en_label2ans.pkl'
        label2answer = cPickle.load(open(label2answer_path, 'rb'))
        preprocess_entries_path = ['./dataset/SLAKE/combine/en_train_target.pkl', './dataset/SLAKE/combine/en_validate_target.pkl',
                                   './dataset/SLAKE/combine/en_test_target.pkl']
        json_file_names = ['question_train.json', 'question_validate.json', 'question_test.json']
        split = ['train', 'valid', 'test']
        entries = {'train': [],
                   'valid': [],
                   'test': []}
        test_open_ended_cnt = 0
        test_close_ended_cnt = 0
        for _i, file_name in enumerate(json_file_names):
            qa_path = os.path.join(data_path, file_name)
            print("preprocessing file:", qa_path)
            qa_file = json.load(open(qa_path, encoding='utf-8'))
            preprocess_entries = cPickle.load(open(preprocess_entries_path[_i], 'rb'))
            for idx, qa_pair in enumerate(qa_file):
                if qa_pair["q_lang"] == 'zh':
                    continue
                qid = qa_pair['qid']
                question = qa_pair['question']
                img_id = qa_pair['img_id']
                answer_type = qa_pair['answer_type']
                answer = qa_pair['answer']
                if split[_i] == 'test':
                    if answer_type =='OPEN':
                        test_open_ended_cnt += 1
                    else:
                        test_close_ended_cnt += 1
                assert qid == preprocess_entries[idx]['qid'], 'not aligned!'
                # print(preprocess_entries[idx])
                # # preprocess answer, make sure 'a,b,c' and 'b,a,c' are the same type
                # # answer_list = list(set(answer.replace(', ', ',').split(',')))
                # answer_list = sorted(answer.replace(', ', ',').split(','))
                # answer_tmp = ''
                # for i in range(len(answer_list)):
                #     answer_tmp += answer_list[i]
                #     if i < len(answer_list) - 1:
                #         answer_tmp += ','
                # answer = answer_tmp
                # if answer not in answer2label.keys():
                #     answer2label[answer] = len(answer2label)
                #     label2answer.append(answer)
                # entries[split].append(create_entry(question, answer2label[answer], img_id, answer_type))
                # In SLAKE, a few of answers to question in test samples don't appear in training set, so these samples are not used for test.
                if len(preprocess_entries[idx]['labels']) > 0:
                    entries[split[_i]].append(create_entry(question, preprocess_entries[idx]['labels'][0], img_id, answer_type))
                else:
                    entries[split[_i]].append(create_entry(question, None, img_id, answer_type))
                    print("Unanswerable qa_pair in", split[_i], "split:", qa_pair)

        # print("close-ended questions in test set:", test_close_ended_cnt)
        # print("open-ended questions in test set:", test_open_ended_cnt)
        # print(label2answer)
        print("training samples:", len(entries['train']))
        print("validation samples:", len(entries['valid']))
        print("testing samples:", len(entries['test']))

    else:
        # answer2label = {}
        # label2answer = []
        # for answers of VQA-RAD, we follow 'Overcoming Data Limitation in Medical Visual Question Answering' (MICCAI2019)
        # from https://github.com/aioz-ai/MICCAI19-MedVQA
        ans2label_path = './dataset/VQA-RAD/cache/trainval_ans2label.pkl'
        answer2label = cPickle.load(open(ans2label_path, 'rb'))
        label2ans_path = './dataset/VQA-RAD/cache/trainval_label2ans.pkl'
        label2answer = cPickle.load(open(label2ans_path, 'rb'))
        print(answer2label)
        preprocess_entries_path = ['./dataset/VQA-RAD/cache/train_target.pkl', './dataset/VQA-RAD/cache/test_target.pkl']

        entries = {'train': [],
                   'test': []}

        json_file_names = ['trainset.json', 'testset.json']
        split = ['train', 'test']
        k = len(json_file_names)
        freefrom_cnt = 0
        freefrom_cnt_real = 0
        freefrom_cnt_false = 0
        freefrom_open = 0
        freefrom_close = 0
        para_cnt = 0
        para_cnt_real = 0
        para_cnt_false = 0
        para_open = 0
        para_close = 0
        open_real = 0
        close_real = 0
        open_all = 0
        close_all = 0
        for _i in range(k):
            file_name = json_file_names[_i]
            qa_path = os.path.join(data_path, file_name)
            print("preprocessing file:", qa_path)
            qa_file = json.load(open(qa_path, encoding='utf-8'))
            preprocess_entries = cPickle.load(open(preprocess_entries_path[_i], 'rb'))
            print('data number in', preprocess_entries_path[_i],len(preprocess_entries))
            for idx, qa_pair in enumerate(qa_file):
                qid = qa_pair['qid']
                question = qa_pair['question']
                img_id = qa_pair['image_name']
                answer_type = qa_pair['answer_type']

                # The order of samples in the preprocessed files and in the raw files is actually same, just in case.
                assert qid == preprocess_entries[idx]['qid'] and img_id == preprocess_entries[idx]['image_name'], 'not aligned!'
                # answer = str(qa_pair['answer']).lower()

                # if answer not in answer2label.keys():
                #     answer2label[answer] = len(answer2label)
                #     label2answer.append(answer)
                # entries[split[_i]].append(create_entry(question, answer2label[answer], img_id, answer_type))
                # print("preprocess_entries[idx]['labels']:", preprocess_entries[idx]['labels'])
                # if _i > 0:
                #     if(qa_pair['phrase_type'] == 'para'):
                #         print(qa_pair)
                #         print(preprocess_entries[idx])
                sentence = question.lower()

                if "? -yes/no" in sentence:
                    sentence = sentence.replace("? -yes/no", "")
                if "? -open" in sentence:
                    sentence = sentence.replace("? -open", "")
                if "? - open" in sentence:
                    sentence = sentence.replace("? - open", "")
                sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace(
                    'x ray',
                    'x-ray').replace(
                    '.', '')

                question = sentence

                if _i > 0:
                    if answer_type == 'CLOSED':
                        close_all +=1
                    else:
                        open_all += 1

                if len(preprocess_entries[idx]['labels']) > 0:
                    # In VQA-RAD, some answers for test samples don't appear in training set,
                    entries[split[_i]].append(create_entry(question, preprocess_entries[idx]['labels'][0], img_id, answer_type))
                    if _i > 0:
                        if answer_type == 'CLOSED':
                            close_real += 1
                        else:
                            open_real +=1
                else:
                    # TODO: whether they should participate in the test?
                    entries[split[_i]].append(create_entry(question, None, img_id, answer_type))
                    print("Unanswerable qa_pair in", split[_i], "split:", qa_pair)
                # else:
                # if _i >= 1:
                #     if qa_pair['phrase_type'] == 'freeform':
                #         freefrom_cnt += 1
                #         if answer_type == 'CLOSED':
                #             freefrom_close +=1
                #         else:
                #             freefrom_open += 1
                #
                #         if len(preprocess_entries[idx]['labels']) > 0:
                #             freefrom_cnt_real += 1
                #         else:
                #             freefrom_cnt_false += 1
                #     else:
                #         para_cnt+=1
                #         if answer_type == 'CLOSED':
                #             para_close +=1
                #         else:
                #             para_open += 1
                #
                #         if len(preprocess_entries[idx]['labels']) > 0:
                #             para_cnt_real += 1
                #         else:
                #             para_cnt_false += 1


        print("freefrom_cnt", freefrom_cnt)
        print("freefrom_cnt_real", freefrom_cnt_real)
        print("freefrom_cnt_false", freefrom_cnt_false)
        print("freefrom_close", freefrom_close)
        print("freefrom_open", freefrom_open)

        print("para_cnt", para_cnt)
        print("para_cnt_real", para_cnt_real)
        print("para_cnt_false", para_cnt_false)

        print("para_close", para_close)
        print("para_open", para_open)

        print(sorted(label2answer))
        print("open_real:", open_real)
        print("close_real:", close_real)
        print("open_all:", open_all)
        print("close_all:", close_all)
        # print("answer number:", len(label2answer))
        # text_file = './data/VQA-RAD_text_data.pkl'
        # cPickle.dump([entries, answer2label, label2answer], open(text_file, 'wb'))
    for key, value in entries.items():
        print(key, len(value))
    print("answer number:", len(label2answer))
    text_file = os.path.join('./dataset/', dataset_name, dataset_name + '_text_data.pkl' )
    cPickle.dump([entries, answer2label, label2answer], open(text_file, 'wb'))
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset to preprocess', type=str, choices=['SLAKE', 'VQA-RAD'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    assert args.dataset == 'SLAKE' or args.dataset == 'VQA-RAD'
    preprocessImage(args.dataset)
    preprocessText(args.dataset)

    # img_id2idx, idx2img_id, img_list_in_np = cPickle.load(open(image_data_path, 'rb'))
    # print(img_id2idx)
    # print(img_list_in_np)
    # print(len(img_list_in_np))
    #
    # 408 / 451 = 0.9047
    # Open
    # 142 / 179 = 0.7933
    # Closed
    # 266 / 272 = 0.9779