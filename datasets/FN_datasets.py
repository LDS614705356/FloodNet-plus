import json
import os

import torch
from PIL import Image
import numpy as np
import six

from torch.utils import data
from torch.utils.data import DataLoader

from data_config import DataConfig
from datasets.data_utils import FNDataAugmentation
from datasets.field import TextField

import spacy

spacy_en = spacy.load('en_core_web_sm')

# IMG_FOLDER_NAME = "A"
# IMG_POST_FOLDER_NAME = 'B'
# LIST_FOLDER_NAME = 'train'
# ANNOT_FOLDER_NAME = "label"

IGNORE = 255

label_suffix = '.png'  # jpg for gan dataset, others : png


def load_img_name_list(dataset_path):
    img_name_list = os.listdir(dataset_path)

    return img_name_list


def get_img_path(root_dir, img_name):
    # img_name=os.listdir(root_dir)
    return os.path.join(root_dir, img_name)


class ImageDataset(data.Dataset):
    """VOCdataloder"""

    def __init__(self, root_dir, LIST_FOLDER_NAME, img_size=None, is_train=True, to_tensor=True):
        super(ImageDataset, self).__init__()
        # if img_size is None:
        #     img_size = [3000, 4000]
        self.root_dir = root_dir
        self.img_size = img_size
        self.LIST_FOLDER_NAME = LIST_FOLDER_NAME
        self.drop_dir = root_dir + "/drop.txt"
        self.img_name_list = []
        with open(self.drop_dir, "r") as f:
            self.drop = f.read().splitlines()
        f.close()
        # print(len(self.drop))
        # self.list_path = self.root_dir + '/' + LIST_FOLDER_NAME + '/' + self.list + '.txt'
        self.LIST_FOLDER_ORG_NAME = LIST_FOLDER_NAME + "-org-img"
        self.LIST_FOLDER_LAB_NAME = LIST_FOLDER_NAME + "-label-img"
        self.list_path = os.path.join(self.root_dir, self.LIST_FOLDER_NAME, self.LIST_FOLDER_ORG_NAME)
        self.label_path = os.path.join(self.root_dir, self.LIST_FOLDER_NAME, self.LIST_FOLDER_LAB_NAME)
        with open(os.path.join(self.root_dir, self.LIST_FOLDER_NAME, "vqadata.json"), "r") as f:
            self.vqadata = json.load(f)
        f.close()
        i = 0
        self.index_num = []
        for v in self.vqadata.values():
            if v["Image_ID"].replace(".JPG", ".jpg") not in self.drop:
                if v["Image_ID"].replace(".JPG", ".jpg") in self.img_name_list:
                    self.img_name_list.append(v["Image_ID"].replace(".JPG", ".jpg"))
                    self.index_num.append(i)
                    i = i + 1
                else:
                    i = 0
                    self.img_name_list.append(v["Image_ID"].replace(".JPG", ".jpg"))
                    self.index_num.append(i)
                    i = i + 1
        print(len(self.index_num))
        # self.img_name_list = load_img_name_list(self.list_path)
        # print(len(self.img_name_list))
        # self.img_name_list = list(set(self.img_name_list).difference(set(self.drop)))
        # self.label_name_list = load_img_name_list(self.label_path)
        self.A_size = len(self.img_name_list)  # get the size of dataset A
        # print(self.A_size)
        self.to_tensor = to_tensor
        if is_train:
            self.augm = FNDataAugmentation(
                img_size=self.img_size,
                with_random_hflip=True,
                with_random_vflip=True,
                with_scale_random_crop=True,
                with_random_blur=False,
                random_color_tf=False
            )
        else:
            self.augm = FNDataAugmentation(
                img_size=self.img_size
            )

    def __getitem__(self, index):
        name = self.img_name_list[index]
        A_path = get_img_path(self.list_path, self.img_name_list[index % self.A_size])
        # B_path = get_img_post_path(self.root_dir, self.img_name_list[index % self.A_size])

        img = np.asarray(Image.open(A_path).convert('RGB'))
        # img_B = np.asarray(Image.open(B_path).convert('RGB'))

        [img], _ = self.augm.transform([img], [], to_tensor=self.to_tensor)

        return {'A': img, 'name': name}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.A_size


class FNDataset(ImageDataset):

    def __init__(self, root_dir, LIST_FOLDER_NAME, img_size, vocab_path, text_path, is_train=True, label_transform=None,
                 to_tensor=True):
        super(FNDataset, self).__init__(root_dir, LIST_FOLDER_NAME, img_size=img_size, is_train=is_train,
                                        to_tensor=to_tensor)
        self.label_transform = label_transform
        self.sentence_list = {}
        self.question_list = {}
        self.textField = TextField(vocab_path=vocab_path)
        with open(os.path.join(root_dir, LIST_FOLDER_NAME, text_path), "r") as f:
            file = json.load(f)
            self.ann = file
        f.close()

        index = 0

        for it in self.ann.values():
            if it["Image_ID"].replace(".JPG", ".jpg") in self.drop:
                continue
            tmp_list = []
            temp_list = []
            sentence = str(it["Ground_Truth"])
            question = it["Question"]
            sentence = [tok.text for tok in spacy_en.tokenizer(sentence)]
            question = [tok.text for tok in spacy_en.tokenizer(question)]
            sentence[0] = six.text_type.lower(sentence[0])
            question[0] = six.text_type.lower(question[0])
            question = question[:-1]
            sentence_token = [self.textField.vocab.stoi[w] for w in sentence]
            question_token = [self.textField.vocab.stoi[w] for w in question]

            if len(sentence_token) >= 23:
                print("Warning!!! Invalid token num ", len(sentence_token), it["Image_ID"])
            tokens = [self.textField.vocab.stoi["<bos>"]]
            for c in sentence_token:
                tokens.append(c)
            tokens.append(self.textField.vocab.stoi["<eos>"])
            for i in range(23 - len(sentence_token)):
                tokens.append(self.textField.vocab.stoi["<pad>"])

            tmp_list.append(tokens)

            if len(question_token) >= 18:
                print("Warning!!! Invalid token num ", len(question_token))
            tokens = [self.textField.vocab.stoi["<bos>"]]
            for c in question_token:
                tokens.append(c)
            tokens.append(self.textField.vocab.stoi["<eos>"])
            for i in range(18 - len(question_token)):
                tokens.append(self.textField.vocab.stoi["<pad>"])

            temp_list.append(tokens)
            tmp_list = torch.tensor(tmp_list).long()
            self.sentence_list[it["Image_ID"].replace(".JPG", ".jpg")+str(self.index_num[index])] = tmp_list
            temp_list = torch.tensor(temp_list).long()
            self.question_list[it["Image_ID"].replace(".JPG", ".jpg")+str(self.index_num[index])] = temp_list
            index = index + 1
            # for sentence in it["sentence"]:
            #     token_text = sentence["tokens"]
            #     token_text[0] = six.text_type.lower(token_text[0])
            #     token_num = [self.textField.vocab.stoi[w] for w in token_text]
            #
            #     if(len(token_num)>=2):
            #         print("Warning!!! Invalid token num ", len(token_num))
            #     tokens = []
            #     tokens.append(self.textField.vocab.stoi["<bos>"])
            #     for c in token_num:
            #         tokens.append(c)
            #     tokens.append(self.textField.vocab.stoi["<eos>"])
            #     for i in range(2-len(token_num)):
            #         tokens.append(self.textField.vocab.stoi["<pad>"])
            #     # tokens = [torch.tensor(tokens).long()]
            #     # print(tokens)
            #     tmp_list.append(sentence["raw"])
            #     temp_list.append(tokens)
            # self.raw_list[it["filename"]] = tmp_list
            # temp_list = torch.tensor(temp_list).long()
            # self.token_list[it["filename"]] = temp_list
        print(index)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        name_index = self.index_num[index]
        A_path = get_img_path(self.list_path, self.img_name_list[index])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        # remainder = index
        L_path = get_img_path(self.label_path, self.img_name_list[index].replace(".jpg", "_lab.png"))
        # print(img)
        label = np.array(Image.open(L_path), dtype=np.uint8)
        # print("label",label)

        # if you are getting error because of dim mismatch ad [:,:,0] at the end
        # print("classes", label_classes)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        img, label = self.augm.transform([img], [label], to_tensor=self.to_tensor)
        question = self.question_list[name+str(name_index)][0]
        sentence = self.sentence_list[name+str(name_index)][0]
        # img = np.array(img)
        # label = np.array(label)
        # img = torch.from_numpy(img)
        # label = np.array(label)
        # print("label", label)
        # label classes used in pspnet
        # label_classes = [np.sum(label == i) for i in range(10)]
        # label_classes = np.array(label_classes)
        # label_classes[label_classes > 0] = 1
        # label = [TF.to_tensor(img) for img in imgs]
        # label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
        return {'name': name, 'A': img, 'L': label, 'Q': question, 'S': sentence}


class FNDataset_shadow(ImageDataset):

    def __init__(self, root_dir, LIST_FOLDER_NAME, img_size, vocab_path, text_path, is_train=True, label_transform=None,
                 to_tensor=True):
        super(FNDataset_shadow, self).__init__(root_dir, LIST_FOLDER_NAME, img_size=img_size, is_train=is_train,
                                               to_tensor=to_tensor)
        self.label_transform = label_transform
        self.sentence_list = {}
        self.question_list = {}
        self.textField = TextField(vocab_path=vocab_path)
        with open(os.path.join(root_dir, LIST_FOLDER_NAME, text_path), "r") as f:
            file = json.load(f)
            self.ann = file
        f.close()

        index = 0
        for it in self.ann.values():
            if it["Image_ID"].replace(".JPG", ".jpg") in self.drop:
                continue
            tmp_list = []
            temp_list = []
            sentence = str(it["Ground_Truth"])
            question = it["Question"]
            sentence = [tok.text for tok in spacy_en.tokenizer(sentence)]
            question = [tok.text for tok in spacy_en.tokenizer(question)]
            sentence[0] = six.text_type.lower(sentence[0])
            question[0] = six.text_type.lower(question[0])
            question = question[:-1]
            sentence_token = [self.textField.vocab.stoi[w] for w in sentence]
            question_token = [self.textField.vocab.stoi[w] for w in question]

            if len(sentence_token) >= 23:
                print("Warning!!! Invalid token num ", len(sentence_token), it["Image_ID"])
            tokens = [self.textField.vocab.stoi["<bos>"]]
            for c in sentence_token:
                tokens.append(c)
            tokens.append(self.textField.vocab.stoi["<eos>"])
            for i in range(23 - len(sentence_token)):
                tokens.append(self.textField.vocab.stoi["<pad>"])

            tmp_list.append(tokens)

            if len(question_token) >= 18:
                print("Warning!!! Invalid token num ", len(question_token))
            tokens = [self.textField.vocab.stoi["<bos>"]]
            for c in question_token:
                tokens.append(c)
            tokens.append(self.textField.vocab.stoi["<eos>"])
            for i in range(18 - len(question_token)):
                tokens.append(self.textField.vocab.stoi["<pad>"])

            temp_list.append(tokens)
            tmp_list = torch.tensor(tmp_list).long()
            self.sentence_list[it["Image_ID"].replace(".JPG", ".jpg")+str(self.index_num[index])] = tmp_list
            temp_list = torch.tensor(temp_list).long()
            self.question_list[it["Image_ID"].replace(".JPG", ".jpg")+str(self.index_num[index])] = temp_list

            index = index + 1
            # for sentence in it["sentence"]:
            #     token_text = sentence["tokens"]
            #     token_text[0] = six.text_type.lower(token_text[0])
            #     token_num = [self.textField.vocab.stoi[w] for w in token_text]
            #
            #     if(len(token_num)>=2):
            #         print("Warning!!! Invalid token num ", len(token_num))
            #     tokens = []
            #     tokens.append(self.textField.vocab.stoi["<bos>"])
            #     for c in token_num:
            #         tokens.append(c)
            #     tokens.append(self.textField.vocab.stoi["<eos>"])
            #     for i in range(2-len(token_num)):
            #         tokens.append(self.textField.vocab.stoi["<pad>"])
            #     # tokens = [torch.tensor(tokens).long()]
            #     # print(tokens)
            #     tmp_list.append(sentence["raw"])
            #     temp_list.append(tokens)
            # self.raw_list[it["filename"]] = tmp_list
            # temp_list = torch.tensor(temp_list).long()
            # self.token_list[it["filename"]] = temp_list

        print(index)

    def __getitem__(self, index):
        name = self.img_name_list[index]
        name_index = self.index_num[index]
        A_path = get_img_path(self.list_path, self.img_name_list[index])
        img = np.asarray(Image.open(A_path).convert('RGB'))
        # remainder = index
        L_path = get_img_path(self.label_path, self.img_name_list[index].replace(".jpg", "_lab.png"))
        # print(img)
        label = np.array(Image.open(L_path), dtype=np.uint8)
        # print("label",label)

        # if you are getting error because of dim mismatch ad [:,:,0] at the end
        # print("classes", label_classes)
        #  二分类中，前景标注为255
        if self.label_transform == 'norm':
            label = label // 255

        img, label = self.augm.transform([img], [label], to_tensor=self.to_tensor)
        question = self.question_list[name+str(name_index)][0]
        sentence = self.sentence_list[name+str(name_index)][0]
        # img = np.array(img)
        # label = np.array(label)
        # img = torch.from_numpy(img)
        # label = np.array(label)
        # print("label", label)
        # label classes used in pspnet
        # label_classes = [np.sum(label == i) for i in range(10)]
        # label_classes = np.array(label_classes)
        # label_classes[label_classes > 0] = 1
        # label = [TF.to_tensor(img) for img in imgs]
        # label = torch.from_numpy(np.array(label, np.uint8)).unsqueeze(dim=0)
        return {'name': name, 'A': img, 'L': label, 'Q': question, 'S': sentence}


if __name__ == '__main__':
    data_name = 'FloodNet'
    dataConfig = DataConfig().get_data_config(data_name)
    root_dir = dataConfig.root_dir

    split = "train"
    split_val = 'val'

    training_set = FNDataset(root_dir=root_dir, LIST_FOLDER_NAME=split,
                                    img_size=[4000, 3000], vocab_path="vocab_vqa.json", text_path="vqadata.json",
                                    is_train=True)
    # print(training_set[30])
    # print(len(training_set))
    val_set = FNDataset_shadow(root_dir=root_dir, LIST_FOLDER_NAME=split_val,
                               img_size=[4000, 3000], vocab_path="vocab_vqa.json", text_path="vqadata.json", is_train=False)

    datasets = {'train': training_set, 'val': val_set}
    dataloaders = {x: DataLoader(datasets[x], batch_size=4,
                                 shuffle=True, num_workers=4,
                                 drop_last=True)
                   for x in ['train', 'val']}

    total = len(dataloaders['train'])
    print(total)

    for Each in dataloaders['train']:
        print(Each['Q'])
    # for Each in dataloaders['train']:
    #     # for each in Each['R']:
    #     print(Each['R'])
    # for batch_id, batch in enumerate(dataloaders['train'], 0):
    #     print(batch["name"])
    #     A = batch["A"]
    #     L = batch["L"]
    #     T = batch["T"]
    #     R = batch["R"]
    #     R_list = [r for it in R for r in it]
    #     print(R_list)
    #     R = []
    #     for i in range(4):
    #         R.append([])
    #     for i in range(20):
    #         R[i % 4].append(R_list[i])
    #     print(R)
    #     # for k in range(4):
    #     #     tmp_list = []
    #     #     for i in range(len(R_list)):
    #     #         tmp_list.append(R_list[i] if i%4 == k)
    #     # R = [R_list[i] for k in range(4) for i%4 == k]
    #     # for it in R:
    #
    #     # print(T)
    #     # print(R)
    #     # print(A.shape)
    #
    #     for i in range(4):
    #         mini_A = torch.split(A, 1, dim=0)
    #         mini_L = torch.split(L, 1, dim=0)
    #         for K in range(len(mini_L)):
    #             A_batch = torch.split(mini_A[K], 16, dim=1)
    #             L_batch = torch.split(mini_L[K], 16, dim=1)
    #             for i in A_batch:
    #                 print(i.shape)
                # mini_A = A[i:i+1, mini_batch:min(mini_batch+16, 100)].squeeze(dim=0)
                # mini_L = batch["L"][i:i+1, mini_batch:min(mini_batch+16, 100)]
                # print(mini_A.shape)

