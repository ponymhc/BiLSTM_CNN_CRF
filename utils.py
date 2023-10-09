import re
import string
from tqdm import tqdm
import json
import pandas as pd
from collections import defaultdict
from gensim.models import KeyedVectors
import gensim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch
import torch.nn as nn
from collections import Counter
from multiprocessing import Pool
import math
import gc
import gzip
import wordninja

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    return file

def write_json(json_file, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_file, f)


class SplitJoinWords():
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.corpus = f.read()
        self.dictionary = self.build_freq_stats()
        self.segmenter = wordninja.LanguageModel('conf/freq_stats.txt.gz')

    def words(self, text): 
        return re.findall('[a-zA-Z]+', text)
    
    def build_freq_stats(self):
        freq_stats = Counter(self.words(self.corpus))
        sorted_keys = sorted(freq_stats, key=lambda k: freq_stats[k], reverse=True)
        with gzip.open('conf/freq_stats.txt.gz', 'wt', encoding='utf-8') as f:
            for word in sorted_keys:
                f.write(word+'\n')
        return set(sorted_keys)

    def split(self, word):
        return self.segmenter.split(word)

class Tokenizer():
    def __init__(self, segmenter=None):
        self.segmenter = segmenter
        self.pattern = re.compile(r'[\n\r\t]|(\d)|[\!\"\#\$\%\&\\\'\(\)\*\+\/\:\;\<\=\>\?\@\[\\\\\]\^\`\{\|\}\~\，\。\？\！\：\、\《\》\ ]|([\-\_\,\.])')

    def contains_arabic(self, text):
        pattern = r'[\u0600-\u06FF]+'  # 匹配阿拉伯文的正则表达式
        matches = re.search(pattern, text)
        if matches:
            return True
        return False
    
    def viterbi_tokenize(self, sentence):
        token_list = []
        tokens = re.split(self.pattern, sentence)
        for token in tokens:
            if token and token not in '\-\_' and not self.contains_arabic(token):
                token_list.extend(self.segmenter.split(token))
            elif token and (token in '\-\_' or self.contains_arabic(token)):
                token_list.append(token)
        return token_list
    
    def tokenize(self, sentence):
        token_list = []
        tokens = re.split(self.pattern, sentence)
        for token in tokens:
            if token:
                token_list.append(token)
        return token_list


def build_data(data_path, output_path, with_label=True):
    '''
    清洗、分词数据集
    '''
    if with_label:
        label_list = []
        text_list = []
        with open(data_path,'r',encoding='utf-8') as f:
            texts = []
            labels = []
            text_list = []
            label_list = []
            for line in tqdm(f):
                if line != '\n':
                    line_list = line.strip('\n').split('\t')
                    text_list.append(line_list[0])
                    label_list.append(line_list[1])
                else:
                    texts.append(text_list)
                    labels.append(label_list)
                    text_list = []
                    label_list = []
        with open(output_path,'w') as f:
                for text, label in tqdm(zip(texts, labels)):
                    json.dump({"text":text,"label":label},f)
                    f.write('\n')
    else:
        text_list = []
        with open(data_path,'r',encoding='utf-8') as f:
            texts = []
            text_list = []
            for line in tqdm(f):
                if line != '\n':
                    text_list.append(line.strip('\n'))
                else:
                    texts.append(text_list)
                    text_list = []
        with open(output_path,'w') as f:
                for text in tqdm(texts):
                    json.dump({"text":text},f)
                    f.write('\n')


def build_train_val_test(train_data_path, 
                         val_data_path, 
                         test_data_path, 
                         train_build_path, 
                         val_build_path, 
                         test_build_path):
    '''
    清洗训练集、验证集、测试集 
    '''
    print('清洗数据中...')
    build_data(train_data_path, train_build_path)
    build_data(val_data_path, val_build_path)
    build_data(test_data_path, test_build_path)
    
def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        file = json.load(f)
    return file

def write_json(json_file, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(json_file, f)


def build_word2id_char2id_tag2id(data_path, 
                         case=True, 
                         pretrain_vector_path=None, 
                         max_word2id_size=None, 
                         max_char2id_size=None, 
                         min_word_freq=1, 
                         min_char_freq=1):
    '''
    case: 区分大小写, 默认为不区分
    pretrain_vector: 预训练词向量的path, 以便生成word2vec中词汇的word2id
    max_word2id_size: 最大词表大小
    max_char2id_size: 最大字符表大小
    min_word_freq: 最小词频
    min_char_freq: 最小字符频
    '''
    # 生成 word2id
    word_freq = defaultdict(int)
    char_freq = defaultdict(int)
    word2id = {}
    tag2id = {}
    char2id = {}
    if not pretrain_vector_path:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                sentence = json.loads(line)
                text = sentence['text']
                tags = sentence['label']
                for word, tag in zip(text, tags):
                    if tag not in tag2id:
                        tag2id[tag] = len(tag2id)
                    if case:
                        word_freq[word] += 1
                    else:
                        word_freq[word.lower()] += 1
                    for char in word:
                        if case:
                            char_freq[char] += 1
                        else:
                            char_freq[char.lower()] += 1
        vocab_list = [(word, freq) for word, freq in word_freq.items() if freq >= min_word_freq]
        vocab_list.sort(key=lambda x: x[1], reverse=True)
        char_list = [(char, freq) for char, freq in char_freq.items() if freq >= min_char_freq]
        if max_word2id_size:
            vocab_list = vocab_list[:max_word2id_size]
        if max_char2id_size:
            char_list = char_list[:max_char2id_size]
        word2id = {'<PAD>': 0, '<UNK>': 1}
        char2id = {'<PAD>': 0, '<UNK>': 1}
        word2id.update({word_count[0]: idx+2 for idx, word_count in enumerate(vocab_list)})
        char2id.update({char_count[0]: idx+2 for idx, char_count in enumerate(char_list)})
    else:
        model = KeyedVectors.load_word2vec_format(pretrain_vector_path, binary=False)
        word2id = {'<PAD>': 0, '<UNK>': 1}
        if tuple(map(int, gensim.__version__.split('.'))) > (4,0,0):
            vocab = model.key_to_index
        else:
            vocab = model.wv.vocab
        for idx,i in enumerate(list(vocab)):
            word2id[i] = idx + 2
        with open(data_path,'r') as f:
            for line in f:
                sentence = json.loads(line)
                tags = sentence['label']
                text = text = sentence['text']
                for word, tag in zip(text, tags):
                    if tag not in tag2id:
                        tag2id[tag] = len(tag2id)
                    for char in word:
                        if case:
                            char_freq[char] += 1
                        else:
                            char_freq[char.lower()] += 1
        char_list = [(char, freq) for char, freq in char_freq.items() if freq >= min_char_freq]
        if max_char2id_size:
            char_list = char_list[:max_char2id_size]
        char2id = {'<PAD>': 0, '<UNK>': 1} 
        char2id.update({char_count[0]: idx+2 for idx, char_count in enumerate(char_list)})

    id2word = {v:k for k,v in word2id.items()}
    id2tag = {v:k for k,v in tag2id.items()}
    id2char = {v:k for k,v in char2id.items()}

    write_json(word2id, './conf/word2id.json')
    write_json(char2id, './conf/char2id.json')
    write_json(tag2id, './conf/tag2id.json')
    write_json(id2word, './conf/id2word.json')
    write_json(id2tag, './conf/id2tag.json')
    write_json(id2char, './conf/id2char.json')
    
    return word2id, char2id, tag2id

def filter_word2vec(word2id, output_path, word2vec_path):
    '''
   词向量筛选 
    '''
    print('加载词向量...')
    model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    # 创建新的词向量字典，用于存储筛选后的词汇和对应的词向量
    filtered_word2vec = {}
    # 遍历词汇表，如果词汇在预训练词向量模型中存在，则将其添加到新的词向量字典中
    print('筛选词向量...')
    if tuple(map(int, gensim.__version__.split('.'))) > (4,0,0):
        vocab = model.key_to_index
    else:
        vocab = model.wv.vocab
    for word in tqdm(vocab):
        if word in word2id:
            filtered_word2vec[word] = model[word]
    
    # 将筛选后的词向量写入到新的文本文件中
    with open(output_path, 'w', encoding='utf-8') as f:
        for word, vector in filtered_word2vec.items():
            vector_str = ' '.join(str(val) for val in vector)
            line = f"{word} {vector_str}\n"
            f.write(line)
    
    # 向量化后的维度
    with open(output_path, 'r',encoding='utf-8') as f:
        line = f.readline().split(' ')

        vector_size = len(line) - 1

    # 筛选后的词表大小
    with open(output_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    vocab_size = len(lines)

    # 将词表大小和维度数添加到文件头部
    with open(output_path, 'r+',encoding='utf-8') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(('%d %d\n' % (vocab_size, vector_size)) + content)

    print(f"Filtered word vectors saved to '{output_path}'.")


class MyDataset(Dataset):
    def __init__(self, data_path, word2id, tag2id, char2id, max_sentence_length=64, max_word_length=20, with_label=True):
        self.with_label = with_label
        self.word2id = word2id
        self.char2id = char2id
        self.texts = []
        self.data_path = data_path
        self.max_word_length = max_word_length
        if self.with_label:
            self.tag2id = tag2id
            self.labels = []
            with open(data_path,'r') as f:
                for line in f:
                    sentence = json.loads(line)
                    self.texts.append(sentence['text'])
                    self.labels.append(sentence['label'])
        else:
            with open(data_path,'r') as f:
                for line in f:
                    sentence = json.loads(line)
                    self.texts.append(sentence['text'])
        # mask
        self.mask = []
        for sentence in self.texts:
            mask = [1] * len(sentence)
            self.mask.append(mask)

        if self.with_label:
            for i in range(len(self.texts)):
                length = len(self.texts[i])
                if length < max_sentence_length:
                    pad_length = max_sentence_length - length
                    self.texts[i].extend(['<PAD>'] * pad_length)
                    self.labels[i].extend(['O'] * pad_length)
                    self.mask[i].extend([0] * pad_length)
                else:
                    self.texts[i] = self.texts[i][:max_sentence_length]
                    self.labels[i] = self.labels[i][:max_sentence_length]
                    self.mask[i] = self.mask[i][:max_sentence_length]
        else:
            for i in range(len(self.texts)):
                length = len(self.texts[i])
                if length < max_sentence_length:
                    pad_length = max_sentence_length - length
                    self.texts[i].extend(['<PAD>'] * pad_length)
                    self.mask[i].extend([0] * pad_length)
                else:
                    self.texts[i] = self.texts[i][:max_sentence_length]
                    self.mask[i] = self.mask[i][:max_sentence_length]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.with_label:
            label = self.labels[idx]
            label_to_id = []
        sentence = self.texts[idx]
        mask = self.mask[idx]
        sentence_to_id = []
        char_to_id = []
        if self.with_label:
            for word, tag, m in zip(sentence, label, mask):
                if word in self.word2id:
                    sentence_to_id.append(self.word2id[word])
                else:
                    sentence_to_id.append(self.word2id['<UNK>'])
                word_length = len(word)
                word_to_char_to_idx = []
                if len(word) <= self.max_word_length:
                    chars_list = [char for char in word] + ['<PAD>'] * (self.max_word_length - word_length)
                    for char in chars_list:
                        char_id = self.char2id.get(char, self.char2id['<UNK>'])
                        word_to_char_to_idx.append(char_id)
                else:
                    chars_list = [char for char in word[:self.max_word_length]]
                    for char in chars_list:
                        char_id = self.char2id.get(char, self.char2id['<UNK>'])
                        word_to_char_to_idx.append(char_id)
                char_to_id.append(word_to_char_to_idx)
                label_to_id.append(self.tag2id[tag])
            return torch.LongTensor(sentence_to_id), torch.LongTensor(char_to_id), torch.LongTensor(label_to_id), torch.tensor(mask).bool()
        else:
            for word,  m in zip(sentence, mask):
                if word in self.word2id:
                    sentence_to_id.append(self.word2id[word])
                else:
                    sentence_to_id.append(self.word2id['<UNK>'])
                word_length = len(word)
                word_to_char_to_idx = []
                if len(word) <= self.max_word_length:
                    chars_list = [char for char in word] + ['<PAD>'] * (self.max_word_length - word_length)
                    for char in chars_list:
                        char_id = self.char2id.get(char, self.char2id['<UNK>'])
                        word_to_char_to_idx.append(char_id)
                else:
                    chars_list = [char for char in word[:self.max_word_length]]
                    for char in chars_list:
                        char_id = self.char2id.get(char, self.char2id['<UNK>'])
                        word_to_char_to_idx.append(char_id)
                char_to_id.append(word_to_char_to_idx)
            return torch.LongTensor(sentence_to_id), torch.LongTensor(char_to_id), torch.tensor(mask).bool()



def init_network(model, method='xavier', exclude='embedding', seed=42):
    '''
    权重初始化
    method: xavier, kaiming (默认xavier)
    '''
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass

