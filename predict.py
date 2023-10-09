import json
import torch
from utils import build_data, MyDataset, DataLoader, read_json
from model.BiLSTM_CNN_CRF import BilstmCnnCrf
import os
from train_eval import test, entity_extract, cal_metrics
import argparse
from tqdm import tqdm
import string
import re
import wordninja
import numpy as np
import itertools
from sklearn import metrics


parser = argparse.ArgumentParser(description='TextCNN Predict')
parser.add_argument('--model_file', default='0', type=str, help='模型加载路径')
parser.add_argument('--pretrain_vector_file', default='0', type=str, help='词向量文件')
parser.add_argument('--predict_file', default='oot.txt', type=str, help='预测数据集')
parser.add_argument('--with_label', default=0, type=int, help='待预测数据是否有标签，1: 有，0: 没有')
parser.add_argument('--infer_batch_size', default=1024, type=int, help='推理batch_size')
args = parser.parse_args()

if __name__ == '__main__':
    # 设置线程数
    torch.set_num_threads(16)
    # 全局随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    # 读取训练中的参数
    params = read_json('./conf/params_dict.json')
    pretrain_vector_file = args.pretrain_vector_file
    if pretrain_vector_file == '0':
        pretrain_vector_file = params['pretrain_vector_file']
    pretrain_vector_path = os.path.join('conf', pretrain_vector_file)
    model_file = args.model_file
    predict_file = args.predict_file
    if args.with_label == 1:
        with_label = True
    else:
        with_label = False
    infer_batch_size = args.infer_batch_size
    if model_file == '0':
        model_save_path = params['model_saved_path']
    else:
        model_save_path = os.path.join('save_model',model_file)
    print(model_save_path)
    predict_data_path = os.path.join('./data/raw_data', predict_file)
    word2id_path = 'conf/word2id.json'
    tag2id_path = 'conf/tag2id.json'
    char2id_path = 'conf/char2id.json'
    id2word_path = 'conf/id2word.json'
    id2tag_path = 'conf/id2tag.json'

    with open(word2id_path,'r') as f:
        word2id = json.load(f)
    with open(tag2id_path,'r') as f:
        tag2id = json.load(f)
    with open(char2id_path,'r') as f:
        char2id = json.load(f)

    predict_build_path = os.path.join('./data/process_data', predict_file.split('.')[0]+'.json')

    # 数据处理
    build_data(predict_data_path, predict_build_path, with_label=with_label)
    test_dataset = MyDataset(predict_build_path, word2id, tag2id, char2id, with_label=with_label)
    test_dataloader = DataLoader(test_dataset, batch_size=infer_batch_size)

    # 设置计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BilstmCnnCrf(word2id=word2id, 
                         char2id=char2id,
                        num_classes=len(tag2id), 
                        word_embedding_dim=params['word_embedding_dim'], 
                        char_embedding_dim=params['char_embedding_dim'],
                        num_filters=params['num_filters'], 
                        filter_size=params['filter_size'], 
                        num_layers=params['num_layers'], 
                        hidden_dim=params['hidden_dim'], 
                        drop_out=params['drop_out'], 
                        embedding_pretrained=params['pretrain_vector'],
                        pretrain_vector_path=pretrain_vector_path)
    
    model.to(device)
    true_entities, predict_entities = test(model, test_dataloader, model_save_path, device, id2word_path, id2tag_path, test=True, with_label=with_label)
    
    
    predict_result_save_path = os.path.join('./data/predict_result', predict_file.split('.')[0] + '_predict_result.txt')
    with open(predict_result_save_path, 'w') as f: 
        for entity_list in tqdm(predict_entities):
            entity = ' '.join([i.split('/')[0] for i in entity_list[:-1]])
            entity_class = entity_list[0].split('/')[1][2:]
            loc_list = entity_list[-1].split('_')
            sentenct_loc = loc_list[0]
            entity_start = loc_list[1]
            entity_end = loc_list[2]
            f.write(entity_class + ": " + entity + '\t' + sentenct_loc + '\t' + entity_start + '\t' + entity_end + '\n')
            

    # # 将预测结果和待预测数据一起写入新文件，并用\t分隔它们
    # id2tag = {v:k for k,v in tag2id.items()}
    # predict_output = args.predict_file.split('.')[0] + '_predict_result.txt'
    # if with_label != 0:
    #     with open('./data/predict_output/' + predict_output, 'w', encoding='utf-8') as f:
    #         for i in range(len(predict_data_list)):
    #             new_line = predict_data_list[i] + '\t'+ label_list[i] +'\t' + id2tag[predicts[i]] + '\t' + str(probs[i]) + '\n'
    #             f.write(new_line)
    # else:
    #     with open('./data/predict_output/' + predict_output, 'w', encoding='utf-8') as f:
    #         for i in range(len(predict_data_list)):
    #             new_line = predict_data_list[i] + '\t' + id2tag[predicts[i]] + '\t' + str(probs[i]) + '\n'
    #             f.write(new_line)
