import time
from datetime import timedelta
import utils
import torch
import numpy as np
from sklearn import metrics
import json
import torch.nn.functional as F
import os
from tqdm import tqdm
import itertools

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(model, 
          model_name, 
          train_dataloader, 
          val_dataloader, 
          test_dataloader, 
          optimizer, 
          scheduler, 
          epochs, 
          early_stop, 
          tag2id_path, 
          id2word_path,
          id2tag_path,
          device):
    
    start_time = time.time()
    model.train()
    total_batch = 0  # 记录进行到多少batch
    val_best_acc = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    pre_save_path=None
    for epoch in range(epochs):
        epoch_loss = 0
        for i, (words_to_id, chars_to_id, labels, masks) in enumerate(train_dataloader):
            words_to_id = words_to_id.to(device)
            chars_to_id = chars_to_id.to(device)
            labels = labels.to(device)
            masks = masks.to(device)
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播计算损失
            loss = model.compute_loss(words_to_id, chars_to_id, labels, masks)
            # 反向传播计算梯度
            loss.backward()
            optimizer.step()
            # 统计损失
            epoch_loss += loss.item()
            if (i+1) % 10 == 0 or total_batch == 0:
                print(f'epoch: {epoch + 1}  batch: {i+1}/{len(train_dataloader)}  train_loss: {loss.item()}')
            if (total_batch+1) % 400 == 0 and total_batch != 0:
                print('Evaluating...')
                # 每多少轮输出在训练集和验证集上的效果
                true_entities, predict_entities, val_acc, val_loss, recall, f1_score, true_entities_length, \
                    predict_entities_length, acc_entities_length = evaluate(model, val_dataloader, device, id2word_path, id2tag_path)
                model.train()
                if val_acc > val_best_acc:
                    val_best_acc = val_acc
                    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                    if pre_save_path:
                        os.remove(pre_save_path)
                    model_saved_path = f"./save_model/{model_name}_{time_str}.pt"
                    torch.save(model.state_dict(), model_saved_path)
                    improve = 'Yes'
                    last_improve = total_batch
                    pre_save_path = model_saved_path
                else:
                    improve = 'No'
                time_dif = get_time_dif(start_time)
                print(f'epoch: {epoch + 1}  batch: {i+1}/{len(train_dataloader)}  val_loss: {val_loss},  val_entities_acc: {val_acc},  val_entities_recall: {recall},  val_entities_f1: {f1_score}')
                print(f'val_true_entities: {true_entities_length},  val_predict_entities: {predict_entities_length},  val_acc_entities: {acc_entities_length}')
                print(f'time: {time_dif} improve: {improve}')
            total_batch += 1
            if total_batch - last_improve > early_stop:
                # 验证集loss超过early_stop个batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
            # 更新学习率
            scheduler.step()
        if flag:
            break
    print('test...')
    test(model, test_dataloader, model_saved_path ,device, id2word_path, id2tag_path)
    return model_saved_path

def test(model, test_dataloader, save_path, device, id2word_path, id2tag_path, test=True, with_label=True):
    # test
    model.load_state_dict(torch.load(save_path, map_location=device.type))
    model.eval()
    start_time = time.time()
    if with_label:
        true_entities, predict_entities, test_acc, test_loss, recall, f1_score, true_entities_length, predict_entities_length, acc_entities_length, \
            test_report, test_confusion = evaluate(model, test_dataloader, device, id2word_path, id2tag_path, test=test, with_label=with_label)
        print(f'test_loss: {test_loss},  test_entities_acc: {test_acc},  test_entities_recall: {recall},  test_entities_f1: {f1_score}')
        print(f'test_true_entities: {true_entities_length},  test_predict_entities: {predict_entities_length},  test_acc_entities: {acc_entities_length}')
        print("Predict report:")
        print(test_report)
        print("Confusion matrix:")
        print(test_confusion)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        return true_entities, predict_entities
    else:
        true_entities, predict_entities = evaluate(model, test_dataloader, device, id2word_path ,id2tag_path, with_label=with_label)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        return true_entities, predict_entities

def evaluate(model, val_dataloader, device, id2word_path=None ,id2tag_path=None, test=False, with_label=True):
    id2tag = utils.read_json(id2tag_path)
    id2word = utils.read_json(id2word_path)
    model.eval()
    loss_total = 0
    sentences = []
    labels_all = []
    predict_all = []
    max_length = None
    if with_label:
        with torch.no_grad():
            for words_to_id, chars_to_id, labels, masks in tqdm(val_dataloader):
                if not max_length:
                    max_length = words_to_id.shape[1]
                words_to_id = words_to_id.to(device)
                chars_to_id = chars_to_id.to(device)
                labels = labels.to(device)
                masks = masks.to(device)
                loss = model.compute_loss(words_to_id, chars_to_id, labels, masks)
                out = model.decode(words_to_id, chars_to_id, masks)
                loss_total += loss.item()
                sentences.extend(words_to_id.tolist())
                labels_all.extend(labels.tolist())
                predict_all.extend([predict + [0] * (max_length - len(predict)) for predict in out])
        true_entities, predict_entities = entity_extract(sentence_list=sentences, 
                                                            predict_tag=predict_all, 
                                                            id2word=id2word, 
                                                            id2tag=id2tag, 
                                                            true_tag=labels_all,
                                                            with_label=True)
        accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length = cal_metrics(true_entities, predict_entities)
        if test:
            class_list = [''] * len(id2tag)
            for k,v in id2tag.items():
                class_list[int(k)] = v
            labels_all = list(itertools.chain.from_iterable(labels_all))
            predict_all = list(itertools.chain.from_iterable(predict_all))
            report = metrics.classification_report(labels_all, predict_all, labels=range(len(id2tag)), target_names=class_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return true_entities, predict_entities, accuracy, loss_total / len(val_dataloader), recall, f1_score, true_entities_length, predict_entities_length, acc_entities_length, report, confusion
        return true_entities, predict_entities, accuracy, loss_total / len(val_dataloader), recall, f1_score, true_entities_length, predict_entities_length, acc_entities_length
    else:
        with torch.no_grad():
            for words_to_id, chars_to_id, masks in tqdm(val_dataloader):
                words_to_id = words_to_id.to(device)
                chars_to_id = chars_to_id.to(device)
                masks = masks.to(device)
                out = model.decode(words_to_id, chars_to_id, masks)
                sentences.extend(words_to_id.tolist())
                predict_all.extend(out)
        true_entities, predict_entities = entity_extract(sentence_list=sentences, 
                                                            predict_tag=predict_all, 
                                                            id2word=id2word, 
                                                            id2tag=id2tag,
                                                            with_label=False)
        return true_entities, predict_entities
    
def entity_extract(sentence_list, predict_tag, id2word, id2tag, true_tag=[], with_label=True):
    '''
    抽取实体并标记实体位置位于句子编号和token下标
    '''
    true_entities, true_entity = [], []
    predict_entities, predict_entity = [], []
    for line_num, tags in enumerate(predict_tag):
        for char_num in range(len(tags)):

            char_text = id2word[str(sentence_list[line_num][char_num])]
            predict_tag_type = id2tag[str(predict_tag[line_num][char_num])]
            
            if with_label:
                true_tag_type = id2tag[str(true_tag[line_num][char_num])]
                if true_tag_type[0] == "S":
                    true_entity = [char_text + "/" + true_tag_type]
                    true_entity.append(str(line_num) + "_" + str(char_num) + "_" + str(char_num))
                    true_entities.append(true_entity)
                    true_entity = []
                elif true_tag_type[0] == "B":
                    true_entity = [char_text + "/" + true_tag_type]
                elif true_tag_type[0] == "I" and len(true_entity) != 0 and true_entity[-1].split("/")[1][1:] == true_tag_type[1:]:
                    true_entity.append(char_text + "/" + true_tag_type)
                elif true_tag_type[0] == 'E' and len(true_entity) != 0 and true_entity[-1].split("/")[1][1:] == true_tag_type[1:]:
                    true_entity.append(char_text + "/" + true_tag_type)
                elif true_tag_type[0] == "O" and len(true_entity) != 0 :
                    true_entity.append(str(line_num) + "_" + str(char_num - len(true_entity)) + "_" + str(char_num))
                    true_entities.append(true_entity)
                    true_entity=[]
                else:
                    true_entity=[]

            if predict_tag_type[0] == "S":
                predict_entity = [char_text + "/" + predict_tag_type]
                predict_entity.append(str(line_num) + "_" + str(char_num) + "_" + str(char_num))
                predict_entities.append(predict_entity)
                predict_entity = []
            elif predict_tag_type[0] == "B":
                predict_entity = [char_text + "/" + predict_tag_type]
            elif predict_tag_type[0] == "I" and len(predict_entity) != 0 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)
            elif predict_tag_type[0] == "E" and len(predict_entity) != 0 and predict_entity[-1].split("/")[1][1:] == predict_tag_type[1:]:
                predict_entity.append(char_text + "/" + predict_tag_type)
            elif predict_tag_type[0] == "O" and len(predict_entity) != 0:
                predict_entity.append(str(line_num) + "_" + str(char_num - len(predict_entity)) + "_" + str(char_num))
                predict_entities.append(predict_entity)
                predict_entity = []
            else:
                predict_entity = []
    return true_entities, predict_entities


def cal_metrics(true_entities, predict_entities):

    acc_entities = [entity for entity in predict_entities if entity in true_entities]

    acc_entities_length = len(acc_entities)
    predict_entities_length = len(predict_entities)
    true_entities_length = len(true_entities)

    if acc_entities_length > 0:
        accuracy = float(acc_entities_length / predict_entities_length)
        recall = float(acc_entities_length / true_entities_length)
        f1_score = 2 * accuracy * recall / (accuracy + recall)
        return accuracy, recall, f1_score, acc_entities_length, predict_entities_length, true_entities_length
    else:
        return 0, 0, 0, acc_entities_length, predict_entities_length, true_entities_length
