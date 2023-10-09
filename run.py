import torch
import numpy as np
from train_eval import train
import argparse
import os
from utils import build_word2id_char2id_tag2id, filter_word2vec, MyDataset, DataLoader, init_network, build_train_val_test, write_json
from model.BiLSTM_CNN_CRF import BilstmCnnCrf
import torch_pruning as tp

parser = argparse.ArgumentParser(description='TextCNN Classification')
parser.add_argument('--model_name', default='TextCNN', type=str, help='模型名称')
parser.add_argument('--clean_data', default=1, type=int, help='是否清洗数据, 1: 清洗, 0: 不清洗')
parser.add_argument('--train_file', default='train.txt', type=str, help='训练集文件')
parser.add_argument('--val_file', default='val.txt', type=str, help='验证集文件')
parser.add_argument('--test_file', default='test.txt', type=str, help='测试集文件')
parser.add_argument('--case', default=1, type=int, help='是否区分大小写，1: 区分，0: 不区分')
parser.add_argument('--min_word_freq', default=1, type=int, help='最小词频')
parser.add_argument('--min_char_freq', default=1, type=int, help='最小字符频')
parser.add_argument('--pretrain_vector', default=1, type=int, help='是否使用预训练词向量, 1: 使用, 0: 不使用')
parser.add_argument('--reduce_vector', default=1, type=int, help='是否筛选预训练词向量, 1: 筛选, 0: 不筛选')
parser.add_argument('--pretrain_vector_file', default='un_word2vec_cbow_230720.txt', type=str, help='预训练词向量文件')
parser.add_argument('--max_sentence_length', default=64, type=int, help='最大文本长度')
parser.add_argument('--max_word_length', default=20, type=int, help='最大文本长度')
parser.add_argument('--train_batch_size', default=1024, type=int, help='训练集batch size')
parser.add_argument('--val_batch_size', default=2048, type=int, help='验证集batch size')
parser.add_argument('--test_batch_size', default=2048, type=int, help='测试集batch size')
parser.add_argument('--word_embedding_dim', default=300, type=int, help='词嵌入维度')
parser.add_argument('--char_embedding_dim', default=20, type=int, help='字符嵌入维度')
parser.add_argument('--num_filters', default=256, type=int, help='卷积核数量')
parser.add_argument('--filter_size', default=256, type=int, help='卷积核尺寸')
parser.add_argument('--hidden_dim', default=256, type=int, help='lstm隐藏层维度')
parser.add_argument('--num_layers', default=2, type=int, help='lstm层数')
parser.add_argument('--drop_out', default=0.5, type=float, help='随机神经元失活')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='学习率')
parser.add_argument('--lr_decay_step', default=5000, type=int, help='学习率衰减step数')
parser.add_argument('--lr_decay_gamma', default=0.7, type=float, help='学习率衰减比率')
parser.add_argument('--epochs', default=50, type=int, help='迭代次数')
parser.add_argument('--early_stop', default=10000, type=int, help='提前停止所需要batch数')
args = parser.parse_args()


if __name__ == '__main__':
    # torch线程数
    torch.set_num_threads(16)
    # 获取参数字典
    params_dict = args.__dict__
    # 全局随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样
    model_name = args.model_name
    if args.clean_data == 1:
        clean_data = True
    else:
        clean_data = False
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    if args.case == 1:
        case = True
    else:
        case = False
    min_word_freq = args.min_word_freq
    min_char_freq = args.min_char_freq
    if args.pretrain_vector == 1:
        pretrain_vector = True
    else:
        pretrain_vector = False
    if args.reduce_vector == 1:
        reduce_vector = True
    else:
        reduce_vector = False
    pretrain_vector_file = args.pretrain_vector_file
    max_sentence_length = args.max_sentence_length
    max_word_length = args.max_word_length
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    test_batch_size = args.test_batch_size
    if args.reduce_vector == 1:
        reduce_vector = True
    else:
        reduce_vector = False
    learning_rate = args.learning_rate
    lr_decay_step = args.lr_decay_step
    lr_decay_gamma = args.lr_decay_gamma
    epochs = args.epochs
    early_stop = args.early_stop

    conf_dir = 'conf'
    raw_data_dir = 'data/raw_data'
    process_data_dir = 'data/process_data'
    train_build_path = 'train.json'
    val_build_path = 'val.json'
    test_build_path = 'test.json'

    train_file_path = os.path.join(raw_data_dir,train_file)
    val_file_path = os.path.join(raw_data_dir,val_file)
    test_file_path = os.path.join(raw_data_dir,test_file)
    train_clean_data = os.path.join(process_data_dir,train_build_path)
    val_clean_data = os.path.join(process_data_dir,val_build_path)
    test_clean_data = os.path.join(process_data_dir,test_build_path)

    # 分词清洗数据集
    if clean_data:
        build_train_val_test(train_file_path, val_file_path, test_file_path, train_clean_data, val_clean_data, test_clean_data)

    # 生成词表和标签表
    print('生成词表...')
    word2id, char2id, tag2id = build_word2id_char2id_tag2id(train_clean_data, 
                                                            min_word_freq=min_word_freq, 
                                                            min_char_freq=min_char_freq,
                                                            case=case)
    print(f'词表大小：{len(word2id)}')

    # 筛选预训练词向量筛选
    if reduce_vector:
        word2vec_path = os.path.join(conf_dir, pretrain_vector_file)  # 预训练词向量模型路径
        reduce_vector_file = 'filter_' + pretrain_vector_file
        vocab = word2id  # 要筛选的词汇表
        pretrain_vector_path = os.path.join(conf_dir, reduce_vector_file)  # 输出筛选后预训练词向量文件路径
        filter_word2vec(vocab, pretrain_vector_path, word2vec_path)
        params_dict['pretrain_vector_file'] = reduce_vector_file
    else:
        pretrain_vector_path = os.path.join(conf_dir, pretrain_vector_file)
    # 构建预训练词向量的词表和标签表
    if pretrain_vector:
        print('生成预训练词向量词表...')
        word2id, char2id, tag2id = build_word2id_char2id_tag2id(data_path=train_clean_data, 
                                               pretrain_vector_path=pretrain_vector_path,
                                               min_word_freq=min_word_freq, 
                                               min_char_freq=min_char_freq,
                                               case=case)
        print(f'词表大小: {len(word2id)}')

    # 封装数据
    print('封装数据...')
    word2id_file = 'word2id.json'
    tag2id_file = 'tag2id.json'
    id2word_file = 'id2word.json'
    id2tag_file = 'id2tag.json'
    word2id_path = os.path.join(conf_dir,word2id_file)
    tag2id_path = os.path.join(conf_dir,tag2id_file)
    id2word_path = os.path.join(conf_dir,id2word_file)
    id2tag_path = os.path.join(conf_dir,id2tag_file)
    train_dataset = MyDataset(train_clean_data, word2id, tag2id, char2id, max_sentence_length, max_word_length)
    val_dataset = MyDataset(val_clean_data, word2id, tag2id, char2id, max_sentence_length, max_word_length)
    test_dataset = MyDataset(test_clean_data, word2id, tag2id, char2id, max_sentence_length, max_word_length)
    
    train_dataloader = DataLoader(train_dataset,batch_size=train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset,batch_size=val_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size)

    # 设置计算设备
    device = torch.device("cpu")

    # 初始化模型
    num_filters = args.num_filters
    filter_size = args.filter_size
    word_embedding_dim = args.word_embedding_dim
    char_embedding_dim = args.char_embedding_dim
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    drop_out = args.drop_out

    model = BilstmCnnCrf(
                word2id,
                char2id, 
                num_classes=len(tag2id), 
                word_embedding_dim=word_embedding_dim, 
                char_embedding_dim=char_embedding_dim,
                num_filters=num_filters,
                filter_size=filter_size,
                hidden_dim=hidden_dim, 
                num_layers=num_layers,
                drop_out=drop_out,
                embedding_pretrained=pretrain_vector, 
                pretrain_vector_path=pretrain_vector_path)
    model.to(device)
    init_network(model, 'kaiming')
    # example_inputs = [torch.randint(1,100, [1, 64]).to(device), ]
    # base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
    # print(f'FLOPs: {base_macs}  num_params: {base_nparams}')

    # 定义优化器和学习率衰减
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    # 训练
    model_saved_path = train(model=model, 
        model_name=model_name, 
        train_dataloader=train_dataloader, 
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer, 
        epochs=epochs, 
        scheduler=scheduler,
        early_stop=early_stop,
        tag2id_path=tag2id_path,
        id2word_path=id2word_path,
        id2tag_path=id2tag_path,
        device=device)
    params_dict['model_saved_path'] = model_saved_path
    # 保存参数字典
    write_json(params_dict, './conf/params_dict.json')
