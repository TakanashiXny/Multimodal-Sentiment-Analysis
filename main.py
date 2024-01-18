import pandas as pd
from train import train_process
from model import CLModel, myModel, myModel2
import argparse
from transformers import BertConfig, AutoTokenizer
import torch
import numpy as np
from train import train_process
from prediction import predict

def parse_arguments():
    parser = argparse.ArgumentParser(description='parameters')  # 创建解析器
    parser.add_argument('--model', type=str, default='cat', help='input the model')  # 添加参数
    parser.add_argument('--epoch', type=int, default='10', help='训练轮数')  # 添加参数
    parser.add_argument('--warmup', type=int, default='20', help='预训练步数')  # 添加参数
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='学习率衰减')  # 添加参数
    parser.add_argument('--lr', type=float, default=4e-6, help='学习率')  # 添加参数
    parser.add_argument('--train', type=str, default='train', help='训练或测试')  # 添加参数

    args = parser.parse_args()  # 解析参数
    return args

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {"back_translation": True, 'batch_size': 32, 'train_size': 0.8, 'fuse_type': 'att', 
              'train_dim': 768, 'activate_fun': 'gelu', 'text_model': 'bert-base', 'image_model': 'resnet-50',
              'temperature': 0.07, 'device': device, 'optim': 'adamW',
              'dropout': 0.3, 'model': 'CLModel', 'epoch': 20, 'weight_decay': 1e-2,
              'warmup': 20, 'learning_rate': 5e-5, 'max_seq': 64}
    
    df_for_train = pd.read_csv('./data/train_text.csv')
    df_for_test = pd.read_csv('./data/test_text.txt')

    torch.manual_seed(100)
    torch.cuda.manual_seed(100)
    torch.cuda.manual_seed_all(100)
    np.random.seed(100)
    """在种子不变的情况下保证结果一致"""
    torch.backends.cudnn.deterministic = True

    # model = CLModel(config)
    # model = model.to(device)

    
    bert_tokenizer = AutoTokenizer.from_pretrained('./model/bert_tokenizer')
    
    bert_config = BertConfig('./bert')

    epoch = args.epoch
    warmup = args.warmup
    weight_decay = args.weight_decay
    learning_rate = args.lr

    if args.train == 'train':

        if args.model == 'cat':
            model = myModel()
            model = model.to(device)
            train_process(model, 
                        df_for_train,
                        back_translation=False,
                        epoch=epoch, 
                        batch_size=32, 
                        learning_rate=learning_rate,
                        warmup=warmup, 
                        weight_decay=weight_decay, 
                        unfusion=0, 
                        device=device)
        elif args.model == 'add':
            model = myModel2()
            model = model.to(device)
            train_process(model, 
                        df_for_train,
                        back_translation=False,
                        epoch=epoch, 
                        batch_size=32, 
                        learning_rate=learning_rate,
                        warmup=warmup, 
                        weight_decay=weight_decay, 
                        unfusion=0, 
                        device=device)
        elif args.model == 'CLMLF':
            model = CLModel()
            model = model.to(device)
            train_process(model, 
                        df_for_train,
                        back_translation=True,
                        epoch=epoch, 
                        batch_size=32, 
                        learning_rate=learning_rate,
                        warmup=warmup, 
                        weight_decay=weight_decay, 
                        unfusion=0, 
                        device=device)
    
    else:
        model = myModel()
        model = model.to(device)
        model.load_state_dict(torch.load('./model/model.pt'))
        predict(model, df_for_test, 32)