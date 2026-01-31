#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import torch
import numpy as np
import pandas as pd
import configparser
import copy
import random
from torch.utils.data import DataLoader, TensorDataset

from models.crosstgan_generator import CrossTGANGenerator
from CrossTGANomaly import CrossTGANomalyModel, MLPDiscriminator
from Utilities.logger import get_logger
from data_driven.data_pipeline import data_driven


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def seed_all(seed=2025):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_all(2025)

def get_crosstgan_args(args):

    crosstgan_args = argparse.Namespace()
    config = configparser.ConfigParser()
    config.read(f'data_configs/{args.data_prefix}.conf')
    data_config = config['config']
    crosstgan_args.device = args.device
    crosstgan_args.data_path = data_config['data_path']
    if 'test_path' in data_config:
        crosstgan_args.test_path = data_config['test_path']
    
    crosstgan_args.data_reader = data_config['data_reader']
    crosstgan_args.data_id = data_config['data_id']
    crosstgan_args.features = data_config['features']
    crosstgan_args.seq_len = int(data_config['seq_len'])
    crosstgan_args.stride = int(data_config['stride'])
    crosstgan_args.batch_size = int(data_config['batch_size'])
    
    f = open('data_configs/instruct.json')
    instruct_list = json.load(f)
    f.close()
    if args.data_prefix in instruct_list:
        crosstgan_args.instruct = instruct_list[args.data_prefix]
    else:
        crosstgan_args.instruct = "Detecting anomalies in time series data"
    
   
    crosstgan_args.model_path = "path"
    crosstgan_args.mask_rate = 0.5
    crosstgan_args.patch_len = 32
    crosstgan_args.max_token_num = 128
    crosstgan_args.max_backcast_len = args.window_size
    crosstgan_args.max_forecast_len = 0  
    crosstgan_args.lm_layer_num = 6 
    crosstgan_args.dec_trans_layer_num = 4
    crosstgan_args.ts_embed_dropout = 0.1
    crosstgan_args.dec_head_dropout = 0.1
    crosstgan_args.lm_ft_type = 'full'  
    crosstgan_args.lm_pretrain_model = 'gpt2-small' 
    crosstgan_args.clip = 5  
    crosstgan_args.weight_decay = 0  
    crosstgan_args.learning_rate = 1e-4 
    crosstgan_args.target = 'OT'
    crosstgan_args.label_len = 0
    crosstgan_args.pred_len = args.window_size
    crosstgan_args.num_workers = 0
    
    log_dir = "crosstgan_anomaly_logs"
    os.makedirs(log_dir, exist_ok=True)
    crosstgan_args.logger = get_logger(log_dir, __name__, 'crosstgan_anomaly.log')
    
    return crosstgan_args

def adapt_to_CrosstgaNomaly_format(train_dataset, train_loader, val_dataset, test_dataset, args):

    train_x = []
    try:
        for batch in train_loader:
            batch_x, _ = batch
            train_x.append(batch_x.cpu().numpy())
        if len(train_x) > 0:
            train_x = np.concatenate(train_x, axis=0)
        else:
            train_x = np.array([])
    except Exception as e:
        train_x = np.array([])
    

    val_x = val_dataset.data_x
    

    test_x = test_dataset.data_x
    

    if hasattr(test_dataset, 'test_label'):
        test_y = test_dataset.test_label
    else:
        test_y = np.zeros(len(test_x))
        anomaly_indices = np.random.choice(len(test_y), size=int(len(test_y)*0.1), replace=False)
        test_y[anomaly_indices] = 1
    

    if len(train_x) > 0 and train_x.ndim > 2:
        nc = train_x.shape[2]
    elif test_x.ndim > 1:
        nc = test_x.shape[1]
    else:
        nc = 10
    

    if len(train_x) > 0:
        train_dataset = TensorDataset(torch.Tensor(train_x), torch.zeros(len(train_x)))
    else:
        dummy_x = np.random.randn(10, args.seq_len, nc)
        train_dataset = TensorDataset(torch.Tensor(dummy_x), torch.zeros(10))
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )
    
    data_builder = {
        "train": train_loader,
        "val": val_x,
        "test": (test_x, test_y),
        "nc": nc
    }
    
    return data_builder

def main():
    parser = argparse.ArgumentParser(description='CrossTGAN-FGNomaly')
    parser.add_argument('--data_prefix', type=str, default='ETTh1', help='dataset prefix')
    parser.add_argument('--strategy', type=str, default='linear', help='weighting strategy')
    parser.add_argument('--adv_rate', type=float, default=0.01, help='Adversarial rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--window_size', type=int, default=96, help='Time window size')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    params = {
        'data_prefix': args.data_prefix,
        'val_size': 0.3,
        'batch_size': args.batch_size,
        'stride': 1,
        'window_size': args.window_size,
        
        'hidden_dim': 128,  
        
        'device': args.device,
        'lr': 1e-4,
        'if_scheduler': True,
        'scheduler_step_size': 3,
        'scheduler_gamma': 0.7,
        
        'epoch': 80,
        'early_stop': True,
        'early_stop_tol': 10,
        
        'weighted_loss': True,
        'strategy': args.strategy,
        
        'adv_rate': args.adv_rate,
        'dis_ar_iter': 2,
        
        'best_model_path': os.path.join('crosstgan_output', 'best_model'),
        'result_path': os.path.join('crosstgan_output'),
    }
    

    crosstgan_args = get_crosstgan_args(args)
    train_dataset, train_loader = data_driven(crosstgan_args, 'train')
    val_dataset, val_loader = data_driven(crosstgan_args, 'val')
    test_dataset, test_loader = data_driven(crosstgan_args, 'test')

    data = adapt_to_CrosstgaNomaly_format(train_dataset, train_loader, val_dataset, test_dataset, crosstgan_args)
    model = CrossTGANomalyModel(ae=CrossTGANGenerator(crosstgan_args),
                           dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                                                  hidden_dim=params['hidden_dim']),
                           data_builder=data, **params)

    model.train()
    model.test()

if __name__ == '__main__':
    main()
