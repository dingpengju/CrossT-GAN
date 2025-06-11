#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import configparser
import copy
import random
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.preprocessing import StandardScaler

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

class FeatureAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(FeatureAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
    
    def encode(self, x):
        return self.encoder(x)

def get_crosstgan_args(data_prefix, args):

    crosstgan_args = argparse.Namespace()
    
    config = configparser.ConfigParser()
    config_path = f'data_configs/{data_prefix}.conf'
    if not os.path.exists(config_path):
        return None
        
    config.read(config_path)
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
    if data_prefix in instruct_list:
        crosstgan_args.instruct = instruct_list[data_prefix]
    else:
        crosstgan_args.instruct = "Detecting anomalies in time series data"
    
    crosstgan_args.model_path = "path"
    crosstgan_args.mask_rate = 0.5
    crosstgan_args.patch_len = 16
    crosstgan_args.max_token_num = 128
    crosstgan_args.max_backcast_len = args.window_size
    crosstgan_args.max_forecast_len = 0 
    crosstgan_args.lm_layer_num = 4  
    crosstgan_args.dec_trans_layer_num = 2
    crosstgan_args.ts_embed_dropout = 0.3
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
    
    return crosstgan_args

def load_multiple_datasets(datasets, args, flag='train'):

    all_datasets = []
    all_loaders = []
    
    for dataset in datasets:
        crosstgan_args = get_crosstgan_args(dataset, args)
        if crosstgan_args is None:
            continue

        if not hasattr(crosstgan_args, 'logger'):
            log_dir = "crosstgan_anomaly_logs"
            os.makedirs(log_dir, exist_ok=True)
            crosstgan_args.logger = get_logger(log_dir, __name__, 'crosstgan_anomaly.log')
            
        dataset_obj, loader = data_driven(crosstgan_args, flag)
        all_datasets.append(dataset_obj)
        all_loaders.append(loader)
    
    return all_datasets, all_loaders

def train_autoencoder(data_list, latent_dim, device, num_s=1):

    models = []
    

    for i, data in enumerate(data_list):
        samples, seq_len, features = data.shape
        data_2d = data.reshape(samples * seq_len, features)
        scaler = StandardScaler()
        data_2d = scaler.fit_transform(data_2d) 
        hidden_dim = min(128, features * 2) 
        model = FeatureAutoencoder(features, hidden_dim, latent_dim).to(device)
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data_2d).to(device))
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)  
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in loader:
                x = batch[0]
                x_recon, _ = model(x)
                loss = criterion(x_recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
        
        models.append((model, scaler))
    
    return models

def adapt_to_CrosstgaNomaly_format(train_datasets, train_loaders, val_datasets, test_dataset, test_loader, args):
    
    all_train_data = []
    try:
        if isinstance(train_loaders, list):
           
            for i, loader in enumerate(train_loaders):
                dataset_samples = []
                for batch in loader:
                    batch_x, _ = batch
                    dataset_samples.append(batch_x.cpu().numpy())
                
                if dataset_samples:
                    dataset_data = np.concatenate(dataset_samples, axis=0)
                    all_train_data.append(dataset_data)
        else:

            single_dataset = []
            for batch in train_loaders:
                batch_x, _ = batch
                single_dataset.append(batch_x.cpu().numpy())
            
            if single_dataset:
                single_dataset = np.concatenate(single_dataset, axis=0)
                all_train_data.append(single_dataset)
    except Exception as e:
        import traceback
        traceback.print_exc()

    val_x = val_datasets[0].data_x if val_datasets else np.array([])
    test_x = test_dataset.data_x

    if hasattr(test_dataset, 'test_label'):
        test_y = test_dataset.test_label
    else:
        test_y = np.zeros(len(test_x))
        anomaly_indices = np.random.choice(len(test_y), size=int(len(test_y)*0.1), replace=False)
        test_y[anomaly_indices] = 1

    if not all_train_data:
        dummy_x = np.random.randn(10, args.window_size, 20)  
        train_dataset = TensorDataset(torch.Tensor(dummy_x), torch.zeros(10))
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        
        data_builder = {
            "train": train_loader,
            "val": val_x,
            "test": (test_x, test_y),
            "nc": 20  
        }
        
        return data_builder
    

    latent_dim = args.latent_dim if hasattr(args, 'latent_dim') else 32
    autoencoder_models = train_autoencoder(all_train_data, latent_dim, args.device, num_epochs=6)
    

    transformed_datasets = []
    for i, (data, (model, scaler)) in enumerate(zip(all_train_data, autoencoder_models)):
        samples, seq_len, features = data.shape
        
        data_2d = data.reshape(samples * seq_len, features)
        data_2d = scaler.transform(data_2d)

        data_tensor = torch.FloatTensor(data_2d).to(args.device)

        model.eval()
        with torch.no_grad():
            encoded_data = model.encode(data_tensor).cpu().numpy()

        transformed_data = encoded_data.reshape(samples, seq_len, latent_dim)
        transformed_datasets.append(transformed_data)
    
    train_x = np.concatenate(transformed_datasets, axis=0)
    
    if len(test_x) > 0:
        test_samples, test_seq_len, test_features = test_x.shape
    
        test_2d = test_x.reshape(test_samples * test_seq_len, test_features)

        scaler = StandardScaler()
        test_2d = scaler.fit_transform(test_2d)
        hidden_dim = min(128, test_features * 2)
        test_model = FeatureAutoencoder(test_features, hidden_dim, latent_dim).to(args.device)
        test_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(test_2d).to(args.device))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=True)
        optimizer = optim.Adam(test_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        test_model.train()
        for epoch in range(6):  
            total_loss = 0
            for batch in test_loader:

                x = batch[0]
                x_recon, _ = test_model(x)
                loss = criterion(x_recon, x)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()         
                total_loss += loss.item()

        test_model.eval()
        with torch.no_grad():
            test_tensor = torch.FloatTensor(test_2d).to(args.device)
            test_encoded = test_model.encode(test_tensor).cpu().numpy()
        
        test_x = test_encoded.reshape(test_samples, test_seq_len, latent_dim)

    nc = latent_dim

    train_dataset = TensorDataset(torch.Tensor(train_x), torch.zeros(len(train_x)))
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    parser = argparse.ArgumentParser(description='Cross-domain timing data anomaly detection (self-encoder projection)')
    parser.add_argument('--train_datasets', type=str, default='MSL,SMAP,SMD', 
                        help='Training dataset with multiple datasets separated by commas')
    parser.add_argument('--test_dataset', type=str, default='SWaT', 
                        help='test dateset')
    parser.add_argument('--strategy', type=str, default='linear', 
                        help='weighting strategy')
    parser.add_argument('--adv_rate', type=float, default=0.001, 
                        help='adversarial rate')
    parser.add_argument('--gpu', type=int, default=0, 
                        help='GPU ID')
    parser.add_argument('--window_size', type=int, default=96, 
                        help='Time window size')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='batch size')
    parser.add_argument('--latent_dim', type=int, default=128, 
                        help='Self-encoder potential spatial dimensions')
    args = parser.parse_args()
    
    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    
    train_datasets = args.train_datasets.split(',')
    test_dataset = args.test_dataset

    params = {
        'val_size': 0.3,
        'batch_size': args.batch_size,
        'stride': 1,
        'window_size': args.window_size,
        
        'hidden_dim': 128,  
        
        'device': args.device,
        'lr': 1e-4,
        'if_scheduler': True,
        'scheduler_step_size': 5,
        'scheduler_gamma': 0.5,
        
        'epoch': 80,
        'early_stop': True,
        'early_stop_tol': 10,
        
        'weighted_loss': True,
        'strategy': args.strategy,
        
        'adv_rate': args.adv_rate,
        'dis_ar_iter': 1,
        
        'best_model_path': os.path.join('crosstgan_output', 'best_model'),
        'result_path': os.path.join('crosstgan_output'),
    }
    
    
    log_dir = "cross_domain_logs"
    os.makedirs(log_dir, exist_ok=True)
    logger = get_logger(log_dir, __name__, 'main_crosstgan_domain.log')
    train_datasets_obj, train_loaders = load_multiple_datasets(train_datasets, args, 'train')
    val_datasets_obj, val_loaders = load_multiple_datasets(train_datasets, args, 'val') 
    test_args = get_crosstgan_args(test_dataset, args)
    if test_args is None:
        return

    if not hasattr(test_args, 'logger'):
        test_args.logger = logger
        
    test_dataset_obj, test_loader = (test_args, 'test')

    data = adapt_to_CrosstgaNomaly_format(train_datasets_obj, train_loaders, val_datasets_obj, test_dataset_obj, test_loader, args)

    params['data_prefix'] = f"{'_'.join(train_datasets)}_to_{test_dataset}_ae"

    model = CrossTGANomalyModel(ae=CrossTGANGenerator(test_args),
                         dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                                                hidden_dim=params['hidden_dim']),
                         data_builder=data, **params)
    

    model.train()
    model.test()

if __name__ == '__main__':
    main() 