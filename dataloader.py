import pandas as pd
import torch
from torch.utils.data import Dataset
import json
import numpy as np
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence


class FakeDataset(Dataset):
    def __init__(self, csv_file, split='train', vali_ratio=0.1, test_ratio=0.1, random_state=42):
        """
        split: 'train', 'vali', 'test'
        vali_ratio: 验证集比例
        test_ratio: 测试集比例
        """
        df = pd.read_csv(csv_file)

        # 划分 train/vali/test
        train_val, test = train_test_split(df, test_size=test_ratio, random_state=random_state)
        train, vali = train_test_split(train_val, test_size=vali_ratio / (1 - test_ratio), random_state=random_state)

        if split == 'train':
            self.df = train.reset_index(drop=True)
        elif split == 'vali':
            self.df = vali.reset_index(drop=True)
        elif split == 'test':
            self.df = test.reset_index(drop=True)
        else:
            raise ValueError("split must be 'train', 'vali', or 'test'")

        self.data = []
        for _, row in self.df.iterrows():
            record = {}
            record['domain_id'] = torch.tensor(row['domain_id'], dtype=torch.long)
            record['userID'] = torch.tensor(row['userID'], dtype=torch.long)
            record['itemID'] = torch.tensor(row['itemID'], dtype=torch.long)
            record['scenario_id'] = torch.tensor(row['scenario_id'], dtype=torch.long)
            record['is_click'] = torch.tensor(row['is_click'], dtype=torch.long)
            record['is_play'] = torch.tensor(row['is_play'], dtype=torch.long)
            record['is_pay'] = torch.tensor(row['is_pay'], dtype=torch.long)

            record['image_features'] = np.array(json.loads(row['image_features']), dtype=np.float32)
            record['text_features'] = np.array(json.loads(row['text_features']), dtype=np.float32)

            self.important_features = [col for col in row.index if col.startswith('user_') or col.startswith('item_')]

            for col in self.important_features:
                record[col] = torch.tensor(row[col], dtype=torch.float32)

            self.continuous_features_dim = {col: 1 for col in row.index if
                                            any(d in col for d in ['game', 'video', 'music', 'theme', 'read'])
                                            and ('_continue_' in col)}

            for col in self.continuous_features_dim.keys():
                record[col] = torch.tensor(row[col], dtype=torch.float32)

            for col in self.important_features:
                self.continuous_features_dim[col] = 1

            self.categorical_features = {col: 100 for col in row.index if
                                         any(d in col for d in ['game', 'video', 'music', 'theme', 'read'])
                                         and ('_concrete_' in col)}

            for col in self.categorical_features.keys():
                record[col] = torch.tensor(row[col], dtype=torch.long)

            self.categorical_features['domain_id'] = 5
            self.categorical_features['userID'] = 100000
            self.categorical_features['itemID'] = 100000
            self.categorical_features['scenario_id'] = 3
            self.categorical_features['is_click'] = 3
            self.categorical_features['is_play'] = 3
            self.categorical_features['is_pay'] = 3

            self.sequence_features = [col for col in row.index if '_sequen' in col]
            for col in self.sequence_features:
                record[col] = np.array(json.loads(row[col]), dtype=np.float32)
                self.continuous_features_dim[col] = 1

            self.data.append(record)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_features(self):
        return self.important_features, self.categorical_features, self.continuous_features_dim, self.sequence_features


def collate_fn(batch):
    """
    batch: list of dicts from FakeDataset
    自动分类：
        - categorical: stack
        - continuous: stack
        - sequence: pad
    """

    batch_out = {}

    # 取第一个样本，用来识别特征
    sample = batch[0]

    # 1) categorical 特征
    categorical_keys = [k for k in sample.keys()
                        if (isinstance(sample[k], torch.Tensor) and sample[k].dtype in (torch.int, torch.long))]

    for key in categorical_keys:
        batch_out[key] = torch.stack([b[key] for b in batch], dim=0)

    # 2) continuous 特征
    continuous_keys = [k for k in sample.keys()
                       if (isinstance(sample[k], torch.Tensor) and sample[k].dtype == torch.float32)]

    # 去掉 categorical 重复
    continuous_keys = [k for k in continuous_keys if k not in categorical_keys]

    for key in continuous_keys:
        batch_out[key] = torch.stack([b[key] for b in batch], dim=0)

    # 3) sequence 特征（numpy array → padding）
    sequence_keys = [k for k in sample.keys()
                     if isinstance(sample[k], np.ndarray)]

    for key in sequence_keys:
        seq_list = [torch.tensor(b[key], dtype=torch.float32) for b in batch]
        padded = pad_sequence(seq_list, batch_first=True, padding_value=0.0)
        batch_out[key] = padded

    return batch_out
