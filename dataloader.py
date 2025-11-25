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
        train, vali = train_test_split(train_val, test_size=vali_ratio/(1-test_ratio), random_state=random_state)

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
            record['app_id'] = torch.tensor(row['app_id'], dtype=torch.long)
            record['userID'] = torch.tensor(row['userID'], dtype=torch.long)
            record['itemID'] = torch.tensor(row['itemID'], dtype=torch.long)
            record['scenario_id'] = torch.tensor(row['scenario_id'], dtype=torch.long)
            record['is_click'] = torch.tensor(row['is_click'], dtype=torch.float)
            record['is_play'] = torch.tensor(row['is_play'], dtype=torch.float)
            record['is_pay'] = torch.tensor(row['is_pay'], dtype=torch.float)

            fixed_features = [col for col in row.index if col.startswith('user_') or col.startswith('item_')]
            record['fixed'] = torch.tensor(row[fixed_features].values.astype(np.float32))

            domain_features = [col for col in row.index if any(d in col for d in ['game','video','music','theme','read'])
                               and ('_continue_' in col or '_concrete_' in col)]
            record['domain'] = torch.tensor(row[domain_features].values.astype(np.float32))

            sequence_features = [col for col in row.index if '_sequen' in col]
            for col in sequence_features:
                record[col] = np.array(json.loads(row[col]), dtype=np.float32)

            self.data.append(record)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """
    batch: list of dicts from FakeDataset
    """
    batch_out = {}

    # 先处理非序列字段，直接stack
    simple_fields = ['app_id', 'userID', 'itemID', 'scenario_id', 'is_click', 'is_play', 'is_pay', 'fixed', 'domain']
    for key in simple_fields:
        batch_out[key] = torch.stack([torch.tensor(b[key]) for b in batch], dim=0)

    # 处理序列字段，pad到batch中最长的序列
    seq_keys = [k for k in batch[0].keys() if k not in simple_fields]
    for key in seq_keys:
        # 将每个序列转为 tensor
        seqs = [torch.tensor(b[key]) for b in batch]
        # pad_sequence 会返回 [max_len, batch_size]，需要 transpose
        padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
        batch_out[key] = padded  # shape: [batch_size, max_len]

    return batch_out