import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import argparse
import numpy as np
from dataloader import FakeDataset, collate_fn
from collections import defaultdict
from star_cdr import CDIRecModel

class SimpleModel(nn.Module):
    def __init__(self, fixed_dim, domain_dim):
        super().__init__()
        self.fc1 = nn.Linear(fixed_dim + domain_dim, 64)
        self.fc_click = nn.Linear(64, 1)
        self.fc_play = nn.Linear(64, 1)
        self.fc_pay = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        x = torch.cat([batch['fixed'], batch['domain']], dim=-1)
        x = torch.relu(self.fc1(x))

        pred_click = self.sigmoid(self.fc_click(x)).squeeze(-1)
        pred_play = self.sigmoid(self.fc_play(x)).squeeze(-1)
        pred_pay = self.sigmoid(self.fc_pay(x)).squeeze(-1)

        # 对每个标签计算 BCELoss
        loss_click = nn.BCELoss()(pred_click, batch['is_click'])
        loss_play = nn.BCELoss()(pred_play, batch['is_play'])
        loss_pay = nn.BCELoss()(pred_pay, batch['is_pay'])
        total_loss = loss_click + loss_play + loss_pay

        return total_loss, pred_click, pred_play, pred_pay


# ------------------ 验证函数 ------------------
def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            out = model(batch)
            loss = out['loss']
            total_loss += loss.item() * batch['is_click'].size(0)
            all_labels.append(batch['is_click'].cpu().numpy())
            all_preds.append(out['pred_click'].cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, auc


# ------------------ 验证函数（按 app_id 统计，model 输出元组） ------------------
def evaluate_by_app(model, dataloader, device):
    model.eval()

    all_labels = defaultdict(lambda: defaultdict(list))
    all_preds = defaultdict(lambda: defaultdict(list))
    all_losses = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            # 将 tensor 移到 device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # 假设 model 输出 (loss, pred_click, pred_play, pred_pay)
            out = model(batch)
            loss, pred_click, pred_play, pred_pay = out

            app_ids = batch['app_id'].cpu().numpy()
            labels = {k: batch[k].cpu().numpy() for k in ['is_click', 'is_play', 'is_pay']}
            preds = {
                'is_click': pred_click.cpu().numpy(),
                'is_play': pred_play.cpu().numpy(),
                'is_pay': pred_pay.cpu().numpy()
            }

            for i, app_id in enumerate(app_ids):
                all_losses[app_id].append(loss.item())
                for label in ['is_click', 'is_play', 'is_pay']:
                    all_labels[app_id][label].append(labels[label][i])
                    all_preds[app_id][label].append(preds[label][i])

    # 统计每个 app_id 的 loss 和 auc
    result_loss = {}
    result_auc = {}
    for app_id in all_labels:
        result_loss[app_id] = np.mean(all_losses[app_id])
        result_auc[app_id] = {}
        for label in ['is_click', 'is_play', 'is_pay']:
            try:
                result_auc[app_id][label] = roc_auc_score(
                    np.concatenate(all_labels[app_id][label]),
                    np.concatenate(all_preds[app_id][label])
                )
            except ValueError:
                result_auc[app_id][label] = float('nan')

    return result_loss, result_auc


# ------------------ 训练函数 ------------------
# def train_one_epoch(model, dataloader, optimizer, device, vali_loader=None, vali_step=50):
#     model.train()
#     all_labels = []
#     all_preds = []
#     total_loss = 0
#     step = 0
#     for batch in dataloader:
#         step += 1
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(device)
#         optimizer.zero_grad()
#         out = model(batch)
#         loss = out['loss']
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item() * batch['is_click'].size(0)
#         all_labels.append(batch['is_click'].cpu().numpy())
#         all_preds.append(out['pred_click'].cpu().numpy())
#
#         # 每隔 vali_step 测试一次
#         if vali_loader is not None and step % vali_step == 0:
#             vali_loss, vali_auc = evaluate(model, vali_loader, device)
#             print(f"  Step {step} | Val Loss: {vali_loss:.4f} | Val AUC: {vali_auc:.4f}")
#
#     all_labels = np.concatenate(all_labels)
#     all_preds = np.concatenate(all_preds)
#     auc = roc_auc_score(all_labels, all_preds)
#     avg_loss = total_loss / len(dataloader.dataset)
#     return avg_loss, auc


# ------------------ 训练函数（按 app_id 统计） ------------------
def train_one_epoch(model, dataloader, optimizer, device, vali_loader=None, vali_step=50):
    model.train()

    all_labels = defaultdict(lambda: defaultdict(list))
    all_preds = defaultdict(lambda: defaultdict(list))
    all_losses = defaultdict(list)

    step = 0
    for batch in dataloader:
        step += 1

        # 将 tensor 移到 device
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        optimizer.zero_grad()

        # 假设 model 输出 (loss, pred_click, pred_play, pred_pay)
        out = model(batch)
        loss, pred_click, pred_play, pred_pay = out
        loss.backward()
        optimizer.step()

        app_ids = batch['app_id'].cpu().numpy()
        labels = {k: batch[k].cpu().numpy() for k in ['is_click', 'is_play', 'is_pay']}
        preds = {
            'is_click': pred_click.detach().cpu().numpy(),
            'is_play': pred_play.detach().cpu().numpy(),
            'is_pay': pred_pay.detach().cpu().numpy()
        }

        for i, app_id in enumerate(app_ids):
            all_losses[app_id].append(loss.item())  # 或按样本加权
            for label in ['is_click', 'is_play', 'is_pay']:
                all_labels[app_id][label].append(labels[label][i])
                all_preds[app_id][label].append(preds[label][i])

        # 验证集
        if vali_loader is not None and step % vali_step == 0:
            vali_loss_dict, vali_auc_dict = evaluate_by_app(model, vali_loader, device)
            print(f"Step {step} | Val Metrics per App:")
            format_app_metrics(vali_loss_dict, vali_auc_dict, "Validation")
            print("\n")

    # 统计每个 app_id 的 loss 和 auc
    result_loss = {}
    result_auc = {}
    for app_id in all_labels:
        result_loss[app_id] = np.mean(all_losses[app_id])
        result_auc[app_id] = {}
        for label in ['is_click', 'is_play', 'is_pay']:
            try:
                result_auc[app_id][label] = roc_auc_score(
                    np.concatenate(all_labels[app_id][label]),
                    np.concatenate(all_preds[app_id][label])
                )
            except ValueError:
                result_auc[app_id][label] = float('nan')

    return result_loss, result_auc

def format_app_metrics(loss_dict, auc_dict, dataset_name):
    print(f"\n{dataset_name} Metrics:")
    for app_id in loss_dict:
        loss_val = loss_dict[app_id]
        auc_vals = []
        for label in ['is_click', 'is_play', 'is_pay']:
            auc = auc_dict[app_id][label]
            auc_str = f"{auc:.4f}" if not np.isnan(auc) else "nan"
            auc_vals.append(f"{label} AUC: {auc_str}")
        print(f"App {app_id:<5} | Loss: {loss_val:.4f} | " + " | ".join(auc_vals))

# ------------------ 主函数 ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="dataset/my_fake_dataset.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--vali_step", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & DataLoader
    train_dataset = FakeDataset(args.csv_file, split='train')
    vali_dataset = FakeDataset(args.csv_file, split='vali')
    test_dataset = FakeDataset(args.csv_file, split='test')

    important_features, categorical_features, continuous_features_dim, sequence_features = train_dataset.get_features()

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    #
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    # vali_loader = DataLoader(vali_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # # 获取维度
    # sample = train_dataset[0]
    # fixed_dim = sample['fixed'].shape[0]
    # domain_dim = sample['domain'].shape[0]

    # 模型
    # model = SimpleModel(fixed_dim, domain_dim).to(device)
    model = CDIRecModel(important_features=important_features, categorical_features=categorical_features, continuous_features_dim=continuous_features_dim, sequence_features=sequence_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------ 训练 ------------------
    for epoch in range(1, args.epochs + 1):
        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, device, vali_loader, args.vali_step)
        vali_loss, vali_auc = evaluate_by_app(model, vali_loader, device)
        test_loss, test_auc = evaluate_by_app(model, test_loader, device)
        print(f"Epoch {epoch}\n" + "="*50)
        format_app_metrics(train_loss, train_auc, "Train")
        format_app_metrics(vali_loss, vali_auc, "Validation")
        format_app_metrics(test_loss, test_auc, "Test")
        print("="*50 + "\n")


if __name__ == "__main__":
    main()
