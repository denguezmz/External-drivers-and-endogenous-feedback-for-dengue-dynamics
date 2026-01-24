import matplotlib.pyplot as plt
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, init
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

n_weeks = len(vc_np) // 7
remainder = len(vc_np) % 7
caseim_data = Import[365:].detach().numpy().flatten()
caselo_data = Local[365:].detach().numpy().flatten()
casT_data = T[365:].detach().numpy().flatten()
caseR_data = R[365:].detach().numpy().flatten()
# Weekly averages for complete weeks
vc_np_weekly= [vc_np[i*7:(i+1)*7].mean() for i in range(n_weeks)]
Local_weekly = [caselo_data[i*7:(i+1)*7].sum() for i in range(n_weeks)]
Import_weekly = [caseim_data[i*7:(i+1)*7].sum() for i in range(n_weeks)]
T_weekly= [casT_data[i*7:(i+1)*7].mean() for i in range(n_weeks)]
R_weekly= [caseR_data[i*7:(i+1)*7].mean() for i in range(n_weeks)]

data = np.column_stack((vc_np_weekly, Local_weekly, Import_weekly, T_weekly, R_weekly))


total_weekly = np.array(Local_weekly) + np.array(Import_weekly)
labels = np.zeros_like(total_weekly, dtype=np.int64)

labels[total_weekly >= 20] = 1
labels[total_weekly >= 100] = 2
labels[total_weekly >= 300] = 3
X, y, total_cases = [], [], []
window_size = 4
for i in range(len(data) - window_size - 1):
    X.append(data[i : i + window_size])
    y.append([labels[i+window_size], labels[i+window_size+1]])
    total_cases.append(total_weekly[i + window_size])

X1 = np.array(X)
y1 = np.array(y)
total_cases = np.array(total_cases)

X = np.array(X)
y = np.array(y)

from sklearn.preprocessing import StandardScaler
ns, ws, nf = X.shape
X_flat = X.reshape(ns*ws, nf)
scaler = StandardScaler().fit(X_flat)
X = scaler.transform(X_flat).reshape(ns, ws, nf)


X_train, X_test, y_train, y_test, cases_train, cases_test = train_test_split(
    X, y, total_cases, test_size=0.3, random_state=150
)


def to_tensor_dataset(X_np, y_np):
    X_t = torch.tensor(X_np, dtype=torch.float32)
    y_t = torch.tensor(y_np, dtype=torch.long)
    y1 = y_t[:, 0]
    y2 = y_t[:, 1]
    return TensorDataset(X_t, y1, y2)

train_ds = to_tensor_dataset(X_train, y_train)
test_ds  = to_tensor_dataset(X_test,  y_test)

class DiseasePredictorLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=5, num_classes=4):
        super(DiseasePredictorLSTM, self).__init__()
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # dropout = 0.2 if num_layers > 1 else 0
        )

        self.lstm2 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size//2,
            num_layers=num_layers,
            batch_first=True,
            # dropout=0.2 if num_layers > 1 else 0
        )

        lstm1_features = hidden_size
        lstm2_features = hidden_size // 2
        combined_features = lstm1_features + lstm2_features

        self.fc1 = nn.Sequential(
            nn.Linear(combined_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(combined_features + num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm1_out, (hn1, cn1) = self.lstm1(x)
        lstm1_context = hn1[-1]
        lstm2_out, (hn2, cn2) = self.lstm2(x)
        lstm2_context = hn2[-1]
        combined_features = torch.cat((lstm1_context, lstm2_context), dim=1)
        pred_week5 = self.fc1(combined_features)
        combined_with_pred = torch.cat((combined_features, pred_week5), dim=1)
        pred_week6 = self.fc2(combined_with_pred)
        return pred_week5, pred_week6


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = DiseasePredictorLSTM(input_size=5, hidden_size=128, num_layers=2, num_classes=4).to(device)
import copy
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

def evaluate_model(model, loader, device):
    model.eval()
    pred5, true5, pred6, true6 = [], [], [], []

    with torch.no_grad():
        for Xb, y5b, y6b in loader:
            Xb = Xb.to(device)
            p5, p6 = model(Xb)
            pred5.extend(p5.argmax(1).cpu().numpy())
            true5.extend(y5b.numpy())
            pred6.extend(p6.argmax(1).cpu().numpy())
            true6.extend(y6b.numpy())

    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prf = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
        prf_macro = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        prf_micro = precision_recall_fscore_support(y_true, y_pred, average='micro', zero_division=0)
        prf_weighted = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        return {
            'accuracy': acc,
            'macro_precision': prf_macro[0],
            'macro_recall': prf_macro[1],
            'macro_f1': prf_macro[2],
            'micro_precision': prf_micro[0],
            'micro_recall': prf_micro[1],
            'micro_f1': prf_micro[2],
            'weighted_precision': prf_weighted[0],
            'weighted_recall': prf_weighted[1],
            'weighted_f1': prf_weighted[2],
            'class_precision': prf[0],
            'class_recall': prf[1],
            'class_f1': prf[2],
            'report': classification_report(y_true, y_pred, zero_division=0),
            'matrix': confusion_matrix(y_true, y_pred)
        }

    return {
        'week5': get_metrics(true5, pred5),
        'week6': get_metrics(true6, pred6)
    }

def cross_validate2(model, X_train_final, y_train_final, X_test, y_test, cases_train, cases_test,
                   n_splits=5, num_epochs=2000, lr=0.001, early_stop_patience=30):
    class EarlyStopping:
        def __init__(self, patience=100, verbose=False, delta=0):
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = float('inf')
            self.delta = delta
            self.best_model = None

        def __call__(self, val_loss, model):
            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

        def save_checkpoint(self, val_loss, model):
            if val_loss < self.val_loss_min - self.delta:
                if self.verbose:
                    print(f'Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}). Saving model...')
                self.val_loss_min = val_loss
                self.best_model = copy.deepcopy(model.state_dict())

    def oversample_stratified(X, y, cases, target_samples=None):

        y_week1 = y[:, 0]
        class_counts = Counter(y_week1)
        if target_samples is None:
            target_samples = max(class_counts.values())
        X_res, y_res, cases_res = [], [], []
        for cls in class_counts:
            indices = np.where(y_week1 == cls)[0]
            X_res.append(X[indices])
            y_res.append(y[indices])
            cases_res.append(cases[indices])

            needed = target_samples - len(indices)
            if needed > 0:
                duplicate_indices = np.random.choice(indices, size=needed, replace=True)
                X_res.append(X[duplicate_indices])
                y_res.append(y[duplicate_indices])
                cases_res.append(cases[duplicate_indices])

        return (np.concatenate(X_res),
                np.concatenate(y_res),
                np.concatenate(cases_res))
    def augment_critical_samples(X, y, cases, thresholds=[(15, 25, 10), (90, 110, 15), (295, 305, 10)]):
        X_aug = [X]
        y_aug = [y]

        for low, high, factor in thresholds:
            indices = np.where((cases >= low) & (cases <= high))[0]
            if len(indices) > 0:
                X_aug.append(np.repeat(X[indices], factor, axis=0))
                y_aug.append(np.repeat(y[indices], factor, axis=0))

        return np.concatenate(X_aug, axis=0), np.concatenate(y_aug, axis=0)

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=52)
    fold_results = []
    best_model_weights = None
    best_val_loss = float('inf')

    class_weight = torch.tensor([1.2, 2.0, 1.8, 1.6], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_final, y_train_final[:, 0])):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")

        X_train_fold, y_train_fold , cases_train_fold = X_train_final[train_idx], y_train_final[train_idx], cases_train[train_idx]

        X_train_fold, y_train_fold, cases_train_fold = oversample_stratified(X_train_fold, y_train_fold, cases_train_fold)
        X_train_fold, y_train_fold = augment_critical_samples(X_train_fold, y_train_fold, cases_train_fold)

        train_ds = to_tensor_dataset(X_train_fold, y_train_fold)
        val_ds = Subset(to_tensor_dataset(X_train_final, y_train_final), val_idx)

        train_loader = DataLoader(train_ds, batch_size=120, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=100, shuffle=False)

        fold_model = copy.deepcopy(model).to(device)
        optimizer = optim.Adam(fold_model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=60, cooldown=10, min_lr=1e-6, verbose=True)

        early_stopping = EarlyStopping(patience=early_stop_patience, verbose=False, delta=0.0001)

        for epoch in range(num_epochs):
            fold_model.train()
            train_loss = 0.0
            for Xb, y5b, y6b in train_loader:
                Xb, y5b, y6b = Xb.to(device), y5b.to(device), y6b.to(device)

                optimizer.zero_grad()
                pred5, pred6 = fold_model(Xb)

                loss5 = criterion(pred5, y5b)
                loss6 = criterion(pred6, y6b)
                loss = loss5 + loss6

                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
            fold_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for Xb, y5b, y6b in val_loader:
                    Xb, y5b, y6b = Xb.to(device), y5b.to(device), y6b.to(device)
                    pred5, pred6 = fold_model(Xb)
                    val_loss += criterion(pred5, y5b) + criterion(pred6, y6b)

            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            # if epoch > 500:
            early_stopping(avg_val_loss, fold_model)

            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{num_epochs} | LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss / len(train_loader):.4f} | "
                      f"Val Loss: {avg_val_loss:.4f}")

        if early_stopping.val_loss_min < best_val_loss:
            best_val_loss = early_stopping.val_loss_min
            best_model_weights = copy.deepcopy(early_stopping.best_model)
            print(f"New best model found at fold {fold + 1} with val loss: {best_val_loss:.4f}")

        fold_model.load_state_dict(early_stopping.best_model)
        val_metrics = evaluate_model(fold_model, val_loader, device)
        train_loader_eval = DataLoader(train_ds, batch_size=120, shuffle=False)
        train_metrics = evaluate_model(fold_model, train_loader_eval, device)
        fold_results.append({
            'best_val_loss': early_stopping.val_loss_min,
            'metrics': val_metrics,
            'epochs_trained': epoch + 1,
            'train_metrics': train_metrics,
            'epochs_trained': epoch + 1,
        })

    model.load_state_dict(best_model_weights)
    test_loader = DataLoader(to_tensor_dataset(X_test, y_test), batch_size=100, shuffle=False)
    test_metrics = evaluate_model(model, test_loader, device)

    return fold_results, model, test_metrics

# results, best_model, test_metrics = cross_validate(
#     model2, X_train_final, y_train_final, X_test, y_test,
#     n_splits=5, num_epochs=2000, lr=0.0001, early_stop_patience=200
# )

# model_path = f'./weight/warning.pth'
# model_weight = torch.load(model_path)
# model2.load_state_dict(model_weight)
# results, best_model, test_metrics = cross_validate2(
#     model2, X_train, y_train, X_test, y_test, cases_train, cases_test,
#     n_splits=5, num_epochs=1, lr=0.000000000001, early_stop_patience=200
# )
#
# torch.save(best_model.state_dict(), "best_model.pth")


# print("\n=== Cross-validation Results ===")
# for i, res in enumerate(results):
#     print(f"\n=== Fold {i + 1} Summary ===")
#     print(f"Epochs Trained     : {res['epochs_trained']}")
#     print(f"Best Val Loss      : {res['best_val_loss']:.4f}")
#
#     for week in ['week5', 'week6']:
#         train = res['train_metrics'][week]
#         val = res['metrics'][week]
#         print(f"\n{week.upper()} Evaluation:")
#         print(f"  [Train] Accuracy         : {train['accuracy']:.4f}")
#         print(f"          Macro F1         : {train['macro_f1']:.4f}")
#         print(f"          Weighted F1      : {train['weighted_f1']:.4f}")
#         print(f"  [Valid] Accuracy         : {val['accuracy']:.4f}")
#         print(f"          Macro F1         : {val['macro_f1']:.4f}")
#         print(f"          Weighted F1      : {val['weighted_f1']:.4f}")
#         print(f"  Validation Confusion Matrix:\n{val['matrix']}")
#
#
# print("\n=== Final Test Performance ===")
# print("Week 5 Results:")
# print(f"Accuracy: {test_metrics['week5']['accuracy']:.4f}")
# print("Classification Report:")
# print(test_metrics['week5']['report'])
# print("Confusion Matrix:")
# print(test_metrics['week5']['matrix'])
#
# print("\nWeek 6 Results:")
# print(f"Accuracy: {test_metrics['week6']['accuracy']:.4f}")
# print("Classification Report:")
# print(test_metrics['week6']['report'])
# print("Confusion Matrix:")
# print(test_metrics['week6']['matrix'])

import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [18, 14]
plt.rcParams['font.size'] = 10

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = [18, 12]
plt.rcParams['font.size'] = 10
from string import ascii_uppercase
def plot_all_metrics(results, test_metrics):
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    metrics = [('accuracy', 'Accuracy'), ('macro_f1', 'Macro F1'), ('weighted_f1', 'Weighted F1')]
    weeks = ['week5', 'week6']

    colors = {
        'train': '#1f77b4',
        'val': '#2ca02c',
        'test': '#d62728'
    }
    subplot_labels = list(ascii_uppercase[:6])  # A,B,C,D,E,F
    for row, week in enumerate(weeks):
        for col, (metric, label) in enumerate(metrics):
            ax = axes[row, col]
            for spine in ax.spines.values():
                spine.set_color('black')
                spine.set_linewidth(1.2)
            label_idx = row * 3 + col
            ax.text(0, 1.1, subplot_labels[label_idx], transform=ax.transAxes,
                   fontsize=14, fontweight='bold', va='top')
            n_folds = len(results)
            train_metrics = [res['train_metrics'][week][metric] for res in results]
            val_metrics = [res['metrics'][week][metric] for res in results]
            test_metric = test_metrics[week][metric]

            x_labels = []
            x_ticks = []
            bar_width = 0.35
            x_pos = 0
            title_map = {
                'week5': 'One-Week Warning',
                'week6': 'Two-Week Warning'
            }

            for fold in range(n_folds):
                x_labels.append(f'Fold {fold + 1}')
                x_ticks.append(x_pos + bar_width / 2)
                ax.bar(x_pos, train_metrics[fold], width=bar_width,
                       color=colors['train'], alpha=0.8, label='Train' if fold == 0 else '')
                ax.bar(x_pos + bar_width, val_metrics[fold], width=bar_width,
                       color=colors['val'], alpha=0.8, label='Validation' if fold == 0 else '')

                x_pos += bar_width * 2 + 0.3

            separator_pos = x_pos - 0.15
            x_labels.append('Test')
            x_ticks.append(x_pos + 0.5)

            ax.bar(x_ticks[-1], test_metric, width=bar_width * 1.5,
                   color=colors['test'], alpha=0.8, label='Test')

            ax.axvline(x=separator_pos, color='gray', linestyle='--', linewidth=1)

            # ax.set_title(f'{week.upper()} {label}', fontsize=12)
            ax.set_title(f'{title_map[week]} {label}', fontsize=12)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_ylim(0, 1.2)
            ax.legend(loc='upper right',ncol=3,
            )

    plt.tight_layout()
    plt.show()



import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 16

start_date = datetime(2016, 1, 1)

def week_to_date(week_num):
    first_day = start_date + timedelta(weeks=int(week_num) - 1)
    last_day = first_day + timedelta(days=6)
    return last_day


time5 = np.arange(5, 5 + len(preds5))
time6 = np.arange(6, 6 + len(preds6))
dates5 = [week_to_date(w) for w in time5]
dates6 = [week_to_date(w) for w in time6]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

# param_colors = {'a': '#d62728', 'b': '#ff7f0e', 'c': '#1f77b4', 'd': '#2ca02c', 'e': '#9467bd'}
labels = ["Controlled", "Critical", "Outbreak", "Severe Outbreak"]
cmap = {
    0: "#DFFFE0",
    1: "#ADD8E6",
    2: "#FFFACD",
    3: "#F08080",

}
AA = 0.6
risk_patches = [mpatches.Patch(color=cmap[i], alpha=AA, label=labels[i],
                              edgecolor='none') for i in range(4)]

def draw_colored_background_dates(ax, dates, pred_classes, cmap):
    start = dates[0] - timedelta(days=3.5)
    prev_class = pred_classes[0]
    for i in range(1, len(pred_classes)):
        if pred_classes[i] != prev_class:
            end = dates[i - 1] + timedelta(days=3.5)
            ax.axvspan(start, end, color=cmap[int(prev_class)], alpha=AA, edgecolor='none')  # 增加透明度
            start = dates[i] - timedelta(days=3.5)
            prev_class = pred_classes[i]
    end = dates[-1] + timedelta(days=3.5)
    ax.axvspan(start, end, color=cmap[int(prev_class)], alpha=AA, edgecolor='none')
draw_colored_background_dates(ax1, dates5, preds5, cmap)
draw_colored_background_dates(ax2, dates6, preds6, cmap)

ax1.plot(dates5, total_case_data[4:4 + len(preds5)], color= '#b71c1c'  , linewidth=2.0, alpha=0.8, label="Real Cases")
ax2.plot(dates6, total_case_data[5:5 + len(preds5)], color= '#b71c1c'  , linewidth=2.0, alpha=0.8)

ax1.text(-0.04, 1.12, 'a', transform=ax1.transAxes,
         fontweight='bold', va='top')
ax2.text(-0.04, 1.12, 'b', transform=ax2.transAxes,
         fontweight='bold', va='top')

for ax in (ax1, ax2):
    ax.axhline(y=20, color='grey', linestyle='--', linewidth=1.0)
    ax.axhline(y=100, color='grey', linestyle='--', linewidth=1.0)
    ax.axhline(y=300, color='grey', linestyle='--', linewidth=1.0)

legend_patches = [
    mpatches.Patch(color=cmap[0], alpha=AA, label="Controllable"),
    mpatches.Patch(color=cmap[1], alpha=AA, label="Critical"),
    mpatches.Patch(color=cmap[2], alpha=AA, label="Outbreak"),
    mpatches.Patch(color=cmap[3], alpha=AA, label="Severe Outbreak"),
]

ax1.legend(handles=[*legend_patches, ax1.lines[0]],
           loc="upper left", fontsize=12,framealpha=0.5)

ax1.set_title("One-week-ahead Warning")
ax1.set_ylabel("Cases")

ax2.set_title("Two-week-ahead Warning")
ax2.set_xlabel("Date")
ax2.set_ylabel("Cases")

for ax in (ax1, ax2):
    ax.grid(False)
    ax.set_facecolor('white')
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color('black')

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    ax.set_xlim(dates5[0] - timedelta(days=7), dates5[-1] + timedelta(days=7))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)

vline_date = datetime(2020, 1, 1)

for ax in (ax1, ax2):
    ax.axvline(vline_date, color="black", linestyle="--", linewidth=1.5)
    ax.text(vline_date + timedelta(days=20),
            ax.get_ylim()[1]*0.85,            #
            "Model Extension",
            color="black",
            va="top", ha="left",rotation=90)
# 调整布局
plt.tight_layout()
plt.show()


