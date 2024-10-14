import time
import logging
import random
import os
import numpy as np
import torch
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from config import CFG
import albumentations as A
from albumentations.pytorch import ToTensorV2

def data_split(folds):
    Fold = GroupKFold(n_splits=CFG.n_fold)
    group = folds['subject_id']
    for n, (train_index, val_index) in enumerate(Fold.split(folds, folds["Atelectasis"], group)):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    return folds

def get_score(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[1]):
        score = label_ranking_average_precision_score([y_true[:,i]], [y_pred[:,i]])
        scores.append(score)
    avg_score = np.mean(scores)
    return avg_score, scores

def get_result(result_df):
    preds = result_df[[f'pred_{c}' for c in CFG.target_cols]].values
    labels = result_df[CFG.target_cols].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores.round(4)}')

def init_logger(log_file='train.log'):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler1 = logging.StreamHandler()
    handler1.setFormatter(logging.Formatter("%(message)s"))
    handler2 = logging.FileHandler(filename=log_file)
    handler2.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def asMinutes(s):
    m = int(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
            A.ShiftScaleRotate(p=0.5, rotate_limit=15),
            A.HorizontalFlip(p=0.5),
            A.Cutout(max_h_size=int(CFG.size * 0.05), max_w_size=int(CFG.size * 0.05), num_holes=5, p=0.1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

def mixup_data(x, y, alpha=.1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred.view(-1), y_a.view(-1).float()) + (1 - lam) * criterion(pred.view(-1), y_b.view(-1).float())