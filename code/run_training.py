import os
import gc
import torch
import pandas as pd
from config import CFG
from dataset import TrainDataset, TestDataset
from model import create_model
from train import train_loop
from utils import get_transforms, get_score, get_result, init_logger, data_split

LOGGER = init_logger()

def main():
    if not os.path.exists(CFG.OUTPUT_DIR):
        os.makedirs(CFG.OUTPUT_DIR)
    folds = pd.read_csv(CFG.TRAIN_CSV_PATH)
    folds = data_split(folds)
    oof_df = pd.DataFrame()
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            _oof_df = train_loop(folds, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    oof_df.to_csv(os.path.join(CFG.OUTPUT_DIR, 'oof_df.csv'), index=False)

if __name__ == '__main__':
    main()