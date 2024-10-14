import pandas as pd
from config import CFG
from inference import run_inference, save_predictions
from postprocess import get_frontal, position_wise_mean

def main():
    # Load test data
    test_df = pd.read_csv(f'{CFG.TEST_CSV_PATH_TASK1}')
    sub_df = pd.read_csv(f'{CFG.SUB_CSV_PATH_TASK1}')

    # Model paths
    model_paths = [f'{CFG.OUTPUT_DIR}/{CFG.model_name}_fold{fold}_best.pth' for fold in CFG.trn_fold]

    # Run inference
    predictions = run_inference(test_df, model_paths)

    # Save predictions
    sub_df[CFG.target_cols] = predictions
    sub_df.to_csv(f'{CFG.OUTPUT_DIR}/predictions_task1.csv', index=False)
    sub_df[CFG.sub_cols_task2].to_csv(f'{CFG.OUTPUT_DIR}/predictions_task2.csv', index=False)

    # Postprocess predictions
    test_df = get_frontal(test_df)
    sub_df = test_df.merge(sub_df, on='dicom_id', suffixes=('', '_grouped'))
    sub_df  = position_wise_mean(sub_df)
    sub_df[CFG.sub_cols_task1].to_csv(f'{CFG.OUTPUT_DIR}/predictions_task1_pp.csv', index=False)
    sub_df[CFG.sub_cols_task2].to_csv(f'{CFG.OUTPUT_DIR}/predictions_task2_pp.csv', index=False)

if __name__ == '__main__':
    main()