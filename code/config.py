class CFG:
    debug = False
    apex = True
    print_freq = 100
    num_workers = 8
    model_name = "maxvit_tiny_tf_512.in1k" #'convnextv2_tiny.fcmae_ft_in22k_in1k_384' #'tf_efficientnetv2_s.in21k_ft_in1k'
    size = 512
    scheduler = "get_cosine_schedule_with_warmup"
    batch_scheduler = True if scheduler == "get_cosine_schedule_with_warmup" else False
    epochs = 20
    warmup_epochs = 0
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 16
    weight_decay = 1e-2
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_cols = [
        'Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta',
        'Cardiomegaly', 'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema',
        'Enlarged Cardiomediastinum', 'Fibrosis', 'Fissure', 'Fracture', 'Granuloma',
        'Hernia', 'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis', 'Lobar Atelectasis',
        'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule', 'Normal', 'Pleural Effusion', 'Pleural Other',
        'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax',
        'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture', 'Round(ed) Atelectasis',
        'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'Tuberculosis'
    ]
    sub_cols_task1 = ['dicom_id', 'Adenopathy', 'Atelectasis', 'Azygos Lobe',
       'Calcification of the Aorta', 'Cardiomegaly', 'Clavicle Fracture',
       'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum',
       'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 'Hernia',
       'Hydropneumothorax', 'Infarction', 'Infiltration', 'Kyphosis',
       'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule',
       'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening',
       'Pneumomediastinum', 'Pneumonia', 'Pneumoperitoneum', 'Pneumothorax',
       'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture',
       'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices',
       'Tortuous Aorta', 'Tuberculosis']
    sub_cols_task2 = ['dicom_id', 'Atelectasis', 'Calcification of the Aorta', 'Cardiomegaly',
       'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum',
       'Fibrosis', 'Fracture', 'Hernia', 'Infiltration', 'Lung Lesion',
       'Lung Opacity', 'Mass', 'Normal', 'Nodule', 'Pleural Effusion',
       'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia',
       'Pneumoperitoneum', 'Pneumothorax', 'Subcutaneous Emphysema',
       'Support Devices', 'Tortuous Aorta']
    target_size = len(target_cols)
    n_fold = 5
    trn_fold = [0,1,2,3,4]
    mixup = False
    
    EXP = "011"
    OUTPUT_DIR = f'/content/drive/MyDrive/cxr_lt_2024/exp/{EXP}/'
    TRAIN_PATH = '/content/mimic-cxr-jpg-512/physionet.org/files/mimic-cxr-jpg/2.0.0'
    TEST_PATH = '/content/mimic-cxr-jpg-512/physionet.org/files/mimic-cxr-jpg/2.0.0'
    TRAIN_CSV_PATH = '/content/task1_development_starting_kit/train_labeled.csv'
    TEST_CSV_PATH_TASK1 = '/content/task1_development_starting_kit/development.csv'
    TEST_CSV_PATH_TASK2 = '/content/task2_development_starting_kit/development.csv'
    SUB_CSV_PATH_TASK1 = '/content/drive/MyDrive/cxr_lt_2024/data/development_task1_sample_submission.csv'
    SUB_CSV_PATH_TASK2 = '/content/drive/MyDrive/cxr_lt_2024/data/development_task2_sample_submission.csv'