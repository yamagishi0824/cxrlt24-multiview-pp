import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import CFG
from dataset import TestDataset
from model import create_model
from utils import get_transforms
from config import CFG

def load_model(model_path):
    model = create_model()
    model.load_state_dict(torch.load(model_path)['model'])
    model.eval()
    return model

def inference(models, test_loader, device):
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc="Inference Progress"):
            images = images.to(device)
            model_outputs = []
            for model in models:
                outputs = model(images) / 2
                outputs += model(images.flip(-1)) / 2
                model_outputs.append(outputs.sigmoid().cpu().numpy())
            # Average predictions from all models
            model_outputs = np.mean(model_outputs, axis=0)
            predictions.append(model_outputs)
    return np.concatenate(predictions)

def run_inference(test_df, model_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = TestDataset(test_df, transform=get_transforms(data='valid'))
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False,
                             num_workers=CFG.num_workers, pin_memory=True)

    models = []
    for model_path in model_paths:
        model = load_model(model_path)
        model.to(device)
        model.eval() 
        models.append(model)

    final_predictions = inference(models, test_loader, device)
    return final_predictions

def save_predictions(test_df, predictions, output_path):
    submission = pd.DataFrame({'image_id': test_df['image_id']})
    for i, col in enumerate(CFG.target_cols):
        submission[col] = predictions[:, i]
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")