import timm
from config import CFG

def create_model():
    model = timm.create_model(CFG.model_name, num_classes=len(CFG.target_cols))
    return model