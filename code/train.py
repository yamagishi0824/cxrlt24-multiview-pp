import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from config import CFG
from dataset import TrainDataset
from utils import get_transforms, AverageMeter, timeSince
from loss import get_loss

def train_loop(folds, fold):
    LOGGER.info(f"========== fold: {fold} training ==========")

    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='valid'))

    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    model = create_model()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = get_scheduler(optimizer)

    class_instance_nums = calculate_class_instance_nums(train_folds, CFG.target_cols)
    total_instance_num = len(train_folds)
    criterion = get_loss('asl', class_instance_nums, total_instance_num)

    best_score = 0.

    for epoch in range(CFG.epochs):
        start_time = time.time()

        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        valid_labels = valid_folds[CFG.target_cols].values
        score, scores = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  AUC: {np.round(scores, decimals=4)}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 'preds': preds},
                       CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')

    check_point = torch.load(CFG.OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    valid_folds[[f'pred_{c}' for c in CFG.target_cols]] = check_point['preds']
    valid_folds['preds'] = check_point['preds'].argmax(1)

    return valid_folds

def train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(images)
            loss = criterion(y_preds, labels.float())
        
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
        
        losses.update(loss.item(), batch_size)
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          lr=optimizer.param_groups[0]['lr']))
        
        if CFG.batch_scheduler:
            scheduler.step()
    
    return losses.avg

def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    start = end = time.time()
    
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        
        with torch.no_grad():
            y_preds = model(images)
        
        loss = criterion(y_preds, labels.float())
        losses.update(loss.item(), batch_size)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
        
        end = time.time()
        
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          remain=timeSince(start, float(step+1)/len(valid_loader)),
                          loss=losses))
    
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def get_scheduler(optimizer):
    if CFG.scheduler == 'get_cosine_schedule_with_warmup':
        steps_per_epoch = len(train_dataset) // CFG.batch_size
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=steps_per_epoch * CFG.warmup_epochs,
            num_training_steps=steps_per_epoch * CFG.epochs
        )
    # Add other scheduler options here if needed
    return scheduler