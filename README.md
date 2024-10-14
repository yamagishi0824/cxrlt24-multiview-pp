# CXRLT24 Multiview PP

This repository contains our solution for the MICCAI 2024 CXR-LT (Chest X-Ray Long-Tailed) challenge, achieving 4th place in Subtask 2 and 5th in Subtask 1.

## Project Overview

We present an ensemble method for long-tailed chest X-ray (CXR) classification using ConvNeXt V2 and MaxViT models. Our approach combines state-of-the-art image classification techniques with asymmetric loss for handling class imbalance and view-based prediction aggregation to enhance overall performance.

## Repository Structure

The `code` directory contains the following Python scripts:

- `config.py`: Configuration settings for the project
- `dataset.py`: Dataset handling and preprocessing
- `inference.py`: Model inference logic
- `model.py`: Model architecture definitions
- `postprocess.py`: Post-processing techniques including view-based aggregation
- `run_inference.py`: Script to run inference on test data
- `run_training.py`: Script to initiate the training process
- `train.py`: Training loop and logic
- `utils.py`: Utility functions used across the project

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
coming soon...
```
