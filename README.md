# Fingerprint Authentication System

Open-set fingerprint authentication system using a hybrid of 
MobileNetV2 deep embeddings and classical descriptors (HOG, LBP, 
minutiae, ridge frequency) with PCA+LDA dimensionality reduction 
and EER-based threshold rejection.

## Overview
This project implements an open-set fingerprint identification and 
authentication system. The pipeline combines MobileNetV2 deep features 
with five classical descriptors into a 10,592-dimensional hybrid 
descriptor. PCA and LDA reduce this to a 487-dimensional discriminative 
space. Cosine similarity matching against a gallery of 488 enrolled 
identities is used for identification. Unknown persons are rejected 
using an EER-tuned threshold of 0.310.

## Results
- Rank-1 Accuracy: 48.2%
- Rank-10 Accuracy: 81.8%
- EER: 42.8%
- AUC: 0.625
- Threshold: 0.310

## Pipeline
1. Preprocessing — CLAHE normalization, segmentation, enhancement, 
   binarization, skeletonization
2. Feature Extraction — MobileNetV2 + HOG + LBP + Minutiae + 
   Ridge Frequency (10,592 dims)
3. Dimensionality Reduction — PCA + LDA (487 dims)
4. Matching — Cosine similarity against gallery templates
5. Open-Set Rejection — EER threshold at 0.310

## Dataset
Dataset not included. Place fingerprint images in the `/data` folder 
following the format `YYY_R0_KKK.bmp`.

## Author
Reddhi Das — NC State University, Spring 2026
