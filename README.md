# Unsupervised Feature Elimination via Generative Adversarial Networks: Application to Hair Removal in Melanoma Classification
This repository contains source codes to train an unsupervised feature elimination algorithm using GAN and demonstrate it in hair removal task for melanoma classification.

## Prerequisites
- Melanoma Hair Removal Dataset ([Link to the dataset website](https://www.kaggle.com/c/siim-isic-melanoma-classification/data))

## Installation
- Download the repository
```
git clone https://github.com/joyfuldahye/unsupervised-feature-elimination-via-gan.git
```
- Install required libraries
```
pip install -r requirements.txt
```

## Training by running a shellscript file
1. Set dataset(`--dir_train_data_image`), csv(`--dir_data_csv_hair`, `--dir_data_csv_hair`) and result path(`--dir_output`) in `src/scripts/train_feature_elimination.sh` according to your environment
2. Modify `src/scripts/train_feature_elimination.sh` as you want to try
3. Run the following commands in your terminal
```
cd src/scripts
chmod +x train_feature_elimination.sh
./train_feature_elimination.sh
```

## Monitoring training and output images
Current training codes save learning curves and output images in `result`. Please note that you are free to change the path to save result outputs and methods to monitor the results.
- `result/output/'your-experiment-name(e.g., 20200724_l1_lamdis10_lamgp10_lrd00001_lrg00001_bs4_ndisc5_nep10000_ex1_monitor_train)'`: contains training curves and output images.
    * The training curves are simultaneously updated as the training progresses with the interval(`--sample_interval`) you've set in `src/scripts/train_feature_elimination.sh`.
    * `result/output/'your-experiment-name/learn-curve-distance`: shows a training curve of distance term in the generator loss
    * `result/output/'your-experiment-name/learn-curve-loss-d`: shows a training curve of discriminator loss
    * `result/output/'your-experiment-name/learn-curve-loss-g`: shows a training curve of generator loss
- `result/train_info/'your-experiment-name(e.g., 20200724_l1_lamdis10_lamgp10_lrd00001_lrg00001_bs4_ndisc5_nep10000_ex1_monitor_train)'`: contains the configuration and model architecture details of the experiment 'your-experiment-name'.