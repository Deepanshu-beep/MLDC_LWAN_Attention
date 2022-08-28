# <div align="center"> MLDC using LWAN and Dual-attention module </div>
This repository contains the code for PACLIC 2022 paper: Bio-Medical Multi-label Scientific Literature Classification using LWAN and Dual-attention module.

# Preprocessing Databases

Install the requirements using the following command.
```
pip install -r requirements.txt
```

## 1. LitCovid Database
Download LitCovid's [train](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Train.csv), [dev](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Dev.csv) and [test](https://ftp.ncbi.nlm.nih.gov/pub/lu/LitCovid/biocreative/BC7-LitCovid-Test-GS.csv) set.
Preprocess the database by running:
```
python preprocess_litcovid.py \
--path path to csv file to be modified \
--output path where preprocessed csv file needs to be stored
```

## 2. Ohsumed Database
Download the [tar](http://disi.unitn.it/moschitti/corpora/ohsumed-all-docs.tar.gz) file for Ohsumed database and untar it.
Preprocess the database by running:
```
python preprocess_Ohsumed.py \
--path directory path to ohsumed database \
--output path where preprocessed csv file needs to be stored
```

## 3. WHO Database
We already scraped the WHO dataset and you can download the preprocessed data [here](https://drive.google.com/file/d/1AyB_de5N8fzopkGRcB56wcUyvTczwJyB/view?usp=sharing).

# Training
Run the following command for training the LitCovid database or any similar database with seperate csv for training and validation:
```
python train.py \
--dataset litcovid \
--train_path path to preprocessed csv file for training \
--dev_path path to preprocessed csv file for validation \
--model 'bioformers/bioformer-cased-v1.0'
```

Run the following command for validating the Ohsumed, WHO database or any similar database with single csv which you wish to split for training and validation:
```
python train.py \
--dataset ohsumed \
--train_path path to preprocessed csv file for training \
--model 'bioformers/bioformer-cased-v1.0'
```

# Inference
Run the following command if you wish to evaluate the performance of above trained model:
```
python inference.py \
--dataset ohsumed \
--train_path path to preprocessed csv file for training \
--test_path path to preprocessed csv file for evaluating the model \
--model_checkpoint path to saved checkpoint of model after training \
--model 'allenai/specter'
```
