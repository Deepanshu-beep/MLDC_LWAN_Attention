import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
import torchmetrics as torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import argparse
from sklearn.model_selection import train_test_split

# %matplotlib inline  
# %config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)

MAX_TOKEN_COUNT = 512
N_EPOCHS = 10
BATCH_SIZE = 4

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['litcovid','ohsumed','WHO'])
parser.add_argument('--train_path', type=str)
parser.add_argument('--dev_path', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

class TopicAnnotationDataset(Dataset):

  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: AutoTokenizer,
    max_token_len: int = 512,
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index=int):
    
    # extract row using index
    data_row = self.data.iloc[index]

    # get data (text, labels)
    text = data_row['title_abs']
    labels = data_row[LABEL_COLUMNS]

    # apply tokenization
    inputs = self.tokenizer.encode_plus(
        text, 
        max_length=self.max_token_len,
        padding="max_length", 
        truncation=True, 
        return_tensors="pt", 
    )

    return dict(
        text=text,
        input_ids=inputs["input_ids"].flatten(),
        attention_mask=inputs["attention_mask"].flatten(),
        labels=torch.FloatTensor(labels) 
    )


class TopicAnnotationDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    
    # setup train data
    self.train_dataset =  TopicAnnotationDataset(
        data=self.train_df,
        tokenizer=self.tokenizer,
        max_token_len=self.max_token_len
    )

    # setup test data
    self.test_dataset = TopicAnnotationDataset(
        data=self.test_df,
        tokenizer=self.tokenizer,
        max_token_len=self.max_token_len
    )

  def train_dataloader(self):

    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=2
    )

  def val_dataloader(self):

    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=2
    )

  def test_dataloader(self):
    
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        num_workers=2
    )

# fully-functional networks module
class FFN(nn.Module):
  def __init__(self, in_feat, out_feat, dropout):
      super(FFN, self).__init__()
      self.in2hid = nn.Linear(in_feat, in_feat)
      self.hid2out = nn.Linear(in_feat, out_feat)

      self.activation = nn.ReLU()
      self.dropout = nn.Dropout(dropout)

  def forward(self, input):
      hid = self.activation(self.dropout(self.in2hid(input)))
      return self.hid2out(hid)

# self-attention module
class Attention(nn.Module):
  def __init__(self, dimensions, attention_type="general"):
    super(Attention, self).__init__()

    if attention_type not in ['dot', 'general']:
      raise ValueError('Invalid attention type selected.')
    
    self.attention_type = attention_type
    if self.attention_type == "general":
      self.linear_in = nn.Linear(dimensions, dimensions, bias=False)

    self.linear_out = nn.Linear(dimensions * 2, dimensions, bias=False)
    self.softmax = nn.Softmax(dim=-1)
    self.tanh = nn.Tanh()

  def forward(self, query, context):
    batch_size, output_len, dimensions = query.size()
    query_len = context.size(1)

    if self.attention_type == "general":
        query = query.reshape(batch_size * output_len, dimensions)
        query = self.linear_in(query)
        query = query.reshape(batch_size, output_len, dimensions)

    # (batch_size, output_len, dimensions) * (batch_size, query_len, dimensions) ->
    # (batch_size, output_len, query_len)
    attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

    # Compute weights across every context sequence
    attention_scores = attention_scores.view(batch_size * output_len, query_len)
    attention_weights = self.softmax(attention_scores)
    attention_weights = attention_weights.view(batch_size, output_len, query_len)

    # (batch_size, output_len, query_len) * (batch_size, query_len, dimensions) ->
    # (batch_size, output_len, dimensions)
    mix = torch.bmm(attention_weights, context)

    # concat -> (batch_size * output_len, 2*dimensions)
    combined = torch.cat((mix, query), dim=2)
    combined = combined.view(batch_size * output_len, 2 * dimensions)

    # Apply linear_out on every 2nd dimension of concat
    # output -> (batch_size, output_len, dimensions)
    output = self.linear_out(combined).view(batch_size, output_len, dimensions)
    output = self.tanh(output)

    return output, attention_weights

class TopicAnnotationTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    
    super().__init__()

    # specter embedding model
    self.specter = AutoModel.from_pretrained(model_name, return_dict=True) 

    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss(weight=class_weight)
    self.sequence_length = 512
    self.fc = nn.Linear(self.specter.config.hidden_size, n_classes)
    self.relu = nn.ReLU()
    self.softmax = nn.Softmax(dim=1)
    self.classifier = nn.Linear(self.sequence_length, 1)
    self.sigmoid = nn.Sigmoid()

    self.attention = Attention(self.specter.config.hidden_size)
    self.hidden2labels = nn.Linear(self.specter.config.hidden_size, n_classes)

  def forward(self, input_ids, attention_mask, labels=None):

    specter_output = self.specter(input_ids, attention_mask)
    # [batch_size x sequence_length x hidden_size]
    specter_output = specter_output.last_hidden_state
    
    specter_output, x = self.attention(specter_output, specter_output)
    specter_output, x = self.attention(specter_output, specter_output)

    # Transform data [batch_size x sequence_length x hidden_size] => [batch_size x sequence_length x n_classes]
    specter_output = self.hidden2labels(specter_output).transpose(1,2)
    # apply sigmoid function to the same
    output = self.sigmoid(self.classifier(specter_output).squeeze(-1))

    loss = 0
    if labels is not None:
      loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
    return loss

  def test_step(self, batch, batch_idx):
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]

    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=BATCH_SIZE)
    return loss

  def training_epoch_end(self, outputs):

    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(LABEL_COLUMNS):
      auroc = torchmetrics.AUROC(num_classes=len(LABEL_COLUMNS))
      class_roc_auc = auroc(predictions[:, i], labels[:, i])
    #   self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)

  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-6)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
        optimizer=optimizer,
        lr_scheduler=dict(
            scheduler=scheduler,
            interval='step'
            )
        )

if __name__=='__main__':
    if args.dataset=='litcovid':
        train_df = pd.read_csv(args.train_path)
        test_df = pd.read_csv(args.dev_path)
        LABEL_COLUMNS = ['Treatment', 'Mechanism', 'Prevention', 'Case Report', 'Diagnosis', 'Transmission', 'Epidemic Forecasting']
    else:
        train_df = pd.read_csv(args.train_path)
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=1)
        if args.dataset=='WHO':
            LABEL_COLUMNS = train_df.columns[4:].to_list()
        else:
            LABEL_COLUMNS = train_df.columns[2:].to_list()

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = TopicAnnotationDataset(train_df,tokenizer=tokenizer)
    data_module = TopicAnnotationDataModule(train_df,test_df,tokenizer,batch_size=BATCH_SIZE,max_token_len=MAX_TOKEN_COUNT)
    # calculate class-wise weights
    label_count = train_df[LABEL_COLUMNS].sum().to_dict()
    count = list(label_count.values())
    max_val = max(count)
    class_weight = [max_val/val for val in count]
    # transfer to gpu
    class_weight = torch.tensor(class_weight, device='cuda')
    steps_per_epoch=len(train_df) // BATCH_SIZE
    total_training_steps = steps_per_epoch * N_EPOCHS
    warmup_steps = total_training_steps // 5
    model = TopicAnnotationTagger(n_classes=len(LABEL_COLUMNS),n_warmup_steps=warmup_steps,n_training_steps=total_training_steps)
    # set-up checkpoints annd logs directory
    checkpoint_callback = ModelCheckpoint(
        dirpath="model_checkpoints",
        filename="model_checkpoints",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    logger = TensorBoardLogger("model_logs", name="topic-annotations")

    # early-stopping criterion
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    # setup trainer and add training arguments
    trainer = pl.Trainer(
        logger=logger,
        # logger = wandb_logger,
        checkpoint_callback=True,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=N_EPOCHS,
        gpus=1,
        progress_bar_refresh_rate=30,
    )
    trainer.fit(model, data_module)
    