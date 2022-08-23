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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, choices=['litcovid','ohsumed','WHO'])
parser.add_argument('--train_path', type=str)
parser.add_argument('--test_path', type=str)
parser.add_argument('--model_checkpoint', type=str)
parser.add_argument('--model', type=str)
args = parser.parse_args()

BATCH_SIZE = 4
MAX_TOKEN_COUNT = 512

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

if __name__=='__main__':
    train_df = pd.read_csv(args.train_path)
    test_df = pd.read_csv(args.test_path)
    if args.dataset=='litcovid':
        LABEL_COLUMNS = ['Treatment', 'Mechanism', 'Prevention', 'Case Report', 'Diagnosis', 'Transmission', 'Epidemic Forecasting']
    elif args.dataset=='WHO':
        LABEL_COLUMNS = train_df.columns[4:].to_list()
    else:
        LABEL_COLUMNS = train_df.columns[2:].to_list()

    model_name = args.model
    label_count = train_df[LABEL_COLUMNS].sum().to_dict()
    count = list(label_count.values())
    max_val = max(count)
    class_weight = [max_val/val for val in count]
    # transfer to gpu
    class_weight = torch.tensor(class_weight, device='cuda')
    trained_model = TopicAnnotationTagger.load_from_checkpoint(args.model_checkpoint, n_classes=len(LABEL_COLUMNS))
    trained_model.freeze()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = trained_model.to(device)

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # create test data
    test_dataset = TopicAnnotationDataset(
        test_df,
        tokenizer,
        max_token_len=MAX_TOKEN_COUNT
    )
    predictions = []
    labels = []

    # generate predictions 
    for n,item in enumerate(tqdm(test_dataset)):

        _, prediction = trained_model(
            item["input_ids"].unsqueeze(dim=0).to(device),
            item["attention_mask"].unsqueeze(dim=0).to(device),
        )
        predictions.append(prediction.flatten())
        labels.append(item["labels"].int())
    predictions = torch.stack(predictions).detach().cpu()
    labels = torch.stack(labels).detach().cpu()
    y_pred = predictions.numpy()
    y_true = test_df[LABEL_COLUMNS].to_numpy()

    upper, lower = 1, 0

    y_pred = np.where(y_pred > 0.5, upper, lower)

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            target_names=LABEL_COLUMNS,
            zero_division=0
        )
    )