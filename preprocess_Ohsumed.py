import os
import argparse
from transformers import AutoTokenizer
from sklearn import preprocessing
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./Data/ohsumed-all')
parser.add_argument('--output', type=str, default='./Data/ohsumed_preprocessed.csv')

args = parser.parse_args()

def process_labels(text):
  labels = [label for label in LABEL_COLUMNS if text.find(label) != -1]
  labels = ';'.join(labels)
  return labels

if __name__=='__main__':
    model_name = "allenai/specter"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    labels = os.listdir(args.path)
    data = {}

    for i in labels:
        for j in os.listdir(f'{args.path}/{i}'):
            if j not in data.keys():
                with open(f'{args.path}/{i}/{j}') as f:
                    info = f.readlines()
                info[0] = info[0]+tokenizer.sep_token
                info = ''.join(info)
                info = info.replace('\n','')
                label = set()
                label.add(i)
                info_labels = [info,label]
                data[j] = info_labels
            else:
                info,label = data[j]
                label.add(i)
                data[j] = [info, label]

    title_abs = [x[0] for x in data.values()]
    label = [';'.join(x[1]) for x in data.values()]
    df = pd.DataFrame({'pmid':data.keys(), 'title_abs':title_abs, 'labels':label})
    LABEL_COLUMNS = set()
    for i in df.labels:
        for j in i.split(';'):
            LABEL_COLUMNS.add(j)
    LABEL_COLUMNS = sorted(LABEL_COLUMNS)

    labels = df.labels.apply(process_labels)
    df["labels"] = labels
    Labels_df = pd.Series([labels_temp.split(';') for labels_temp in list(df.labels)],index=df['pmid']).to_dict()

    pmid_df = df.pmid
    unique_pmid_df = df.pmid.unique()

    Le = preprocessing.LabelEncoder()
    Le.fit(list(LABEL_COLUMNS))
    list(Le.classes_)

    # Encoded version of Labels 
    Encoded_labels_df = dict((key, Le.transform(values).tolist()) for key, values in Labels_df.items())

    y = [tuple(values) for key, values in Encoded_labels_df.items()]

    # Onehot encoding
    from sklearn.preprocessing import MultiLabelBinarizer

    onehot = MultiLabelBinarizer()

    Encodings_tr = onehot.fit_transform(y)

    class_mappings_tr = dict(zip(Le.classes_, onehot.classes_))

    for key in class_mappings_tr.keys():

    # Assigning new columns as labels and their respective boolean values
    # litcovid_dataset.insert(, key, Encodings[:, class_mappings[key]])
        df[key] = Encodings_tr[:, class_mappings_tr[key]]

    df.drop(columns=['labels'], inplace=True)
    df.to_csv(args.output,index=False)