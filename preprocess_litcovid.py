import pandas as pd
import argparse
from transformers import AutoTokenizer
from sklearn import preprocessing

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./Data/BC7-LitCovid-Train.csv')
parser.add_argument('--output', type=str, default='./Data/LitCovid_preprocessed_train.csv')

args = parser.parse_args()


def process_labels(text):
  labels = [label for label in LABEL_COLUMNS if text.find(label) != -1]
  labels = ';'.join(labels)
  return labels

if __name__=='__main__':
    df = pd.read_csv(args.path)
    df['abstract'] = df['abstract'].fillna(' ')
    model_name = "allenai/specter"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    title_abs_df = [row["title"] + tokenizer.sep_token + (row.get("abstract") or '') for _, row in df.iterrows()]
    df["title_abs"] = title_abs_df

    LABEL_COLUMNS = set()
    for i in df.label:
        for j in i.split(';'):
            LABEL_COLUMNS.add(j)
    LABEL_COLUMNS = sorted(LABEL_COLUMNS)

    labels = df.label.apply(process_labels)
    df["labels"] = labels
    Labels_df = pd.Series([labels_temp.split(';') for labels_temp in list(df.label)],index=df['pmid']).to_dict()

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

    df.drop(columns=['label'], inplace=True)
    df.to_csv(args.output,index=False)
