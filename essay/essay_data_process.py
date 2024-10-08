import os
import pandas as pd
import uuid
from bs4 import BeautifulSoup
import re
from sklearn.model_selection import train_test_split

# Define a special character set and use the set to improve the search speed.
SPECIAL_CHAR_SET = set('{£\x97¹å\\\x85<\x99é]+Ö\xa0>|\\x80~©/\x93$Ó²^;`á*(¶®[\x94\x91#-ó)=}')

def clean_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def squash_whitespace(text):
    return ' '.join(text.split())

def drop_square_brackets_content(text):
    return re.sub(r'\[.*?\]', '', text)

def purge_special_characters(text):
    return ''.join(char for char in text if char not in SPECIAL_CHAR_SET)

def discard_eot_token(text):
    return text.replace("<|end_of_text|>", "")

def preprocess_essay(text):
    text = discard_eot_token(text)
    text = text.encode("ascii", errors="ignore").decode()
    text = text.strip()
    text = text.strip('"')
    text = purge_special_characters(text)
    if text and text[-1] != ".":
        text = '.'.join(text.split('.')[:-1]) + '.'
    text = clean_html(text)
    text = drop_square_brackets_content(text)
    return text

def generate_random_string(text):
    return 'ESSAY_' + uuid.uuid4().hex[:8]

def prepare_ESSAY_data(dir_data: str) -> pd.DataFrame:
    dir_daigt_v2 = os.path.join(dir_data, 'DAIGT_V2/train_v2_drcat_02.csv')
    df_daigt_v2 = pd.read_csv(dir_daigt_v2).rename(columns={'label': 'generated'})
    df_daigt_v2.drop(columns=["RDizzl3_seven"], inplace=True)
    df_daigt_v2["id"] = df_daigt_v2["text"].apply(generate_random_string)
    df_daigt_v2 = df_daigt_v2[["id", "text", "source", "generated"]]
    df_daigt_v2.drop_duplicates(subset=["text"], inplace=True)
    return df_daigt_v2

def shuffle_and_split_data(df: pd.DataFrame, test_size: float = 0.2):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)
    return df_train, df_test

def prepare_data(dir_data: str, dir_output: str, test_size: float = 0.2):
    df_daigt_v2 = prepare_ESSAY_data(dir_data)
    print("*"*50, "Dataset shape", "="*50)
    print("Total DAIGT_V2 essays: ", df_daigt_v2.shape[0])

    print("Preprocessing data...")
    df_daigt_v2['text'] = df_daigt_v2['text'].apply(preprocess_essay)

    print("Checking for duplicates...")
    print("Duplicates in dataset: ", df_daigt_v2.duplicated().sum())
    print("Duplicates by ID: ", df_daigt_v2.duplicated(subset=["id"]).sum())

    print("Shuffling and splitting data into train and test sets...")
    df_train, df_test = shuffle_and_split_data(df_daigt_v2, test_size=test_size)

    print("Saving files...")
    df_train.to_csv(os.path.join(dir_output, 'train_essays.csv'), index=False)
    df_test.to_csv(os.path.join(dir_output, 'test_essays.csv'), index=False)
    print("*"*110)

if __name__ == "__main__":
    data_dir = '/Users/adong/Desktop/course/6140/CS6140-MUGC-project/essay/data' 
    output_dir = '/Users/adong/Desktop/course/6140/CS6140-MUGC-project/essay/data/processed' 
    os.makedirs(output_dir, exist_ok=True)
    prepare_data(data_dir, output_dir)
