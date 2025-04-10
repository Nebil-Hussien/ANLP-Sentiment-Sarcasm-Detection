import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
sns.set_theme() # use seaborn plotting style
    
# load all datasets indiviudally
train_main = pd.read_csv('datasets/train.En.csv', index_col=0)
train_2018_emojihash = pd.read_csv('datasets/task_A_2018_train_emojihash.csv', sep=';', index_col=0)
train_2018_emoji = pd.read_csv('datasets/task_A_2018_train_emoji.csv', sep=';', index_col=0)
train_2018 = pd.read_csv('datasets/task_A_2018_train.csv', sep=';', index_col=0)
train_mustard = pd.read_csv('datasets/mustard.csv', index_col=0)
train_multimodal1 = pd.read_csv('datasets/multimodal_train.csv', index_col=0)
train_multimodal2 = pd.read_csv('datasets/multimodal_valid.csv', index_col=0)
train_multimodal3 = pd.read_csv('datasets/multimodal_test.csv', index_col=0)
test_main = pd.read_csv('datasets/task_A_En_test.csv')

# adapt column names and remove unnecessary columns
train_main.drop(columns=['rephrase', 'sarcasm', 'irony', 'satire', 'understatement',
                         'overstatement', 'rhetorical_question'], inplace=True)
for df in [train_2018_emojihash, train_2018_emoji, train_2018]:
    df.rename(columns={'Tweet text': 'tweet', 'Label': 'sarcastic'}, errors='raise', inplace=True)
 
# define datasets based on two criteria: 1) inclusion of main dataset 2) inclusion of emojis and hashtags
def get_main_df() -> tuple[pd.DataFrame]:
    '''
    gets only the main dataset
    
    returns
        the main train and test datasets
    '''
    
    return train_main, test_main

def get_full_df(emojihash: int) -> tuple[pd.DataFrame]:
    '''
    gets all datasets with a choice of data processing stage
        emojihash 0 equates to no preprocessing
        emojihash 1 equates to paraphrased emojis
        emojihash 2 equates to paraphrased emojis and removed hashes
        
    returns
        the entirety of investigated datasets as train and the main test dataset
    '''
    
    if emojihash == 0:
        train = pd.concat([train_main, train_2018, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    elif emojihash == 1:
        train = pd.concat([train_main, train_2018_emoji, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    elif emojihash == 2:
        train = pd.concat([train_main, train_2018_emoji, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    else:
        raise ValueError("emojihash must be between 0 and 2 (inclusive)")
    
    return train, test_main

def get_non_main_df(emojihash: int) -> tuple[pd.DataFrame]:
    '''
    gets all datasets with a choice of data processing stage
        emojihash 0 equates to no preprocessing
        emojihash 1 equates to paraphrased emojis
        emojihash 2 equates to paraphrased emojis and removed hashes
        
    returns
        the entirety of investigated datasets as train and the main test dataset
    '''
    
    if emojihash == 0:
        train = pd.concat([train_2018, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    elif emojihash == 1:
        train = pd.concat([train_2018_emoji, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    elif emojihash == 2:
        train = pd.concat([train_2018_emoji, train_mustard, train_multimodal1,
                             train_multimodal2, train_multimodal3], axis=0)
    else:
        raise ValueError("emojihash must be between 0 and 2 (inclusive)")
    
    return train, test_main
