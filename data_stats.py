from nltk.tokenize import word_tokenize
from dataloading import get_full_df, get_main_df, get_non_main_df

train_main, test = get_main_df()
train_full = get_full_df(2)[0]
train_non_main = get_non_main_df(2)[0]

test = test.rename(columns={'text': 'tweet'})

for train_set in [train_main, train_full, train_non_main, test]:
    ratio = 0
    length = 0
    unique = set()
    for i, row in train_set.iterrows():
        # get number of tokens
        try:
            tok = word_tokenize(row['tweet'], 'english')
            length += len(tok)
            ratio += int(row['sarcastic'])
            unique.update(set(tok))
        except: 
            print(row['tweet'])
    length /= train_set.shape[0]
    ratio /= train_set.shape[0]
    print(f"average length in tokens: {length}")
    print(f"unique words: {len(unique)}")
    print(f"class ratio sarcastic: {ratio}")
    print(f'number of instances: {train_set.shape[0]}')
