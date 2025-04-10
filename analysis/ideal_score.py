from dataloading import test_main
from sys import argv
from os.path import basename
import pandas as pd


all_models = {}
for i in range(1, len(argv)):
    df = pd.read_csv(argv[i])
    df.reindex()
    all_models[basename(argv[i])] = df

for i in range(len(test_main)):
    num_true = 0
    num_false = 0
    correct = ""

    for model_name, preds in all_models.items():
        if preds["correct"][i] == False:
            num_true += 1
        if preds["correct"][i] == True:
            num_false += 1
            correct = model_name

    if num_true == len(all_models):
        print(f"All true {test_main["text"][i]}")
    if num_false == len(all_models):
        print(f"All false {test_main["text"][i]}")
    if num_true == 1 and num_false == len(all_models)-1:
        print(f"Only model {correct} for {test_main["text"][i]}")
