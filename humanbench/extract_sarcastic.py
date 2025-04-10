# Take test set, remove labels and shuffle them
# for human benchmark annotation

import csv
import numpy as np, pandas as pd

with open('task_A_En_test.csv', 'r', newline='', encoding='utf-8-sig') as test_csv:
    testReader = csv.reader(test_csv)   
    
    test = [row for i, row in enumerate(testReader)]

    test = pd.DataFrame(test[1:], columns = test[0])
    print(pd.DataFrame.head(test))

    test["sarcastic"] = "?"
    test.sample(frac = 1)

    test.to_csv("task_A_test_shuffle_to_annotate.csv")
