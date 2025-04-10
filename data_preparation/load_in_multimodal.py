from sys import argv
import json
import pandas as pd

if len(argv) != 3:
    print("Expected load_in_multimodal.py [source] [dest]")
else:
    # https://github.com/headacheboy/data-of-multimodal-sarcasm-detection/blob/master/codes/loadData.py#L68-L89
    obvious_keys = ["sarcasm",
                    "sarcastic",
                    "reposting",
                    "<url>",
                    "joke",
                    "humour",
                    "humor",
                    "jokes",
                    "irony",
                    "ironic",
                    "exgag"]

    texts = []
    labels = []

    with open(argv[1], "r", encoding="utf-8-sig") as f:
        while True:
            line = f.readline()
            if len(line) == 0:
                break

            line_eval = eval(line) # sigh, but this is the way it is intended

            texts.append(line_eval[1])
            labels.append(line_eval[-1])

    as_df = pd.DataFrame({"tweet": texts, "sarcastic": labels})
    as_df.to_csv(argv[2])

