from sys import argv
import json
import pandas as pd
import json

if len(argv) != 3:
    print("Expected load_in_mustard.py [source] [dest]")
else:
    with open(argv[1]) as f:
        json_data = json.load(f)

        utterances = [utterance["utterance"] for utterance in json_data.values()]
        labels = [utterance["sarcasm"] for utterance in json_data.values()]

        as_df = pd.DataFrame({"tweet": utterances, "sarcastic": labels})
        as_df.to_csv(argv[2])
