import pandas as pd
import numpy as np
import re
import string
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        pass
        
    def preprocess_text(self, text):
        """
        Comprehensive text preprocessing function
        """
        # Convert to string if not already
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep the text
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove RT (retweet) symbol
        text = re.sub(r'^RT[\s]+', '', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading and trailing whitespaces
        text = text.strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)
        
        return text

    def is_english(self, text):
        """
        Check if the text is primarily English
        """
        try:
            text.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        return True

    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess the dataset
        """
        # Read the dataset
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        # Make a copy of original data
        df_processed = df.copy()
        
        # Preprocess each tweet
        print("Preprocessing tweets...")
        df_processed['processed_text'] = df_processed['text'].apply(self.preprocess_text)
        
        # Remove empty tweets after preprocessing
        print("Removing empty tweets...")
        df_processed = df_processed[df_processed['processed_text'].str.len() > 0]
        
        # Remove duplicates
        print("Removing duplicates...")
        df_processed = df_processed.drop_duplicates(subset=['processed_text'])
        
        # Basic statistics
        print("\nDataset Statistics:")
        print(f"Original number of tweets: {len(df)}")
        print(f"Number of tweets after preprocessing: {len(df_processed)}")
        print(f"Number of sarcastic tweets: {len(df_processed[df_processed['sarcastic'] == 1])}")
        print(f"Number of non-sarcastic tweets: {len(df_processed[df_processed['sarcastic'] == 0])}")
        
        return df_processed

    def quality_checks(self, df):
        """
        Perform quality checks on the preprocessed data
        """
        issues = []
        
        # Check for empty strings
        empty_count = df['processed_text'].str.len().eq(0).sum()
        if empty_count > 0:
            issues.append(f"Found {empty_count} empty strings")
        
        # Check for very short texts
        short_count = df['processed_text'].str.split().str.len().lt(3).sum()
        if short_count > 0:
            issues.append(f"Found {short_count} very short texts")
        
        # Check class balance
        class_distribution = df['sarcastic'].value_counts(normalize=True)
        if abs(class_distribution[0] - class_distribution[1]) > 0.3:
            issues.append("Severe class imbalance detected")
        
        return issues

    def prepare_data(self, processed_df):
        """
        Prepare data for model training
        """
        texts = processed_df['processed_text'].tolist()
        labels = processed_df['sarcastic'].tolist()
        
        return train_test_split(
            texts, labels, 
            test_size=0.2, 
            random_state=42, 
            stratify=labels
        )

    def get_class_weights(self, labels):
        """
        Calculate class weights for imbalanced dataset
        """
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        return dict(enumerate(class_weights))