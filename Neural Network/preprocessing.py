import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import TextVectorization

class TextPreprocessor:
    def __init__(self, max_words=10000, max_len=100):
        self.max_words = max_words
        self.max_len = max_len
        self.vectorizer = TextVectorization(
            max_tokens=max_words,
            output_sequence_length=max_len,
            standardize='lower_and_strip_punctuation'
        )
        self.tfidf = TfidfVectorizer(max_features=max_words)
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = str(text)
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        text = re.sub(r'^RT[\s]+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def prepare_data(self, file_path):
        print("Loading dataset...")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Split the content into text and labels i have removed the rest of the columns that are not part of the task check train.csv
        texts = []
        labels = []
        
        # Split by digits followed by text to separate entries
        entries = re.findall(r'(.*?)(\d)(?=[A-Za-z]|$)', content)
        
        for text, label in entries:
            if text.strip():
                # Ensure binary labels (0 or 1)
                label_value = 1 if int(label) > 0 else 0
                texts.append(text.strip())
                labels.append(label_value)
        
        # Check unique labels
        unique_labels = set(labels)
        print(f"\nUnique labels in dataset: {unique_labels}")
        
        # Creating the DataFrame
        df = pd.DataFrame({
            'text': texts,
            'sarcastic': labels
        })
        
        print(f"\nLoaded {len(df)} examples")
        print("\nLabel distribution:")
        print(df['sarcastic'].value_counts())
        
        # Preprocess texts
        print("\nPreprocessing texts...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['processed_text'].str.len() > 0]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df['processed_text'].values, 
            df['sarcastic'].values,
            test_size=0.2,
            random_state=42,
            stratify=df['sarcastic'].values
        )
        
        # Convert texts to numpy arrays for vectorization
        train_texts = np.array(train_texts)
        val_texts = np.array(val_texts)
        
        # Adapting vectorizer on training data
        print("Vectorizing texts...")
        self.vectorizer.adapt(train_texts)
        
        # Transform texts to sequences
        train_sequences = self.vectorizer(train_texts).numpy()
        val_sequences = self.vectorizer(val_texts).numpy()
        
        # Create TF-IDF features
        print("Creating TF-IDF features...")
        train_tfidf = self.tfidf.fit_transform(train_texts).toarray()
        val_tfidf = self.tfidf.transform(val_texts).toarray()
        
        print("\nData shapes:")
        print(f"Train sequences: {train_sequences.shape}")
        print(f"Validation sequences: {val_sequences.shape}")
        print(f"Train TF-IDF: {train_tfidf.shape}")
        print(f"Validation TF-IDF: {val_tfidf.shape}")
        
        # Verify final label distribution
        print("\nFinal label distribution:")
        print("Training labels:", np.unique(train_labels, return_counts=True))
        print("Validation labels:", np.unique(val_labels, return_counts=True))
        
        return {
            'train_sequence': train_sequences,
            'val_sequence': val_sequences,
            'train_tfidf': train_tfidf,
            'val_tfidf': val_tfidf,
            'train_labels': train_labels,
            'val_labels': val_labels
        }

    def get_vocab_size(self):
        """Return the size of the vocabulary"""
        return len(self.vectorizer.get_vocabulary())

if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = TextPreprocessor()
    data = preprocessor.prepare_data('train.csv')
    print("\nCorpus size:", preprocessor.get_vocab_size())