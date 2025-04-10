import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    RobertaTokenizer, 
    AlbertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Add parent directory to path to import dataloading module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloading import get_main_df, get_full_df, get_non_main_df

# Define models
MODEL_NAMES = {
    "distilroberta": "distilroberta-base",
    "albert": "albert-base-v2",
    "roberta": "roberta-base"
}

# Define tokenizer classes for the selected models
TOKENIZER_CLASSES = {
    "distilroberta": "RobertaTokenizer",
    "albert": "AlbertTokenizer",
    "roberta": "RobertaTokenizer"
}

# Define model-specific learning rates
MODEL_LEARNING_RATES = {
    "distilroberta": 3e-5,
    "albert": 4e-5,
    "roberta": 1e-7
}

class SarcasmDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(model, train_loader, optimizer, scheduler, device):
    """Train the model for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate_model(model, test_loader, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    return accuracy, f1, np.array(all_probs), np.array(all_preds), np.array(all_labels)

def get_model_predictions(models, tokenizers, test_texts, test_labels, device, batch_size=16):
    """Get predictions from all models."""
    all_model_probs = {}
    all_model_preds = {}
    
    for model_name, model in models.items():
        print(f"\nGetting predictions from {model_name}...")
        tokenizer = tokenizers[model_name]
        
        # Create dataset and dataloader
        test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Get predictions
        _, _, probs, preds, _ = evaluate_model(model, test_loader, device)
        
        all_model_probs[model_name] = probs
        all_model_preds[model_name] = preds
    
    return all_model_probs, all_model_preds

def ensemble_predictions(all_model_probs, all_model_preds, true_labels, voting="soft"):
    """Ensemble predictions using hard or soft voting."""
    if voting == "soft":
        # Average probabilities across models
        avg_probs = np.mean([all_model_probs[model] for model in all_model_probs], axis=0)
        final_preds = np.argmax(avg_probs, axis=1)
    else:  # Hard voting
        # Take majority vote
        votes = np.array([all_model_preds[model] for model in all_model_preds])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=votes)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, final_preds)
    f1 = f1_score(true_labels, final_preds)
    report = classification_report(true_labels, final_preds)
    
    return accuracy, f1, report, final_preds

def save_model(model, tokenizer, model_name):
    """Save the trained model and tokenizer."""
    output_dir = os.path.join("saved_models", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model {model_name} saved to {output_dir}")

def train_stacking_model(all_model_probs, true_labels):
    """Train a meta-model that learns to combine base model predictions."""
    # Create a feature matrix from all model predictions
    X = np.column_stack([all_model_probs[model][:, 1] for model in all_model_probs])
    
    meta_model = LogisticRegression()
    meta_model.fit(X, true_labels)
    
    # Report in-sample performance (for information only)
    train_preds = meta_model.predict(X)
    train_acc = accuracy_score(true_labels, train_preds)
    train_f1 = f1_score(true_labels, train_preds)
    
    print(f"Stacking Ensemble (in-sample): Accuracy = {train_acc:.4f}, F1 = {train_f1:.4f}")
    
    return meta_model

def load_model_and_tokenizer(model_name, model_path, device):
    """Load a model and its tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except ValueError as e:
        print(f"Fast tokenizer failed, falling back to slow tokenizer: {e}")
        if model_name == "albert":
            tokenizer = AlbertTokenizer.from_pretrained(model_path)
        elif model_name == "distilroberta":
            tokenizer = RobertaTokenizer.from_pretrained(model_path)
        else:
            raise ValueError(f"No tokenizer class defined for {model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)
    
    return model, tokenizer

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train ensemble models for sarcasm detection')
    
    parser.add_argument('--data_option', type=str, default='full',
                        choices=['main_only', 'full', 'exclude_main'],
                        help='Data loading option: main_only, full, or exclude_main')
    
    parser.add_argument('--emojihash_level', type=int, default=1,
                        choices=[0, 1, 2],
                        help='Emoji/hashtag processing level: 0 (none), 1 (emojis), 2 (emojis and hashtags)')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training and evaluation')
    
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate for optimizer')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directory for saved models
    os.makedirs("./saved_models", exist_ok=True)
    
    # Load data based on selected option
    print(f"Loading data with option: {args.data_option}, emoji/hashtag level: {args.emojihash_level}")
    
    if args.data_option == 'main_only':
        train_df, test_df = get_main_df()
    elif args.data_option == 'full':
        train_df, test_df = get_full_df(args.emojihash_level)
    elif args.data_option == 'exclude_main':
        train_df, test_df = get_non_main_df(args.emojihash_level)
    else:
        raise ValueError("Invalid data option. Choose from: 'main_only', 'full', 'exclude_main'")
    
    # Extract text and labels
    # Handle different column names between datasets
    train_text_col = 'tweet' if 'tweet' in train_df.columns else 'text'
    test_text_col = 'tweet' if 'tweet' in test_df.columns else 'text'
    
    train_texts = train_df[train_text_col].values
    train_labels = train_df['sarcastic'].values
    
    test_texts = test_df[test_text_col].values
    test_labels = test_df['sarcastic'].values
    
    # Convert string labels to integers if needed
    if isinstance(train_labels[0], str):
        train_labels = np.array([int(label) for label in train_labels])
    if isinstance(test_labels[0], str):
        test_labels = np.array([int(label) for label in test_labels])
    
    print(f"Loaded {len(train_texts)} training samples and {len(test_texts)} test samples")
    
    # Training parameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    
    # Initialize dictionaries to store models and tokenizers
    trained_models = {}
    tokenizers = {}
    individual_results = {}
    
    # Train and evaluate each model
    for model_name, model_path in MODEL_NAMES.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Load tokenizer and model
        model, tokenizer = load_model_and_tokenizer(model_name, model_path, device)
        
        # Store tokenizer
        tokenizers[model_name] = tokenizer
        
        # Create datasets and dataloaders
        train_dataset = SarcasmDataset(train_texts, train_labels, tokenizer)
        test_dataset = SarcasmDataset(test_texts, test_labels, tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize optimizer with model-specific learning rate
        learning_rate = MODEL_LEARNING_RATES.get(model_name, args.learning_rate)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Training parameters
        num_training_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0.1 * num_training_steps,
            num_training_steps=num_training_steps
        )
        
        # Training loop
        best_f1 = 0
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            avg_loss = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Average loss: {avg_loss:.4f}")
            
            # Evaluate
            accuracy, f1, _, _, _ = evaluate_model(model, test_loader, device)
            print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            
            # Save best model based on F1 score
            if f1 > best_f1:
                best_f1 = f1
                save_model(model, tokenizer, model_name)
        
        # Load the best model for this architecture
        try:
            # Try to load from the saved path
            saved_model_path = os.path.join("saved_models", model_name)
            print(f"Loading model from {saved_model_path}")
            model, tokenizer = load_model_and_tokenizer(model_name, saved_model_path, device)
            trained_models[model_name] = model
        except Exception as e:
            print(f"Error loading saved model: {e}")
            print("Using the last trained model instead.")
            trained_models[model_name] = model
        
        # Final evaluation
        accuracy, f1, _, _, _ = evaluate_model(model, test_loader, device)
        individual_results[model_name] = {'accuracy': accuracy, 'f1': f1}
        
        print(f"\nFinal results for {model_name}:")
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    # Get predictions from all models
    all_model_probs, all_model_preds = get_model_predictions(
        trained_models, tokenizers, test_texts, test_labels, device
    )
    
    # Ensemble with soft voting
    print("\n\nEnsemble with Soft Voting:")
    soft_acc, soft_f1, soft_report, _ = ensemble_predictions(
        all_model_probs, all_model_preds, test_labels, voting="soft"
    )
    print(f"Accuracy: {soft_acc:.4f}, F1 Score: {soft_f1:.4f}")
    print("Classification Report:")
    print(soft_report)
    
    # Ensemble with hard voting
    print("\n\nEnsemble with Hard Voting:")
    hard_acc, hard_f1, hard_report, _ = ensemble_predictions(
        all_model_probs, all_model_preds, test_labels, voting="hard"
    )
    print(f"Accuracy: {hard_acc:.4f}, F1 Score: {hard_f1:.4f}")
    print("Classification Report:")
    print(hard_report)
    
    # Print summary of all results
    print("\n\nSummary of Results:")
    print("Individual Models:")
    for model_name, metrics in individual_results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print("\nEnsemble Methods:")
    print(f"Soft Voting: Accuracy = {soft_acc:.4f}, F1 Score = {soft_f1:.4f}")
    print(f"Hard Voting: Accuracy = {hard_acc:.4f}, F1 Score = {hard_f1:.4f}")

    # Train stacking model
    stacking_model = train_stacking_model(all_model_probs, test_labels)

if __name__ == "__main__":
    main()
