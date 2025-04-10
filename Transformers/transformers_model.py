import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import TextPreprocessor
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class TransformerModel:
    def __init__(self, model_name, model_type):
        self.model_name = model_name
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        ).to(self.device)
        self.training_stats = []

    def train_model(self, train_dataloader, val_dataloader, epochs=3):
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = train_loss / len(train_dataloader)
            
            # Validation
            val_metrics = self.evaluate_model(val_dataloader)
            
            # Store stats
            self.training_stats.append({
                'epoch': epoch + 1,
                'training_loss': avg_train_loss,
                'val_accuracy': val_metrics['accuracy']
            })
            
            print(f'Average training loss: {avg_train_loss}')
            print(f'Validation Accuracy: {val_metrics["accuracy"]}')
            print('Classification Report:')
            print(val_metrics["classification_report"])
            print('\n')

    def evaluate_model(self, dataloader):
        self.model.eval()
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
        
        return {
            'accuracy': accuracy_score(actual_labels, predictions),
            'classification_report': classification_report(actual_labels, predictions),
            'confusion_matrix': confusion_matrix(actual_labels, predictions)
        }

def plot_training_stats(models):
    plt.figure(figsize=(12, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for model in models:
        stats_df = pd.DataFrame(model.training_stats)
        plt.plot(stats_df['epoch'], stats_df['training_loss'], 
                marker='o', label=f'{model.model_type}')
    plt.title('Training Loss by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    for model in models:
        stats_df = pd.DataFrame(model.training_stats)
        plt.plot(stats_df['epoch'], stats_df['val_accuracy'], 
                marker='o', label=f'{model.model_type}')
    plt.title('Validation Accuracy by Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_stats.png')
    plt.close()

def plot_confusion_matrices(models, val_dataloader):
    fig, axes = plt.subplots(1, len(models), figsize=(15, 5))
    
    for idx, model in enumerate(models):
        metrics = model.evaluate_model(val_dataloader)
        cm = metrics['confusion_matrix']
        
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx])
        axes[idx].set_title(f'{model.model_type} Confusion Matrix')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def main():
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Load and preprocess data
    file_path = 'train.csv'
    processed_df = preprocessor.load_and_preprocess_data(file_path)
    
    # Check data quality
    issues = preprocessor.quality_checks(processed_df)
    if issues:
        print("\nQuality Issues Found:")
        for issue in issues:
            print(f"- {issue}")
    
    # Prepare data for training
    train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_data(processed_df)
    
    # Initialize models
    models = [
        TransformerModel('bert-base-uncased', 'BERT'),
        TransformerModel('roberta-base', 'RoBERTa')
    ]
    
    # Train and evaluate each model
    for model in models:
        print(f"\nTraining {model.model_type} model...")
        
        # Create datasets
        train_dataset = SarcasmDataset(train_texts, train_labels, model.tokenizer)
        val_dataset = SarcasmDataset(val_texts, val_labels, model.tokenizer)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16)
        
        # Train model
        model.train_model(train_dataloader, val_dataloader)
        
        # Save model
        model.model.save_pretrained(f'best_{model.model_type.lower()}_model')
        model.tokenizer.save_pretrained(f'best_{model.model_type.lower()}_model')
    
    # Plot training statistics
    plot_training_stats(models)
    
    # Plot confusion matrices
    val_dataset = SarcasmDataset(val_texts, val_labels, models[0].tokenizer)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    plot_confusion_matrices(models, val_dataloader)
    
    # Print final comparison
    print("\nFinal Model Comparison:")
    for model in models:
        metrics = model.evaluate_model(val_dataloader)
        print(f"\n{model.model_type} Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification Report:")
        print(metrics['classification_report'])

if __name__ == "__main__":
    main()