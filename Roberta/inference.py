import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define models
MODEL_NAMES = {
    "roberta": "roberta-large",
    "deberta": "microsoft/deberta-v3-large",
    "xlm-roberta": "xlm-roberta-large"
}

def load_models(model_dir="./saved_models"):
    """Load trained models and tokenizers."""
    models = {}
    tokenizers = {}
    
    for model_name in MODEL_NAMES:
        model_path = os.path.join(model_dir, model_name)
        
        if os.path.exists(model_path):
            print(f"Loading {model_name} from {model_path}")
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            
            tokenizers[model_name] = tokenizer
            models[model_name] = model
        else:
            print(f"Model {model_name} not found at {model_path}")
    
    return models, tokenizers

def preprocess_text(text, tokenizer, max_length=128):
    """Preprocess a single text for inference."""
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    return inputs

def get_prediction(text, model, tokenizer, device):
    """Get prediction for a single text."""
    model.to(device)
    model.eval()
    
    inputs = preprocess_text(text, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = np.argmax(probs)
    
    return pred, probs

def ensemble_predict(text, models, tokenizers, voting="soft", device="cpu"):
    """Make an ensemble prediction on a single text."""
    all_probs = []
    all_preds = []
    
    for model_name, model in models.items():
        tokenizer = tokenizers[model_name]
        pred, probs = get_prediction(text, model, tokenizer, device)
        
        all_probs.append(probs)
        all_preds.append(pred)
    
    if voting == "soft":
        # Average probabilities
        avg_probs = np.mean(all_probs, axis=0)
        final_pred = np.argmax(avg_probs)
        confidence = avg_probs[final_pred]
    else:  # Hard voting
        # Take majority vote
        final_pred = np.bincount(all_preds).argmax()
        # Calculate confidence as proportion of models that voted for this class
        confidence = np.sum(np.array(all_preds) == final_pred) / len(all_preds)
    
    return final_pred, confidence

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    models, tokenizers = load_models()
    
    if not models:
        print("No models found. Please train models first using ensemble_model.py")
        return
    
    # Example texts
    example_texts = [
        "I love waiting in traffic for hours.",
        "The weather is beautiful today.",
        "Oh great, another meeting that could have been an email.",
        "This is the best movie I've ever seen!",
        "I'm so happy to be working on a weekend."
    ]
    
    print("\nMaking predictions on example texts:")
    for text in example_texts:
        # Soft voting
        soft_pred, soft_conf = ensemble_predict(text, models, tokenizers, voting="soft", device=device)
        # Hard voting
        hard_pred, hard_conf = ensemble_predict(text, models, tokenizers, voting="hard", device=device)
        
        print(f"\nText: \"{text}\"")
        print(f"Soft Voting: {'Sarcastic' if soft_pred == 1 else 'Not Sarcastic'} (confidence: {soft_conf:.4f})")
        print(f"Hard Voting: {'Sarcastic' if hard_pred == 1 else 'Not Sarcastic'} (confidence: {hard_conf:.4f})")
    
    # Interactive mode
    print("\n" + "="*50)
    print("Interactive Mode: Type a text to check for sarcasm (or 'quit' to exit)")
    print("="*50)
    
    while True:
        user_input = input("\nEnter text: ")
        if user_input.lower() == 'quit':
            break
        
        # Soft voting
        soft_pred, soft_conf = ensemble_predict(user_input, models, tokenizers, voting="soft", device=device)
        # Hard voting
        hard_pred, hard_conf = ensemble_predict(user_input, models, tokenizers, voting="hard", device=device)
        
        print(f"Soft Voting: {'Sarcastic' if soft_pred == 1 else 'Not Sarcastic'} (confidence: {soft_conf:.4f})")
        print(f"Hard Voting: {'Sarcastic' if hard_pred == 1 else 'Not Sarcastic'} (confidence: {hard_conf:.4f})")

if __name__ == "__main__":
    main() 