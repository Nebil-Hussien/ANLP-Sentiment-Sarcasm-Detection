from preprocessing import TextPreprocessor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# Initialize preprocessor
preprocessor = TextPreprocessor()

# Prepare data
data = preprocessor.prepare_data('train.csv') 

sequence_data = data['train_sequence']  # For CNN/LSTM
tfidf_data = data['train_tfidf']       # For Dense NN
train_labels = data['train_labels']
val_sequence = data['val_sequence']
val_tfidf = data['val_tfidf']
val_labels = data['val_labels']

#Size of the corpus
vocab_size = preprocessor.get_vocab_size()

# Define models
def create_dense_nn():
    model = Sequential([
        Dense(256, activation='relu', input_shape=(tfidf_data.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_cnn():
    model = Sequential([
        Embedding(vocab_size, 100, input_length=sequence_data.shape[1]),
        Conv1D(128, 5, activation='relu'),
        MaxPooling1D(5),
        Conv1D(128, 5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def create_lstm():
    model = Sequential([
        Embedding(vocab_size, 100, input_length=sequence_data.shape[1]),
        LSTM(64, return_sequences=True),
        LSTM(32),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    return model

def train_and_evaluate(model, x_train, x_val, y_train, y_val, model_name):
    # Compile model
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    # Train model
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=10,
        batch_size=32
    )
    
    # Evaluate model
    y_pred = model.predict(x_val)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Print classification report
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_val, y_pred_classes))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    plt.close()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title(f'{model_name} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title(f'{model_name} - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_history.png')
    plt.close()

# Training Dense Neural Network but my feature enginnering is not that good
print("\nTraining Dense Neural Network...")
dense_model = create_dense_nn()
train_and_evaluate(dense_model, tfidf_data, val_tfidf, train_labels, val_labels, "Dense_NN")

# Train CNN but my feature enginnering is not that good
print("\nTraining CNN...")
cnn_model = create_cnn()
train_and_evaluate(cnn_model, sequence_data, val_sequence, train_labels, val_labels, "CNN")

# Train LSTM but my feature enginnering is not that good
print("\nTraining LSTM...")
lstm_model = create_lstm()
train_and_evaluate(lstm_model, sequence_data, val_sequence, train_labels, val_labels, "LSTM")