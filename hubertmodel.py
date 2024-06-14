import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils import shuffle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, matthews_corrcoef
from transformers import HubertModel, Wav2Vec2Processor
import matplotlib.pyplot as plt
import pandas as pd

data_dir = 'data'
healthy_files = [f'{data_dir}/{i}.mp3' for i in range(1, 1041)]
cardiac_arrest_files = [f'{data_dir}/{i}.mp3' for i in range(1041, 2040)]
heart_attack_files = [f'{data_dir}/{i}.mp3' for i in range(2040, 3048)]
heart_valve_disease_files = [f'{data_dir}/{i}.mp3' for i in range(3048, 4116)]
arrhythmia_files = [f'{data_dir}/{i}.mp3' for i in range(4116, 5138)]

file_paths = healthy_files + cardiac_arrest_files + heart_attack_files + heart_valve_disease_files + arrhythmia_files
labels = [0]*len(healthy_files) + [1]*len(cardiac_arrest_files) + [2]*len(heart_attack_files) + [3]*len(heart_valve_disease_files) + [4]*len(arrhythmia_files)

file_paths, labels = shuffle(file_paths, labels, random_state=42)

# HuBERT modelini ve tokenizer'ı yükleme
processor = Wav2Vec2Processor.from_pretrained('facebook/hubert-large-ls960-ft')
model = HubertModel.from_pretrained('facebook/hubert-large-ls960-ft')


def augment_audio(audio):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + 0.005 * noise
    return augmented_audio


# Ses dosyalarından özellik çıkarma fonksiyonu
def extract_features(file_list, augment=False):
    features = []
    for file in file_list:
        audio, rate = librosa.load(file, sr=16000)
        if augment:
            audio = augment_audio(audio)
        
        input_values = processor(audio, return_tensors='pt', sampling_rate=16000).input_values
        with torch.no_grad():
            hidden_states = model(input_values).last_hidden_state
        
        features.append(hidden_states.mean(dim=1).cpu().numpy())
    
    return np.vstack(features)

# Veriyi train ve test olarak bölme
split_idx = int(0.8 * len(file_paths))
train_files, test_files = file_paths[:split_idx], file_paths[split_idx:]
train_labels, test_labels = labels[:split_idx], labels[split_idx:]

# Özellikleri çıkarma
train_features = extract_features(train_files, augment=True)
test_features = extract_features(test_files)

# Tensor dönüşümleri
X_train = torch.tensor(train_features, dtype=torch.float32)
y_train = torch.tensor(train_labels, dtype=torch.long)
X_test = torch.tensor(test_features, dtype=torch.float32)
y_test = torch.tensor(test_labels, dtype=torch.long)

# Karmaşık bir sinir ağı(İleri beslemeli) modeli oluşturma
class ComplexNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ComplexNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


input_size = X_train.shape[1]
num_classes = len(set(labels))
model = ComplexNN(input_size, num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# K-Fold Cross-Validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fpr = dict()
tpr = dict()
roc_auc = dict()

early_stopping_patience = 10

num_epochs = 10
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
    print(f'Fold {fold+1}/{kfold.n_splits}')
    
    X_train_fold = X_train[train_idx]
    y_train_fold = y_train[train_idx]
    X_val_fold = X_train[val_idx]
    y_val_fold = y_train[val_idx]

    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Eğitim
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_fold)
        train_loss = criterion(outputs, y_train_fold)
        train_loss.backward()
        optimizer.step()
        
        train_losses.append(train_loss.item())
        _, predicted = torch.max(outputs, 1)
        train_accuracy = (predicted == y_train_fold).float().mean().item()
        train_accuracies.append(train_accuracy)
        
        # Doğrulama
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = criterion(val_outputs, y_val_fold)
            val_losses.append(val_loss.item())
            _, val_predicted = torch.max(val_outputs, 1)
            val_accuracy = (val_predicted == y_val_fold).float().mean().item()
            val_accuracies.append(val_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss.item():.4f}, '
              f'Val Loss: {val_loss.item():.4f}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Val Accuracy: {val_accuracy:.4f}')
        
        # Erken durdurma
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Erken durdurma tetiklendi.")
                break

    # ROC Eğrisi hesaplama
    with torch.no_grad():
        val_probs = torch.softmax(model(X_val_fold), dim=1).cpu().numpy()
        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve((y_val_fold == i).cpu().numpy(), val_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

# ROC Eğrileri
plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc='lower right')
plt.show()

# Loss vs epochs grafikleri çizme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()

plt.show()

# Test seti üzerinde modelin performansını değerlendirme
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_probs = torch.softmax(test_outputs, dim=1).cpu().numpy()
    _, test_predicted = torch.max(test_outputs, 1)
    test_accuracy = (test_predicted == y_test).float().mean().item()
    print(f'Test Accuracy: {test_accuracy:.4f}')

# Performans metriklerini hesaplama
accuracy = accuracy_score(y_test, test_predicted)
precision = precision_score(y_test, test_predicted, average='weighted')
recall = recall_score(y_test, test_predicted, average='weighted')
f1 = f1_score(y_test, test_predicted, average='weighted')
mcc = matthews_corrcoef(y_test, test_predicted)

# Sensitivity ve Specificity hesaplama
cm = confusion_matrix(y_test, test_predicted)

TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (TP + FP + FN)

sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

# AUC
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve((y_test == i).cpu().numpy(), test_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Metrikleri tablo formatında gösterme
metrics = {
    'Accuracy': [accuracy],
    'Precision': [precision],
    'Recall': [recall],
    'F1 Score': [f1],
    'MCC': [mcc],
    'Sensitivity': [sensitivity.mean()],
    'Specificity': [specificity.mean()],
    'AUC': [np.mean(list(roc_auc.values()))]
}

metrics_df = pd.DataFrame(metrics)
print(metrics_df)

print("Tamamlandı.")
