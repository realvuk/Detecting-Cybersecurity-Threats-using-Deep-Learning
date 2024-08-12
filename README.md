# **Detecting-Cybersecurity-Threats-using-Deep-Learning**

This project demonstrates a binary classification model built using PyTorch. It covers end-to-end steps, including data preprocessing, model definition, training, evaluation, and performance analysis. The dataset used in this project contains labeled data for binary classification tasks.

## **Project Structure**

The directory containing the datasets:
- labelled_train.csv: Training data with features and target labels.
- labelled_test.csv: Test data with features and target labels.
- labelled_validation.csv: Validation data with features and target labels.
- Detecting_Cybersecurity_Threats_using_Deep_Learning..ipynb: Main script to load data, define and train the model, and evaluate performance.
- README.md: This file.

## **Requirements**

To run this project, you will need Python and the following libraries:

pandas: For data manipulation and analysis.
scikit-learn: For data preprocessing (e.g., scaling).
torch: For building and training the neural network.
torchmetrics: For evaluating model performance metrics.
matplotlib: For plotting loss and accuracy curves.
google.colab (if using Google Colab): For mounting Google Drive.

You can install the required libraries using pip:

```
pip install pandas scikit-learn torch torchmetrics matplotlib
```

## **Setup**

### **1. Mount Google Drive (if using Google Colab)**
If you are using Google Colab, mount your Google Drive to access the data files. Run the following code in a Colab notebook:

```
from google.colab import drive
drive.mount('/content/drive')
```


### **2. Place Data Files**

Ensure that your CSV files are located in the correct directory. You should have the following files:

labelled_train.csv
labelled_test.csv
labelled_validation.csv

If the files are not in this directory, update the paths in main.py accordingly.

## **Running the Project**

### **1. Load Data**

The script begins by loading the dataset from CSV files into Pandas DataFrames:

```
train_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data/labelled_train.csv')
test_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data/labelled_test.csv')
val_df = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Data/labelled_validation.csv')
```

It then separates features and labels:

```
X_train = train_df.drop('sus_label', axis=1).values
y_train = train_df['sus_label'].values
X_test = test_df.drop('sus_label', axis=1).values
y_test = test_df['sus_label'].values
X_val = val_df.drop('sus_label', axis=1).values
y_val = val_df['sus_label'].values
```

### **2. Data Preprocessing**

The features are scaled using StandardScaler to standardize them:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
```

The data is then converted to PyTorch tensors:

```
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
```

### **3. Define the Model**

A simple feedforward neural network is defined using nn.Sequential:

```
model = nn.Sequential(
    nn.Linear(X_train.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 1),
    nn.Sigmoid()
)
```

### **4. Training the Model**

The model is trained for 50 epochs using the Adam optimizer and binary cross-entropy loss:

```
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

num_epoch = 50
train_losses = []
val_accuracies = []
for epoch in range(num_epoch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())
    
    
  model.eval()
  with torch.no_grad():
      y_predict_val = model(X_val_tensor).round()
      accuracy = Accuracy(task="binary")
      val_accuracy = accuracy(y_predict_val, y_val_tensor).item()
      val_accuracies.append(val_accuracy)
```

### **5. Model Evaluation**

After training, the model is evaluated on the training, validation, and test datasets:

```
model.eval()
with torch.no_grad():
    y_predict_train = model(X_train_tensor).round()
    y_predict_test = model(X_test_tensor).round()
    y_predict_val = model(X_val_tensor).round()
```

Performance metrics including accuracy, precision, recall, and F1-score are calculated:

```
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

train_accuracy = accuracy(y_predict_train, y_train_tensor).item()
test_accuracy = accuracy(y_predict_test, y_test_tensor).item()
val_accuracy = accuracy(y_predict_val, y_val_tensor).item()

print("Training accuracy: {0}".format(train_accuracy))
print("Validation accuracy: {0}".format(val_accuracy))
print("Testing accuracy: {0}".format(test_accuracy))

print("Confusion Matrix (Test):")
print(confusion_matrix(y_test, y_predict_test.numpy()))
print("Precision (Test):", precision_score(y_test, y_predict_test.numpy()))
print("Recall (Test):", recall_score(y_test, y_predict_test.numpy()))
print("F1 Score (Test):", f1_score(y_test, y_predict_test.numpy()))
```

Loss and accuracy plots are generated to visualize the training process:

```
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

plt.show()
```

**Results**

The output includes:

**Training Accuracy:** Accuracy on the training set.

**Validation Accuracy:** Accuracy on the validation set.

**Testing Accuracy:** Accuracy on the test set.

**Confusion Matrix:** Matrix showing the number of true positives, true negatives, false positives, and false negatives.

**Precision, Recall, F1-Score:** Metrics for assessing the model's performance on the test set.
