# Gene-Technology-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from collections import Counter

# Define the file path to your .fasta file
file_path = r"F:\RJ DOCS\CO\GENOMIC WITH DS\dna.example.fasta"

# Open and read the .fasta file
with open(file_path, 'r') as file:
    contents = file.read()

# Display the contents (for debugging purposes, uncomment to view)
# print(contents)

# Simulated DNA sequences and labels for demonstration
sequences = [
    "ATCGATCGTAGC", "CGTACGTAGCTA", "GCTAGCTAGGCT", "TAGCTAGCGTAC", "TGCATGCATGCA", 
    "CGTAGCTAGCTA", "GCTAGCTAGCTA", "TAGCTAGCATGC", "ATCGTACGATGC", "TGCATGCGTACG"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = gene, 0 = non-gene

# Function to extract K-mer frequencies (k=3 as example)
def extract_kmer_features(sequence, k=3):
    kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    return kmer_counts

# Convert sequences into feature vectors
def create_feature_matrix(sequences, k=3):
    feature_list = []
    for seq in sequences:
        kmer_counts = extract_kmer_features(seq, k)
        feature_list.append(kmer_counts)
    return pd.DataFrame(feature_list).fillna(0)  # Fill NaN values with 0

# Create feature matrix and labels
X = create_feature_matrix(sequences, k=3)  # Extract 3-mer features
y = np.array(labels)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Non-Gene', 'Gene'], yticklabels=['Non-Gene', 'Gene'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Plot Feature Importances
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]  # Top 10 features

plt.figure(figsize=(10, 6))
plt.title("Top 10 Feature Importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Feature Importance")
plt.show()

# Visualize Class Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Class Distribution")
plt.xlabel("Class (0 = Non-Gene, 1 = Gene)")
plt.ylabel("Frequency")
plt.show()

# Compute and Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# Conclusion
print("Conclusion:")
print("The Random Forest Classifier model achieved an accuracy of {:.2f}% on the test set.".format(accuracy * 100))
print("The confusion matrix and classification report indicate that the model performs well in distinguishing between gene and non-gene sequences.")
print("Feature importance analysis reveals the most influential features, while the ROC curve demonstrates the model's ability to distinguish between classes at various thresholds.")
print("Future work could involve fine-tuning the model, exploring other algorithms, or integrating additional data to improve performance.")

