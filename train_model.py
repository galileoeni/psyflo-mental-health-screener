import joblib
import pandas as pd
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load pre-trained DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get embeddings
def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()  # Use [CLS] token embedding

# Load data
df = pd.read_csv('mental_health.csv')
texts = df['Journal_Entry'].tolist()

# Generate embeddings
embeddings = get_embeddings(texts)
joblib.dump(embeddings, 'bert_embeddings.pkl')  
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, df['Mental_State'], test_size=0.4, random_state=42
)

# Train classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)


joblib.dump(clf, 'bert_classifier.pkl')

# Train Urgency Classifier
X_train, X_test, y_urgency_train, y_urgency_test = train_test_split(
    embeddings, df['Urgency'], test_size=0.2, random_state=42
)
urgency_clf = LogisticRegression(max_iter=1000)
urgency_clf.fit(X_train, y_urgency_train)

# Save models
joblib.dump(urgency_clf, 'urgency_model.pkl')
