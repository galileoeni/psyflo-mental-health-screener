import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel

# Load models
theme_model = joblib.load('bert_classifier.pkl')
urgency_model = joblib.load('urgency_model.pkl')  # Assuming you trained this similarly
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

def get_recommendations(theme, urgency):
    resources = {
        'anxiety': {
            'high': ["Immediate: 5-4-3-2-1 grounding technique", "Call campus counseling: 555-1234"],
            'medium': ["Schedule therapist visit", "Try box breathing exercises"],
            'low': ["Journal daily", "Download Calm app"]
        },
        'depression': {
            'high': ["Emergency hotline: 988", "Contact student health services NOW"],
            'medium': ["Schedule wellness check", "Reach out to support group"],
            'low': ["Daily gratitude practice", "15-min sunlight exposure"]
        },
        'academic_stress': {
            'high': ["Academic crisis consultation", "Request deadline extensions"],
            'medium': ["Time management workshop", "Tutoring center visit"],
            'low': ["Pomodoro technique", "Study group formation"]
        }
    }
    return resources.get(theme, {}).get(urgency, ["General self-care tips"])

print("\nMental Health Journal Analyzer")
print("----------------------------")
while True:
    entry = input("\nEnter your journal entry (or 'quit'): ")
    if entry.lower() == 'quit':
        break
    
    # Get predictions
    embedding = get_bert_embedding(entry)
    theme = theme_model.predict(embedding)[0]
    urgency = urgency_model.predict(embedding)[0]
    
    # Get recommendations
    recs = get_recommendations(theme, urgency)
    
    # Print results
    print(f"\nAnalysis Results:")
    print(f"Detected Theme: {theme.upper()}")
    print(f"Urgency Level: {urgency.upper()}")
    print("\nRecommended Actions:")
    for i, action in enumerate(recs, 1):
        print(f"{i}. {action}")
    
    print("\n(Enter another entry or 'quit' to exit)")