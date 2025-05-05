
# Mental Health Journal Analyzer

An NLP-powered tool that analyzes journal entries to detect mental health themes and urgency levels, providing personalized recommendations.

## Features

- Detects 3 mental health themes: `anxiety`, `depression`, and `academic_stress`
- Classifies urgency levels: `low`, `medium`, `high`
- Provides tiered recommendations based on analysis
- Uses DistilBERT for state-of-the-art text understanding

## Prerequisites

- Python 3.8+
- 
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mental-health-analyzer.git
   cd mental-health-analyzer
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # venv\Scripts\activate  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Place your dataset (`data.csv`) in the project root with these columns:
- `Journal_Entry`: Text entries to analyze
- `Mental_State`: Ground truth labels (anxiety/depression/academic_stress)
- `Urgency`: Urgency levels (low/medium/high)

## Training the Models

Run the training script:
```bash
python train_model.py
```

This will:
1. Generate BERT embeddings
2. Train two classifiers:
   - Theme classifier (saved as `bert_classifier.pkl`)
   - Urgency classifier (saved as `urgency_model.pkl`)
3. Print performance metrics

## Using the Analyzer

Run the interactive demo:
```bash
python demo.py
```

Example session:
```
Enter your journal entry: I can't focus on my studies and feel hopeless

Analysis Results:
Detected Theme: ACADEMIC_STRESS
Urgency Level: HIGH

Recommended Actions:
1. Academic crisis consultation
2. Request deadline extensions
```

## File Structure

```
├── data/
│   └── data.csv                  # Input dataset
├── models/
│   ├── bert_classifier.pkl       # Theme classifier
│   └── urgency_model.pkl         # Urgency classifier
├── train_model.py                # Training script
├── demo.py                       # Interactive analyzer
├── requirements.txt              # Dependencies
└── README.md                     # This file
```

## Technical Details

### Models Used
- **Text Embedding**: DistilBERT (distilbert-base-uncased)
- **Classifiers**: Logistic Regression (scikit-learn)


### Requirements File (`requirements.txt`)
```
transformers
torch
scikit-learn
joblib
pandas
```

