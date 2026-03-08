# clean_activities.py
import pandas as pd

# Load your labels
labels_path = "/home/mitsdu/atm_experiment/annotations/final_labels_5videos.csv"
df = pd.read_csv(labels_path)

# Consolidate similar activities
activity_mapping = {
    'suspecious': 'suspicious',
    'cahs out': 'cash_out',
    'cash out': 'cash_out',
    'card out': 'card_out',
    'leaving': 'leaving',
    'nno person':'no person',
    'lealeaving':'leaving',
    '\leaving':'leaving',
     'approaching':'approaching atm',
     'approachign atm':'approaching atm',
     'instering card':'inserting card'
}

df['activity'] = df['activity'].replace(activity_mapping)

# Save cleaned version
cleaned_path = "/home/mitsdu/atm_experiment/annotations/labels_cleaned.csv"
df.to_csv(cleaned_path, index=False)

print("✅ Cleaned activities:")
print(df['activity'].value_counts())
