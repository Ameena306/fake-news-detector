import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the datasets
fake_df = pd.read_csv("dataset/Fake.csv")
real_df = pd.read_csv("dataset/True.csv")

# Add labels
fake_df['label'] = 'FAKE'
real_df['label'] = 'REAL'

# Combine both datasets
df = pd.concat([fake_df, real_df], ignore_index=True)
df = df[['title', 'text', 'label']]  # Keep only needed columns

# Optional: Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Clean text function
def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))  # remove special characters
    text = re.sub(r'\s+', ' ', text)      # remove extra spaces
    return text.lower()

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Save cleaned data
df.to_csv("dataset/cleaned_news.csv", index=False)

print("✅ Preprocessing complete. Cleaned data saved to: dataset/cleaned_news.csv")
print(df.head())
