from datasets import load_dataset

dataset = load_dataset('google-research-datasets/go_emotions', 'raw')
print(dataset['train'][0])

# First, let's see what columns are actually in the dataset
print("Dataset keys:")
print(dataset['train'].features)
print("\nColumn names:")
print(dataset['train'].column_names)

# Check a sample
print("\nSample data:")
print(dataset['train'][0])

emotion_columns = [col for col in dataset['train'].column_names
                   if col not in ['text', 'id', 'author', 'subreddit', 'link_id',
                                  'parent_id', 'created_utc', 'rater_id', 'example_very_unclear']]

print(f"Number of emotion columns: {len(emotion_columns)}")
print(f"Emotions: {emotion_columns}")

# Update EMOTIONS list
EMOTIONS = emotion_columns
print(f"\nUpdated EMOTIONS list with {len(EMOTIONS)} emotions")

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import numpy as np

# Define all 27 emotions
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

print(f"Total emotions: {len(EMOTIONS)}")  # Should print 28

class GoEmotionsDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = str(item['text'])

        # Create multi-label vector (28 emotions) - FIXED
        labels = []
        for emotion in EMOTIONS:
            labels.append(float(item[emotion]))
        labels = torch.tensor(labels, dtype=torch.float)

        # Tokenization
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': labels
        }
        
# Check available splits
print("Available splits in dataset:")
print(dataset.keys())
print("\nDataset structure:")
print(dataset)

print(f"Total emotions: {len(EMOTIONS)}")  # Should be 27


