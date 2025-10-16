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


def preprocess_goemotions():
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Check what splits we have
    available_splits = list(dataset.keys())
    print(f"Available splits: {available_splits}")

    # Get train data
    full_train_data = dataset['train']

    # Create validation split (85/15)
    print("Creating validation split from training data...")
    train_size = int(0.85 * len(full_train_data))
    indices = list(range(len(full_train_data)))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_data = full_train_data.select(train_indices)
    val_data = full_train_data.select(val_indices)

    # Create test split from validation (50/50)
    print("Creating test split from validation data...")
    val_size = len(val_data) // 2
    val_indices_list = list(range(len(val_data)))

    new_val_indices = val_indices_list[:val_size]
    test_indices = val_indices_list[val_size:]

    new_val_data = val_data.select(new_val_indices)
    test_data = val_data.select(test_indices)
    val_data = new_val_data

    print(f"\nFinal split sizes:")
    print(f"Train: {len(train_data)}")
    print(f"Validation: {len(val_data)}")
    print(f"Test: {len(test_data)}")

    # Create datasets
    train_dataset = GoEmotionsDataset(train_data, tokenizer)
    val_dataset = GoEmotionsDataset(val_data, tokenizer)
    test_dataset = GoEmotionsDataset(test_data, tokenizer)

    # Create dataloaders - REDUCED num_workers to avoid multiprocessing issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, tokenizer

# Execute
train_loader, val_loader, test_loader, tokenizer = preprocess_goemotions()

print(f"\n✓ Training batches: {len(train_loader)}")
print(f"✓ Validation batches: {len(val_loader)}")
print(f"✓ Test batches: {len(test_loader)}")

# Test a batch
batch = next(iter(train_loader))
print(f"\nBatch shapes:")
print(f"Input IDs: {batch['input_ids'].shape}")
print(f"Attention mask: {batch['attention_mask'].shape}")
print(f"Labels: {batch['labels'].shape}")  # Should be [32, 27]
print(f"\nSample label vector: {batch['labels'][0]}")
print(f"Number of active emotions in first sample: {batch['labels'][0].sum().item()}")


# Analyze text lengths to determine optimal max_length
text_lengths = [len(text.split()) for text in dataset['train']['text']]

plt.figure(figsize=(12, 5))
plt.hist(text_lengths, bins=50, edgecolor='black')
plt.axvline(x=128, color='r', linestyle='--', label='Max length (128 tokens)', linewidth=2)
plt.title('Distribution of Text Lengths in GoEmotions')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.legend()
plt.show()

print(f"Average length: {sum(text_lengths)/len(text_lengths):.2f} words")
print(f"Median length: {sorted(text_lengths)[len(text_lengths)//2]} words")
print(f"95th percentile: {sorted(text_lengths)[int(len(text_lengths)*0.95)]} words")
print(f"Max length: {max(text_lengths)} words")


