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