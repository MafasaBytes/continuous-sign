"""Quick script to verify vocabulary is clean."""

from pathlib import Path
from src.data.dataset import build_vocabulary

# Build vocabulary with new filtering
print("Building vocabulary with filtering...")
vocab = build_vocabulary(
    Path("data/raw_data/phoenix-2014-signerindependent-SI5/annotations/manual/train.SI5.corpus.csv")
)

print(f"\nVocabulary size: {len(vocab)}")
print(f"Blank ID: {vocab.blank_id}")

# Check for problematic tokens
problematic = []
for idx, word in vocab.idx2word.items():
    if '__' in word or 'IX' in word or 'loc-' in word or 'cl-' in word:
        problematic.append((idx, word))

if problematic:
    print(f"\nWARNING: Found {len(problematic)} problematic tokens:")
    for idx, word in problematic[:10]:
        print(f"  {idx}: {word}")
else:
    print("\nVocabulary is clean! No special tokens found.")

# Show sample vocabulary
print("\nFirst 20 words (excluding blank):")
for i, (idx, word) in enumerate(vocab.idx2word.items()):
    if i > 20:
        break
    if idx != 0:  # Skip blank
        print(f"  {idx}: {word}")