# Phoenix Dataset Annotation Guide

## Special Tokens in Phoenix 2014 Dataset

The Phoenix dataset uses special annotation conventions that are **normal and expected**:

### Boundary Markers
- `__ON__`: Start of sentence/utterance
- `__OFF__`: End of sentence/utterance

### Sign Language Specific Markers
- `IX`: Pointing/deictic sign (index finger pointing)
- `loc-NORD`, `loc-SUED`, `loc-WEST`, `loc-OST`: Location markers
  - `loc-` prefix indicates spatial location
  - Common in German Sign Language for weather reports
- `__EMOTION__`: Emotional expression marker
- `__LEFTHAND__`: Two-handed sign marker (left hand involved)

### Classifier Constructions
- `cl-KOMMEN`: Classifier construction
  - `cl-` prefix indicates classifier handshape/movement

### Other Special Markers
- `IN-KOMMEND`: Compound sign (incoming)
- `MINUS`, `PLUSPLUS`: Numerical modifiers
- `__PU__`: Pause marker

## Why These Exist

These tokens are **linguistic annotations** specific to sign language:
1. Sign language is spatial - locations matter (`loc-`)
2. Sign language uses pointing - `IX` is a common deictic
3. Two-handed signs need markers (`__LEFTHAND__`)
4. Classifier constructions are grammatical (`cl-`)
5. Sentence boundaries are explicit (`__ON__`, `__OFF__`)

## Should We Filter Them?

**Option 1: Keep All Tokens** (Recommended for now)
- Preserves full linguistic information
- Model learns to handle all annotation conventions
- More accurate representation

**Option 2: Filter Special Tokens**
- Remove `__ON__`, `__OFF__`, `__EMOTION__`
- Keep content words and `loc-`, `IX` (they're part of signs)
- Simpler vocabulary but loses information

**Recommendation**: Keep all tokens for now. The model should learn to handle them. If they cause issues later, we can filter during post-processing.

## References
- Phoenix 2014 dataset documentation
- Koller et al. (2015) - Original paper describing the dataset
- Sign language linguistic annotation conventions

