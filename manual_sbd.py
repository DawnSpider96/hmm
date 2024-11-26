# Sadly unneeded because the Conllu file contains sentences

import pandas as pd
from typing import List, Tuple

def detect_sentence_boundaries(df: pd.DataFrame, words_col: str = 'form', pos_tags_col: str = 'upos') -> List[Tuple[List[str], List[str]]]:
    """
    Brute force sentence boundary detection based on punctuation
    (Covers the cases of ending quotation marks)

    """
    words = df[words_col].tolist()
    tags = df[pos_tags_col].tolist()
    
    sentences = []
    current_sentence_words = []
    current_sentence_tags = []
    
    for i, (word, tag) in enumerate(zip(words, tags)):
        current_sentence_words.append(word)
        current_sentence_tags.append(tag)
        
        # Check for sentence endings
        is_end = False
        if word in ['.', '!', '?']: 
            if i + 1 < len(words) and words[i + 1] == '"':
                # Covers the cases of ." ?" !" 
                # by not ending the sentence yet
                # works because only ending quotation marks are like this
                # starting quotation marks are ``
                continue
            is_end = True
        elif word == '"' and i > 0 and words[i-1] == '.':
            is_end = True
            
        if is_end and current_sentence_words:
            sentences.append((current_sentence_words.copy(), current_sentence_tags.copy()))
            current_sentence_words = []
            current_sentence_tags = []
    
    # Don't forget last sentence if it doesn't end with punctuation
    if current_sentence_words:
        sentences.append((current_sentence_words, current_sentence_tags))
    
    return sentences

# Usage
df = pd.read_csv('ptb-train-10-all-lower.csv')
sentences = detect_sentence_boundaries(df)

# Print first few sentences to verify
for i, (words, tags) in enumerate(sentences[:5]):
    print(f"Sentence {i+1}:")
    print(f"Words: {' '.join(words)}")
    print(f"Tags:  {' '.join(tags)}")
    print()