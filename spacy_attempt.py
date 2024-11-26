import pandas as pd
import spacy
from typing import List, Tuple

def process_sentences(df: pd.DataFrame, forms_col: str = 'form', pos_tags_col: str = 'upos', chunk_size: int = 90000) -> List[Tuple[List[str], List[str]]]:
    """
    Process text in chunks to detect sentence boundaries and align with original DataFrame indices.
    
    Args:
        df: DataFrame containing words and POS tags
        words_col: Name of column containing words/tokens
        pos_tags_col: Name of column containing POS tags
        chunk_size: Size of text chunks to process
    
    Returns:
        List of tuples containing (words, pos_tags) for each sentence
    """
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    # Create text with word indices embedded
    forms = df[forms_col].tolist()
    indexed_text = []
    current_position = 0
    form_positions = {}  # Maps text positions to DataFrame indices
    
    for i, word in enumerate(forms):
        form_positions[current_position] = i
        indexed_text.append(word)
        current_position += len(word) + 1  # +1 for space
    
    full_text = ' '.join(indexed_text)
    grouped_data = []
    offset = 0

    # Process text in chunks
    while offset < len(full_text):
        # Determine chunk boundaries
        end = min(offset + chunk_size, len(full_text))
        if end < len(full_text):
            # Find the next space to avoid cutting words
            while end < len(full_text) and full_text[end] != ' ':
                end += 1
        
        # Extract and process chunk
        chunk_text = full_text[offset:end].strip()
        doc = nlp(chunk_text)

        print(f'{len(doc.sents)=}')
        
        # Process sentences in chunk
        for sent in doc.sents:
            # Find the start and end positions of the sentence in the original text
            sent_start_pos = offset + sent.start_char
            sent_end_pos = offset + sent.end_char
            
            # Find the corresponding word indices
            start_word_idx = None
            end_word_idx = None
            
            # Find start index
            for pos in sorted(form_positions.keys()):
                if pos <= sent_start_pos:
                    start_word_idx = form_positions[pos]
                if pos <= sent_end_pos:
                    end_word_idx = form_positions[pos]
                else:
                    break
            
            if start_word_idx is not None and end_word_idx is not None:
                # Extract words and POS tags for the sentence
                sentence_words = df.iloc[start_word_idx:end_word_idx + 1][forms_col].tolist()
                sentence_pos_tags = df.iloc[start_word_idx:end_word_idx + 1][pos_tags_col].tolist()
                
                # Only add if we have actual content
                if sentence_words and sentence_pos_tags:
                    grouped_data.append((sentence_words, sentence_pos_tags))
        
        offset = end
    
    return grouped_data

# Usage
df = pd.read_csv('ptb-train-10-all-lower.csv')
sentences = process_sentences(df)

# Print results
for i, (words, tags) in enumerate(sentences):
    print(f"Sentence {i+1}:")
    print(f"Words: {words}")
    print(f"Tags:  {tags}")
    print()