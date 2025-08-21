"""
Text processing utilities for TTS - phoneme conversion and normalization
"""
import re
import string
import logging
from typing import List, Dict, Optional
from pathlib import Path
import unicodedata

logger = logging.getLogger(__name__)

# Basic phoneme set for English (simplified)
PHONEME_SET = [
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants  
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH',
    # Special tokens
    'SIL',  # Silence
    'SP',   # Short pause
    'UNK',  # Unknown
]

# Create phoneme to index mapping
PHONEME_TO_ID = {phoneme: i for i, phoneme in enumerate(PHONEME_SET)}
ID_TO_PHONEME = {i: phoneme for i, phoneme in enumerate(PHONEME_SET)}

# Basic text cleaners
def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations."""
    abbreviations = {
        'dr.': 'doctor',
        'mr.': 'mister', 
        'mrs.': 'missus',
        'ms.': 'miss',
        'st.': 'saint',
        'etc.': 'et cetera',
        'vs.': 'versus',
        'e.g.': 'for example',
        'i.e.': 'that is',
        '&': 'and',
    }
    
    text_lower = text.lower()
    for abbrev, expansion in abbreviations.items():
        text_lower = text_lower.replace(abbrev, expansion)
    
    return text_lower


def expand_numbers(text: str) -> str:
    """Basic number expansion (simplified)."""
    # This is a very basic implementation
    # For production, use a proper number-to-words library
    
    number_words = {
        '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
        '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
        '10': 'ten', '11': 'eleven', '12': 'twelve', '13': 'thirteen',
        '14': 'fourteen', '15': 'fifteen', '16': 'sixteen', '17': 'seventeen',
        '18': 'eighteen', '19': 'nineteen', '20': 'twenty'
    }
    
    # Replace simple numbers
    words = text.split()
    result = []
    
    for word in words:
        # Remove punctuation for number checking
        clean_word = word.strip(string.punctuation)
        
        if clean_word.isdigit() and clean_word in number_words:
            # Replace number but preserve surrounding punctuation
            prefix = word[:len(word) - len(word.lstrip(string.punctuation))]
            suffix = word[len(word.rstrip(string.punctuation)):]
            result.append(prefix + number_words[clean_word] + suffix)
        else:
            result.append(word)
    
    return ' '.join(result)


def normalize_text(text: str) -> str:
    """Basic text normalization."""
    # Convert to lowercase
    text = text.lower()
    
    # Normalize unicode
    text = unicodedata.normalize('NFD', text)
    
    # Expand abbreviations
    text = expand_abbreviations(text)
    
    # Expand numbers
    text = expand_numbers(text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text


class SimpleG2P:
    """
    Simplified grapheme-to-phoneme converter.
    This is a basic implementation for POC. For production, use a proper G2P system.
    """
    
    def __init__(self):
        """Initialize with basic English pronunciation rules."""
        self.word_to_phonemes = self._create_basic_dictionary()
    
    def _create_basic_dictionary(self) -> Dict[str, List[str]]:
        """Create basic pronunciation dictionary for common words."""
        # This is a minimal dictionary for POC
        # In production, use CMU Pronunciation Dictionary or similar
        
        basic_dict = {
            # Common words for testing
            'hello': ['HH', 'EH', 'L', 'OW'],
            'world': ['W', 'ER', 'L', 'D'],
            'the': ['DH', 'AH'],
            'and': ['AE', 'N', 'D'],
            'to': ['T', 'UW'],
            'a': ['AH'],
            'of': ['AH', 'V'],
            'in': ['IH', 'N'],
            'is': ['IH', 'Z'],
            'it': ['IH', 'T'],
            'you': ['Y', 'UW'],
            'that': ['DH', 'AE', 'T'],
            'he': ['HH', 'IY'],
            'was': ['W', 'AH', 'Z'],
            'for': ['F', 'ER'],
            'on': ['AO', 'N'],
            'are': ['AA', 'R'],
            'as': ['AE', 'Z'],
            'with': ['W', 'IH', 'TH'],
            'his': ['HH', 'IH', 'Z'],
            'they': ['DH', 'EY'],
            'i': ['AY'],
            'at': ['AE', 'T'],
            'be': ['B', 'IY'],
            'this': ['DH', 'IH', 'S'],
            'have': ['HH', 'AE', 'V'],
            'from': ['F', 'R', 'AH', 'M'],
            'or': ['ER'],
            'one': ['W', 'AH', 'N'],
            'had': ['HH', 'AE', 'D'],
            'by': ['B', 'AY'],
            'word': ['W', 'ER', 'D'],
            'but': ['B', 'AH', 'T'],
            'not': ['N', 'AA', 'T'],
            'what': ['W', 'AH', 'T'],
            'all': ['AO', 'L'],
            'were': ['W', 'ER'],
            'we': ['W', 'IY'],
            'when': ['W', 'EH', 'N'],
            'your': ['Y', 'ER'],
            'can': ['K', 'AE', 'N'],
            'said': ['S', 'EH', 'D'],
            'there': ['DH', 'EH', 'R'],
            'each': ['IY', 'CH'],
            'which': ['W', 'IH', 'CH'],
            'do': ['D', 'UW'],
            'how': ['HH', 'AW'],
            'their': ['DH', 'EH', 'R'],
            'if': ['IH', 'F'],
            'will': ['W', 'IH', 'L'],
            'up': ['AH', 'P'],
            'other': ['AH', 'DH', 'ER'],
            'about': ['AH', 'B', 'AW', 'T'],
            'out': ['AW', 'T'],
            'many': ['M', 'EH', 'N', 'IY'],
            'then': ['DH', 'EH', 'N'],
            'them': ['DH', 'EH', 'M'],
            'these': ['DH', 'IY', 'Z'],
            'so': ['S', 'OW'],
            'some': ['S', 'AH', 'M'],
            'her': ['HH', 'ER'],
            'would': ['W', 'UH', 'D'],
            'make': ['M', 'EY', 'K'],
            'like': ['L', 'AY', 'K'],
            'into': ['IH', 'N', 'T', 'UW'],
            'him': ['HH', 'IH', 'M'],
            'time': ['T', 'AY', 'M'],
            'two': ['T', 'UW'],
            'more': ['M', 'ER'],
            'go': ['G', 'OW'],
            'no': ['N', 'OW'],
            'way': ['W', 'EY'],
            'could': ['K', 'UH', 'D'],
            'my': ['M', 'AY'],
            'than': ['DH', 'AE', 'N'],
            'first': ['F', 'ER', 'S', 'T'],
            'been': ['B', 'IH', 'N'],
            'call': ['K', 'AO', 'L'],
            'who': ['HH', 'UW'],
            'its': ['IH', 'T', 'S'],
            'now': ['N', 'AW'],
            'find': ['F', 'AY', 'N', 'D'],
            'long': ['L', 'AO', 'NG'],
            'down': ['D', 'AW', 'N'],
            'day': ['D', 'EY'],
            'did': ['D', 'IH', 'D'],
            'get': ['G', 'EH', 'T'],
            'come': ['K', 'AH', 'M'],
            'made': ['M', 'EY', 'D'],
            'may': ['M', 'EY'],
            'part': ['P', 'AA', 'R', 'T'],
        }
        
        return basic_dict
    
    def _grapheme_to_phoneme_fallback(self, word: str) -> List[str]:
        """
        Very basic fallback G2P using simple letter-to-sound rules.
        This is extremely simplified and only for POC purposes.
        """
        phonemes = []
        
        # Basic consonant mappings
        consonant_map = {
            'b': 'B', 'c': 'K', 'd': 'D', 'f': 'F', 'g': 'G', 'h': 'HH',
            'j': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'p': 'P',
            'q': 'K', 'r': 'R', 's': 'S', 't': 'T', 'v': 'V', 'w': 'W',
            'x': 'K', 'y': 'Y', 'z': 'Z'
        }
        
        # Basic vowel mappings (very simplified)
        vowel_map = {
            'a': 'AE', 'e': 'EH', 'i': 'IH', 'o': 'AO', 'u': 'UH'
        }
        
        for char in word.lower():
            if char in consonant_map:
                phonemes.append(consonant_map[char])
            elif char in vowel_map:
                phonemes.append(vowel_map[char])
            # Skip unknown characters
            
        return phonemes if phonemes else ['UNK']
    
    def convert(self, text: str) -> List[str]:
        """
        Convert text to phonemes.
        
        Args:
            text: Input text
            
        Returns:
            List of phonemes
        """
        # Normalize text
        text = normalize_text(text)
        
        # Split into words
        words = text.split()
        
        phonemes = []
        for word in words:
            # Remove punctuation
            clean_word = word.strip(string.punctuation)
            
            if clean_word in self.word_to_phonemes:
                phonemes.extend(self.word_to_phonemes[clean_word])
            else:
                # Fallback to basic G2P
                phonemes.extend(self._grapheme_to_phoneme_fallback(clean_word))
            
            # Add pause between words
            phonemes.append('SP')
        
        # Remove final pause and add silence
        if phonemes and phonemes[-1] == 'SP':
            phonemes = phonemes[:-1]
        
        # Add silence at beginning and end
        phonemes = ['SIL'] + phonemes + ['SIL']
        
        return phonemes


class TextProcessor:
    """Text processing pipeline for TTS."""
    
    def __init__(self, vocab_size: int = 256):
        """
        Initialize text processor.
        
        Args:
            vocab_size: Vocabulary size for padding
        """
        self.vocab_size = vocab_size
        self.g2p = SimpleG2P()
        self.phoneme_to_id = PHONEME_TO_ID
        self.id_to_phoneme = ID_TO_PHONEME
        
        logger.info(f"TextProcessor initialized with {len(PHONEME_SET)} phonemes")
    
    def text_to_phonemes(self, text: str) -> List[str]:
        """Convert text to phoneme sequence."""
        return self.g2p.convert(text)
    
    def phonemes_to_ids(self, phonemes: List[str]) -> List[int]:
        """Convert phonemes to integer IDs."""
        return [self.phoneme_to_id.get(phoneme, self.phoneme_to_id['UNK']) for phoneme in phonemes]
    
    def ids_to_phonemes(self, ids: List[int]) -> List[str]:
        """Convert integer IDs back to phonemes."""
        return [self.id_to_phoneme.get(id, 'UNK') for id in ids]
    
    def process_text(self, text: str, max_length: Optional[int] = None) -> Dict:
        """
        Complete text processing pipeline.
        
        Args:
            text: Input text
            max_length: Maximum sequence length for padding/truncation
            
        Returns:
            Dictionary with processed text information
        """
        # Convert to phonemes
        phonemes = self.text_to_phonemes(text)
        
        # Convert to IDs
        phoneme_ids = self.phonemes_to_ids(phonemes)
        
        # Apply length constraints
        if max_length is not None:
            if len(phoneme_ids) > max_length:
                phoneme_ids = phoneme_ids[:max_length]
                phonemes = phonemes[:max_length]
            else:
                # Pad with silence
                pad_length = max_length - len(phoneme_ids)
                phoneme_ids.extend([self.phoneme_to_id['SIL']] * pad_length)
                phonemes.extend(['SIL'] * pad_length)
        
        return {
            'text': text,
            'phonemes': phonemes,
            'phoneme_ids': phoneme_ids,
            'length': len([p for p in phonemes if p != 'SIL'])  # Exclude padding
        }


def create_phoneme_dict_file(output_path: Path) -> None:
    """Create phoneme dictionary file for reference."""
    with open(output_path, 'w') as f:
        for i, phoneme in enumerate(PHONEME_SET):
            f.write(f"{phoneme}\t{i}\n")
    
    logger.info(f"Created phoneme dictionary at {output_path}")