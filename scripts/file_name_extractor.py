import re
import os
from logger_setup import get_logger

logger = get_logger()

def extract_filename_from_input(user_input):
    """
    Extract filename(s) from user input using various patterns.
    Returns a single filename, list of filenames, or None if no valid filenames found.
    Supports multiple filenames for compatibility with db_connection filtering.
    Enhanced to handle complex filenames with special characters, versions, and parentheses.
    """
    try:
        user_input = user_input.strip()
        found_filenames = []
        
        # Common file extensions
        extensions = r'(?:pdf|xlsx?|txt|docx?|jpe?g|png|csv|pptx?|zip|rar|7z)'
        
        # Pattern 1: Enhanced filename pattern with better character support
        # Handles: letters, numbers, spaces, hyphens, underscores, parentheses, brackets, version numbers
        filename_pattern = rf'\b([a-zA-Z0-9_\-\s\(\)\[\]\.v]+\.{extensions})\b'
        matches = re.findall(filename_pattern, user_input, re.IGNORECASE)
        for match in matches:
            filename = ' '.join(match.split()).strip()  # Normalize whitespace
            if _is_valid_filename(filename):
                logger.debug(f"Found filename using enhanced pattern matching: {filename}")
                found_filenames.append(filename)
        
        # Pattern 2: Quoted filenames (more permissive)
        quoted_pattern = rf'["\']([^"\']*\.{extensions})["\']'
        quoted_matches = re.findall(quoted_pattern, user_input, re.IGNORECASE)
        for match in quoted_matches:
            filename = ' '.join(match.split()).strip()
            if _is_valid_filename(filename):
                logger.debug(f"Found filename in quotes: {filename}")
                found_filenames.append(filename)
        
        # Pattern 3: File-related keywords followed by filename (enhanced)
        keyword_pattern = rf'\b(?:file|document|report|data|sheet|pdf|excel|spreadsheet|presentation)\s+([a-zA-Z0-9_\-\s\(\)\[\]\.v]+\.{extensions})\b'
        keyword_matches = re.findall(keyword_pattern, user_input, re.IGNORECASE)
        for match in keyword_matches:
            filename = ' '.join(match.split()).strip()
            if _is_valid_filename(filename):
                logger.debug(f"Found filename using keyword pattern: {filename}")
                found_filenames.append(filename)
        
        # Pattern 4: Research Summary pattern (enhanced)
        research_summary_match = re.search(rf'Research Summary\s*:\s*([^\n]+\.{extensions})', user_input, re.IGNORECASE)
        if research_summary_match:
            filename = ' '.join(research_summary_match.group(1).split()).strip()
            if _is_valid_filename(filename):
                logger.debug(f"Found filename in Research Summary: {filename}")
                found_filenames.append(filename)
        
        # Pattern 5: Generic "for [filename]" pattern (enhanced)
        for_pattern = rf'\bfor\s+([a-zA-Z0-9_\-\s\(\)\[\]\.v]+\.{extensions})\b'
        for_matches = re.findall(for_pattern, user_input, re.IGNORECASE)
        for match in for_matches:
            filename = ' '.join(match.split()).strip()
            if _is_valid_filename(filename):
                logger.debug(f"Found filename using 'for' pattern: {filename}")
                found_filenames.append(filename)
        
        # Pattern 6: Filename at end of sentence or input
        end_pattern = rf'([a-zA-Z0-9_\-\s\(\)\[\]\.v]+\.{extensions})(?:\s*[.!?]?\s*$)'
        end_matches = re.findall(end_pattern, user_input, re.IGNORECASE)
        for match in end_matches:
            filename = ' '.join(match.split()).strip()
            if self._is_valid_filename(filename):
                logger.debug(f"Found filename at end of input: {filename}")
                found_filenames.append(filename)
        
        # Pattern 7: Standalone filename detection (most permissive)
        # This catches filenames that might be mentioned casually in text
        standalone_pattern = rf'\b([A-Z][a-zA-Z0-9_\-\s\(\)\[\]\.v]*\.{extensions})\b'
        standalone_matches = re.findall(standalone_pattern, user_input)
        for match in standalone_matches:
            filename = ' '.join(match.split()).strip()
            if _is_valid_filename(filename) and len(filename.split()) >= 2:  # At least 2 words for standalone
                logger.debug(f"Found standalone filename: {filename}")
                found_filenames.append(filename)
        
        # Clean and deduplicate filenames
        cleaned_filenames = []
        seen = set()
        
        for filename in found_filenames:
            # Additional cleaning
            cleaned = _clean_filename(filename)
            if cleaned and cleaned.lower() not in seen and _is_valid_filename(cleaned):
                seen.add(cleaned.lower())
                cleaned_filenames.append(cleaned)
        
        if not cleaned_filenames:
            logger.debug("No filenames found in user input")
            return None
        elif len(cleaned_filenames) == 1:
            return cleaned_filenames[0]
        else:
            logger.debug(f"Found multiple filenames: {cleaned_filenames}")
            return cleaned_filenames
        
    except Exception as e:
        logger.error(f"Error extracting filename from input: {str(e)}")
        return None

def _clean_filename(filename):
    """Clean and normalize filename"""
    if not filename:
        return None
    
    # Remove leading/trailing whitespace and normalize internal whitespace
    cleaned = ' '.join(filename.split())
    
    # Remove common prefixes that might get picked up
    prefixes_to_remove = ['the ', 'a ', 'an ']
    for prefix in prefixes_to_remove:
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):]
    
    # Ensure filename has at least one alphanumeric character before the extension
    name_part = os.path.splitext(cleaned)[0]
    if not re.search(r'[a-zA-Z0-9]', name_part):
        return None
    
    return cleaned.strip()

def _is_valid_filename(filename):
    """Validate if the extracted string is likely a real filename"""
    if not filename or len(filename.strip()) < 5:  # Minimum reasonable filename length
        return False
    
    # Must have a valid extension
    name, ext = os.path.splitext(filename)
    if not ext or len(ext) < 2:
        return False
    
    # Must have some alphanumeric content
    if not re.search(r'[a-zA-Z0-9]', name):
        return False
    
    # Should not be mostly special characters
    alphanumeric_count = len(re.findall(r'[a-zA-Z0-9]', name))
    total_count = len(name)
    if total_count > 0 and (alphanumeric_count / total_count) < 0.3:
        return False
    
    # Common false positives to exclude
    false_positives = {
        'file.pdf', 'document.pdf', 'report.pdf', 'data.xlsx', 
        'sheet.xlsx', 'image.jpg', 'photo.png', 'text.txt'
    }
    if filename.lower() in false_positives:
        return False
    
    return True
