"""
Question parsing and extraction utilities.
"""

import re
from typing import Any

from .models import QuestionComponents, AnswerFormat
from .logging_utils import logger, log_step


def extract_question_components(question_text: str, session: Any) -> QuestionComponents:
    """Parse question into structured components."""
    
    components = QuestionComponents()
    
    # Extract question ID (Q123 format)
    qid_match = re.search(r"Q(\d+)", question_text)
    if qid_match:
        components.question_id = f"Q{qid_match.group(1)}"
    
    # Extract absolute URLs (files and endpoints)
    url_pattern = r"https?://[^\s<>\"'\)\]]+(?:\.\w+)?"
    all_urls = re.findall(url_pattern, question_text)
    
    # Also extract relative URLs (like /path/to/page)
    # Look for patterns like "Scrape /path" or "href="/path""
    relative_url_patterns = [
        r'[Ss]crape\s+(/[^\s<>\"\']+)',  # "Scrape /path"
        r'href="(/[^"]+)"',  # href="/path"
        r'href=\'(/[^\']+)\'',  # href='/path'
        r'[Vv]isit\s+(/[^\s<>\"\']+)',  # "Visit /path"
        r'[Gg]et\s+(?:data\s+)?from\s+(/[^\s<>\"\']+)',  # "Get data from /path"
    ]
    
    for pattern in relative_url_patterns:
        matches = re.findall(pattern, question_text)
        for match in matches:
            if match not in components.relative_urls:
                components.relative_urls.append(match)
    
    # Categorize URLs
    for url in all_urls:
        if any(ext in url.lower() for ext in ['.pdf', '.csv', '.json', '.xlsx', '.txt']):
            components.data_sources.append(url)
        elif '/submit' in url.lower() or '/answer' in url.lower():
            components.submit_url = url
        else:
            components.data_sources.append(url)
    
    # Check for /submit in relative URLs (various patterns)
    if not components.submit_url:
        # Pattern: "POST ... to /submit" or "back to /submit"
        submit_patterns = [
            r'(?:POST|post)\s+.*?(?:to|back to)\s+(/submit[^\s]*)',
            r'(?:submit|Send)\s+.*?(?:to|back to)\s+(/submit[^\s]*)',
            r'(?:to|back to)\s+(/submit)\b',
            r'(/submit)\s',
        ]
        for pattern in submit_patterns:
            submit_match = re.search(pattern, question_text, re.IGNORECASE)
            if submit_match:
                components.relative_submit_url = submit_match.group(1)
                break
    
    # Extract special instructions
    instruction_keywords = ['page', 'column', 'sum', 'average', 'filter', 'where', 'group by', 'count', 'total']
    for keyword in instruction_keywords:
        pattern = rf"(?i){keyword}[\s\S]{{0,50}}"
        matches = re.findall(pattern, question_text)
        components.instructions.extend(matches[:2])  # Take first 2 matches
    
    # Infer answer format from question - be conservative and prefer STRING
    # Only use specific formats when we're very confident
    question_lower = question_text.lower()
    
    # Check for explicit number-only answers
    if any(phrase in question_lower for phrase in ['what is the sum', 'what is the total', 'what is the average', 'how many', 'what is the count']):
        components.answer_format = AnswerFormat.NUMBER
    # Check for explicit chart/image answers
    elif any(word in question_lower for word in ['chart', 'image', 'graph', 'plot', 'visualization', 'base64']):
        components.answer_format = AnswerFormat.BASE64_IMAGE
    # Check for explicit JSON format requirement
    elif 'json' in question_lower and 'answer' in question_lower:
        components.answer_format = AnswerFormat.JSON
    # Default to STRING - this is safest for command strings, codes, etc.
    else:
        components.answer_format = AnswerFormat.STRING
    
    # Extract description (first meaningful sentence)
    sentences = re.split(r'[.!?]', question_text)
    if sentences:
        components.question_description = sentences[0].strip()[:200]
    
    log_step(session, "question_parsed", {
        "question_id": components.question_id,
        "data_sources_count": len(components.data_sources),
        "has_submit_url": components.submit_url is not None,
        "inferred_format": components.answer_format.value,
        "relative_urls": components.relative_urls
    })
    
    return components


def format_answer(answer: Any, expected_format: AnswerFormat) -> Any:
    """Format answer according to expected format.
    
    IMPORTANT: For STRING format, we preserve the original answer from LLM.
    This is crucial for command strings, git commands, uv commands, etc.
    """
    
    if answer is None:
        return None
    
    # First, clean up the answer string if needed
    answer_str = str(answer).strip()
    
    # Remove markdown code block markers if present
    if answer_str.startswith('```'):
        lines = answer_str.split('\n')
        # Remove first line (```python or ```) and last line (```)
        if len(lines) > 2:
            answer_str = '\n'.join(lines[1:-1] if lines[-1].strip() == '```' else lines[1:])
        answer_str = answer_str.strip()
    
    try:
        if expected_format == AnswerFormat.NUMBER:
            # Try to convert to number
            if isinstance(answer, (int, float)):
                if isinstance(answer, float):
                    return round(answer, 2)
                return answer
            else:
                # Try to parse string - extract first number
                import re
                cleaned = answer_str.replace(',', '')
                # Try to find a number in the string
                number_match = re.search(r'-?\d+\.?\d*', cleaned)
                if number_match:
                    num_str = number_match.group()
                    if '.' in num_str:
                        return round(float(num_str), 2)
                    return int(num_str)
                return answer_str  # Return as-is if no number found
        
        elif expected_format == AnswerFormat.BOOLEAN:
            # Only convert to boolean if it's clearly a boolean answer
            if isinstance(answer, bool):
                return answer
            lower = answer_str.lower()
            if lower in ['true', 'false', 'yes', 'no']:
                return lower in ['true', 'yes']
            # Otherwise return as string - it's probably a command or code
            return answer_str
        
        elif expected_format == AnswerFormat.JSON:
            import json
            if isinstance(answer, str):
                try:
                    return json.loads(answer_str)
                except json.JSONDecodeError:
                    return answer_str
            return answer
        
        elif expected_format == AnswerFormat.BASE64_IMAGE:
            return answer_str
        
        else:  # STRING - preserve as-is
            return answer_str
    
    except Exception as e:
        logger.warning(f"Failed to format answer: {e}")
        return answer_str
