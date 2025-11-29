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
    
    # Infer answer format from question
    if any(word in question_text.lower() for word in ['sum', 'total', 'average', 'count', 'percentage', 'how many', 'what is the']):
        components.answer_format = AnswerFormat.NUMBER
    elif any(word in question_text.lower() for word in ['true', 'false', 'yes', 'no', 'is it']):
        components.answer_format = AnswerFormat.BOOLEAN
    elif any(word in question_text.lower() for word in ['chart', 'image', 'graph', 'plot', 'visualization']):
        components.answer_format = AnswerFormat.BASE64_IMAGE
    elif 'json' in question_text.lower():
        components.answer_format = AnswerFormat.JSON
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
    """Format answer according to expected format."""
    
    if answer is None:
        return None
    
    try:
        if expected_format == AnswerFormat.NUMBER:
            # Try to convert to number
            if isinstance(answer, (int, float)):
                # Round to reasonable precision
                if isinstance(answer, float):
                    return round(answer, 2)
                return answer
            else:
                # Try to parse string
                cleaned = str(answer).strip().replace(',', '')
                if '.' in cleaned:
                    return round(float(cleaned), 2)
                return int(cleaned)
        
        elif expected_format == AnswerFormat.BOOLEAN:
            if isinstance(answer, bool):
                return answer
            str_answer = str(answer).lower().strip()
            return str_answer in ['true', 'yes', '1', 'correct']
        
        elif expected_format == AnswerFormat.JSON:
            import json
            if isinstance(answer, str):
                return json.loads(answer)
            return answer
        
        elif expected_format == AnswerFormat.BASE64_IMAGE:
            # Return as-is if already base64
            return str(answer)
        
        else:  # STRING
            return str(answer).strip()
    
    except Exception as e:
        logger.warning(f"Failed to format answer: {e}")
        return str(answer)
