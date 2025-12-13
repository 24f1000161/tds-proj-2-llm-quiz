"""
Analysis execution module with sandboxed code execution.
"""

import sys
from io import StringIO
from typing import Any, Optional
import pandas as pd
import numpy as np

from .logging_utils import logger


async def execute_analysis_code(code: str, df: pd.DataFrame, context: dict = None) -> tuple[Any, str, Optional[str]]:
    """Execute generated pandas code safely.
    
    FIX #2: Added missing imports (requests, re, datetime).
    Now also accepts context dict to pass json_data/dict_data to code.
    """
    
    # FIX #2: Import additional modules that LLM code might need
    import requests
    import re
    import datetime
    import json
    
    # Create sandbox namespace
    namespace = {
        "df": df.copy(),
        "pd": pd,
        "np": np,
        "requests": requests,  # FIX #2
        "re": re,  # FIX #2
        "datetime": datetime,  # FIX #2
        "json": json,  # FIX #2
        "answer": None
    }
    
    # Add context data if available
    if context:
        if 'json_data' in context:
            namespace['json_data'] = context['json_data']
        if 'dict_data' in context:
            namespace['dict_data'] = context['dict_data']
    
    # Capture output
    old_stdout = sys.stdout
    sys.stdout = output_buffer = StringIO()
    
    try:
        exec(code, namespace)
        answer = namespace.get("answer")
        output = output_buffer.getvalue()
        
        logger.info(f"Analysis executed successfully, answer: {answer}")
        return answer, output, None
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Analysis execution failed: {error_msg}")
        return None, "", error_msg
    
    finally:
        sys.stdout = old_stdout


def simple_analysis(df: pd.DataFrame, question_text: str) -> Any:
    """Perform simple analysis based on question keywords."""
    import re
    
    question_lower = question_text.lower()
    
    try:
        # Check for cutoff-based filtering
        cutoff = None
        cutoff_match = re.search(r'cutoff[:\s]+(\d+)', question_text, re.IGNORECASE)
        if cutoff_match:
            cutoff = int(cutoff_match.group(1))
        else:
            # Also check for "CUTOFF VALUE" pattern
            cutoff_match = re.search(r'CUTOFF VALUE[^\d]*(\d+)', question_text)
            if cutoff_match:
                cutoff = int(cutoff_match.group(1))
        
        # Determine the filter operator from audio transcript or question
        use_gte = 'greater than or equal' in question_lower or '>=' in question_text
        
        # Sum operations with cutoff filtering
        if ('sum' in question_lower or 'add' in question_lower) and cutoff is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                col = numeric_cols[0]
                if use_gte:
                    filtered = df[df[col] >= cutoff][col]
                else:
                    filtered = df[df[col] > cutoff][col]
                return int(filtered.sum())
        
        # Sum operations (without cutoff)
        if 'sum' in question_lower or 'total' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # If there's a specific column mentioned
                for col in numeric_cols:
                    if col in question_lower:
                        return df[col].sum()
                # Otherwise sum the first numeric column
                return df[numeric_cols[0]].sum()
        
        # Count operations
        if 'count' in question_lower or 'how many' in question_lower:
            return len(df)
        
        # Average operations
        if 'average' in question_lower or 'mean' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].mean()
        
        # Max operations
        if 'max' in question_lower or 'maximum' in question_lower or 'highest' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].max()
        
        # Min operations
        if 'min' in question_lower or 'minimum' in question_lower or 'lowest' in question_lower:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols[0]].min()
        
        # Default: return first value
        return df.iloc[0, 0] if not df.empty else None
    
    except Exception as e:
        logger.error(f"Simple analysis failed: {e}")
        return None
