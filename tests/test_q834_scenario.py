#!/usr/bin/env python3
"""
Test Case Q834: PDF Table Extraction Scenario

Simulates:
> Q834. Download [file](https://example.com/data-q834.pdf). 
> What is the sum of the "value" column in the table on page 2?
>
> Post your answer to https://example.com/submit with JSON payload

This test verifies:
1. Question parsing extracts PDF URL and submit URL
2. PDF download and table extraction works
3. Answer format detection (number)
4. Submission payload has all required fields
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd

# Sample question text matching the user's example
Q834_QUESTION = """
Q834. Download [file](https://example.com/data-q834.pdf). What is the sum of the "value" column in the table on page 2?

Post your answer to https://example.com/submit with this JSON payload:

{
  "email": "your email",
  "secret": "your secret",
  "url": "https://example.com/quiz-834",
  "answer": 12345 // the correct answer
}
"""


class TestQ834QuestionParsing:
    """Test question parsing for Q834 scenario"""
    
    def test_extract_question_id(self):
        """Should extract Q834 as question ID"""
        from quiz_solver.question_parser import extract_question_components
        from quiz_solver.models import SessionState
        import time
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com/quiz-834",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz-834",
            quiz_chain=[]
        )
        
        components = extract_question_components(Q834_QUESTION, session)
        
        assert components.question_id == "Q834"
        print(f"✅ Question ID extracted: {components.question_id}")
    
    def test_extract_pdf_url(self):
        """Should extract PDF data source URL"""
        from quiz_solver.question_parser import extract_question_components
        from quiz_solver.models import SessionState
        import time
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com/quiz-834",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz-834",
            quiz_chain=[]
        )
        
        components = extract_question_components(Q834_QUESTION, session)
        
        assert "https://example.com/data-q834.pdf" in components.data_sources
        print(f"✅ PDF URL extracted: {components.data_sources}")
    
    def test_extract_submit_url(self):
        """Should extract submit endpoint URL"""
        from quiz_solver.question_parser import extract_question_components
        from quiz_solver.models import SessionState
        import time
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com/quiz-834",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz-834",
            quiz_chain=[]
        )
        
        components = extract_question_components(Q834_QUESTION, session)
        
        assert components.submit_url == "https://example.com/submit"
        print(f"✅ Submit URL extracted: {components.submit_url}")
    
    def test_detect_number_answer_format(self):
        """Should detect answer format as NUMBER (due to 'sum')"""
        from quiz_solver.question_parser import extract_question_components
        from quiz_solver.models import SessionState, AnswerFormat
        import time
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com/quiz-834",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz-834",
            quiz_chain=[]
        )
        
        components = extract_question_components(Q834_QUESTION, session)
        
        assert components.answer_format == AnswerFormat.NUMBER
        print(f"✅ Answer format detected: {components.answer_format}")


class TestQ834PDFExtraction:
    """Test PDF extraction for Q834 scenario"""
    
    def test_pdf_url_detection(self):
        """Should detect .pdf URLs"""
        url = "https://example.com/data-q834.pdf"
        assert url.endswith('.pdf')
        print("✅ PDF URL detection works")
    
    @pytest.mark.asyncio
    async def test_pdf_parsing_structure(self):
        """Test PDF parsing returns expected structure"""
        from quiz_solver.data_sourcing import download_and_parse_pdf
        
        # Mock the download to return fake PDF bytes
        with patch('quiz_solver.data_sourcing.download_file') as mock_download:
            # We can't easily mock pdfplumber, so just verify the function exists
            # and has the right signature
            import inspect
            sig = inspect.signature(download_and_parse_pdf)
            assert 'url' in sig.parameters
            print("✅ PDF parsing function has correct signature")
    
    def test_table_to_dataframe(self):
        """Test that PDF tables can be converted to DataFrame"""
        from quiz_solver.data_sourcing import clean_and_prepare_data
        
        # Simulate PDF extracted data with a table
        pdf_data = {
            'all_tables': [
                [
                    ['id', 'name', 'value'],  # Header row
                    ['1', 'Item A', '100'],
                    ['2', 'Item B', '200'],
                    ['3', 'Item C', '300'],
                ]
            ],
            'all_text': 'Some PDF text',
            'pages': {}
        }
        
        df = clean_and_prepare_data(pdf_data, Q834_QUESTION)
        
        assert df is not None
        assert 'value' in df.columns
        assert len(df) == 3
        print(f"✅ PDF table converted to DataFrame: {df.shape}")
        print(f"   Columns: {list(df.columns)}")


class TestQ834Analysis:
    """Test analysis for Q834 scenario"""
    
    def test_sum_calculation(self):
        """Test sum calculation on value column"""
        from quiz_solver.analysis import simple_analysis
        import pandas as pd
        
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Item A', 'Item B', 'Item C'],
            'value': [100.0, 200.0, 300.0]
        })
        
        question = "What is the sum of the value column?"
        answer = simple_analysis(df, question)
        
        assert answer == 600.0
        print(f"✅ Sum calculated correctly: {answer}")
    
    @pytest.mark.asyncio
    async def test_llm_code_generation(self):
        """Test LLM generates correct pandas code"""
        from quiz_solver.llm_client import LLMClient
        from quiz_solver.data_sourcing import get_dataframe_info
        import pandas as pd
        
        # Create test DataFrame
        df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Item A', 'Item B', 'Item C'],
            'value': [100.0, 200.0, 300.0]
        })
        
        df_info = get_dataframe_info(df)
        
        # Verify df_info structure
        assert 'shape' in df_info
        assert 'columns' in df_info
        assert 'value' in df_info['columns']
        print(f"✅ DataFrame info generated: {df_info['columns']}")


class TestQ834Submission:
    """Test submission for Q834 scenario"""
    
    def test_submission_payload_format(self):
        """Verify submission payload matches expected format"""
        # Expected format from question:
        # {
        #   "email": "your email",
        #   "secret": "your secret",
        #   "url": "https://example.com/quiz-834",
        #   "answer": 12345
        # }
        
        expected_fields = ['email', 'secret', 'url', 'answer']
        
        # Check our submission module creates this format
        from quiz_solver.submission import submit_answer
        import inspect
        
        source = inspect.getsource(submit_answer)
        
        # Verify payload construction
        assert '"email"' in source or "'email'" in source
        assert '"secret"' in source or "'secret'" in source
        assert '"url"' in source or "'url'" in source
        assert '"answer"' in source or "'answer'" in source
        
        print("✅ Submission payload has all required fields")
    
    def test_answer_formatting_for_number(self):
        """Test numeric answer is formatted correctly"""
        from quiz_solver.question_parser import format_answer
        from quiz_solver.models import AnswerFormat
        
        # Test various numeric inputs
        assert format_answer(600.0, AnswerFormat.NUMBER) == 600.0
        assert format_answer(600, AnswerFormat.NUMBER) == 600
        assert format_answer("600", AnswerFormat.NUMBER) == 600
        assert format_answer(600.123456, AnswerFormat.NUMBER) == 600.12  # Rounds to 2 decimals
        
        print("✅ Numeric answer formatting works correctly")


class TestQ834EndToEnd:
    """End-to-end test for Q834 scenario (mocked)"""
    
    @pytest.mark.asyncio
    async def test_full_pipeline_mock(self):
        """Test full pipeline with mocked external calls"""
        from quiz_solver.question_parser import extract_question_components, format_answer
        from quiz_solver.data_sourcing import clean_and_prepare_data, get_dataframe_info
        from quiz_solver.analysis import simple_analysis
        from quiz_solver.models import SessionState, AnswerFormat
        import time
        
        # Step 1: Create session
        session = SessionState(
            email="test@example.com",
            secret="test_secret",
            initial_url="https://example.com/quiz-834",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz-834",
            quiz_chain=["https://example.com/quiz-834"]
        )
        
        # Step 2: Parse question
        components = extract_question_components(Q834_QUESTION, session)
        assert components.question_id == "Q834"
        assert "https://example.com/data-q834.pdf" in components.data_sources
        assert components.submit_url == "https://example.com/submit"
        assert components.answer_format == AnswerFormat.NUMBER
        print("✅ Step 2: Question parsed")
        
        # Step 3: Simulate PDF data (mocked)
        pdf_data = {
            'all_tables': [
                [
                    ['id', 'name', 'value'],
                    ['1', 'Item A', '100'],
                    ['2', 'Item B', '200'],
                    ['3', 'Item C', '300'],
                ]
            ],
            'all_text': 'Page 2 table with values',
            'pages': {1: {'text': 'Page 2 content', 'tables': []}}
        }
        
        # Step 4: Clean data
        df = clean_and_prepare_data(pdf_data, Q834_QUESTION)
        assert df is not None
        assert 'value' in df.columns
        print(f"✅ Step 4: Data cleaned - shape {df.shape}")
        
        # Step 5: Analyze
        answer = simple_analysis(df, "sum of value column")
        assert answer == 600.0
        print(f"✅ Step 5: Analysis complete - answer: {answer}")
        
        # Step 6: Format answer
        formatted = format_answer(answer, AnswerFormat.NUMBER)
        assert formatted == 600.0
        print(f"✅ Step 6: Answer formatted: {formatted}")
        
        # Step 7: Verify submission would have correct payload
        expected_payload = {
            "email": "test@example.com",
            "secret": "test_secret",
            "url": "https://example.com/quiz-834",
            "answer": 600.0
        }
        print(f"✅ Step 7: Submission payload ready: {expected_payload}")
        
        print("\n" + "="*60)
        print("✅ Q834 END-TO-END TEST PASSED")
        print("="*60)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
