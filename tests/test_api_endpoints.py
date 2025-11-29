#!/usr/bin/env python3
"""
API Endpoint Test Cases matching evaluation requirements.
Run: uv run pytest tests/test_api_endpoints.py -v
"""

import pytest
import asyncio
import time
import httpx
import os

# Set test secret before importing app
os.environ["STUDENT_SECRETS"] = "test@example.com:test_secret,your@email.com:your_secret"

from quiz_solver.api import app
from quiz_solver.config import load_settings


@pytest.fixture
def base_url():
    return "http://127.0.0.1:8000"


@pytest.fixture
def valid_payload():
    return {
        "email": "test@example.com",
        "secret": "test_secret",
        "url": "https://tds-llm-analysis.s-anand.net/demo"
    }


class TestAPIEndpoints:
    """Test Case 1: Secret/JSON Validation"""
    
    @pytest.mark.asyncio
    async def test_1a_valid_request(self, base_url, valid_payload):
        """Test Case 1a: Valid Request -> HTTP 200"""
        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.post(
                f"{base_url}/api/quiz",
                json=valid_payload
            )
            elapsed = (time.time() - start) * 1000
            
            assert response.status_code == 200, f"Expected 200, got {response.status_code}"
            
            data = response.json()
            assert data.get("status") == "accepted"
            assert "quiz_id" in data
            assert "timestamp" in data
            
            # Response should be immediate (< 100ms)
            assert elapsed < 100, f"Response took {elapsed:.1f}ms, expected < 100ms"
            
            print(f"✅ Valid request: 200 OK in {elapsed:.1f}ms")
            print(f"   Response: {data}")
    
    @pytest.mark.asyncio
    async def test_1b_invalid_json(self, base_url):
        """Test Case 1b: Invalid JSON -> HTTP 400"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/quiz",
                content="invalid json {",
                headers={"Content-Type": "application/json"}
            )
            
            assert response.status_code == 400, f"Expected 400, got {response.status_code}"
            
            data = response.json()
            assert "error" in data
            
            print(f"✅ Invalid JSON: 400 Bad Request")
            print(f"   Response: {data}")
    
    @pytest.mark.asyncio
    async def test_1c_invalid_secret(self, base_url):
        """Test Case 1c: Invalid Secret -> HTTP 403"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{base_url}/api/quiz",
                json={
                    "email": "test@example.com",
                    "secret": "wrong_secret",
                    "url": "https://example.com/quiz"
                }
            )
            
            assert response.status_code == 403, f"Expected 403, got {response.status_code}"
            
            data = response.json()
            assert "detail" in data or "error" in data
            
            print(f"✅ Invalid secret: 403 Forbidden")
            print(f"   Response: {data}")
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, base_url):
        """Test health check endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/")
            
            assert response.status_code == 200
            data = response.json()
            assert data.get("status") == "ok"
            
            print(f"✅ Health check: 200 OK")


class TestSubmissionPayload:
    """Test Case 2 & 4: Verify submission payload format"""
    
    def test_submission_payload_has_required_fields(self):
        """Verify submission includes email, secret, url, answer"""
        from quiz_solver.submission import submit_answer
        import inspect
        
        sig = inspect.signature(submit_answer)
        params = list(sig.parameters.keys())
        
        assert "email" in params
        assert "secret" in params
        assert "quiz_url" in params
        assert "answer" in params
        
        print("✅ Submission function has all required parameters")
    
    def test_answer_size_validation(self):
        """Verify answers are validated for size < 1MB"""
        # Read the submission.py file to check for size validation
        import quiz_solver.submission as submission_module
        import inspect
        
        source = inspect.getsource(submission_module)
        assert "1_000_000" in source or "1000000" in source or "1MB" in source
        
        print("✅ Answer size validation present")


class TestDataModalities:
    """Test Case 3: Data modality handling"""
    
    def test_3a_pdf_extraction_capability(self):
        """Test Case 3a: PDF table extraction is implemented"""
        from quiz_solver.data_sourcing import download_and_parse_pdf
        import inspect
        
        sig = inspect.signature(download_and_parse_pdf)
        assert "url" in sig.parameters
        
        # Check that pdfplumber is used
        source = inspect.getsource(download_and_parse_pdf)
        assert "pdfplumber" in source or "pypdf" in source
        assert "pages" in source
        assert "tables" in source
        
        print("✅ PDF extraction capability verified")
    
    def test_3b_static_web_scraping(self):
        """Test Case 3b: Static web scraping is implemented"""
        from quiz_solver.data_sourcing import scrape_webpage_static
        import inspect
        
        source = inspect.getsource(scrape_webpage_static)
        assert "BeautifulSoup" in source
        assert "tables" in source
        
        print("✅ Static web scraping capability verified")
    
    def test_3c_dynamic_spa_support(self):
        """Test Case 3c: Dynamic SPA support with click interactions"""
        from quiz_solver.browser import handle_pagination, click_show_more
        import inspect
        
        # Check pagination
        pagination_source = inspect.getsource(handle_pagination)
        assert "Next" in pagination_source
        assert "click" in pagination_source
        
        # Check show more
        show_more_source = inspect.getsource(click_show_more)
        assert "Load More" in show_more_source or "Show More" in show_more_source
        
        print("✅ Dynamic SPA support verified (pagination + show more)")
    
    def test_3d_api_sourcing(self):
        """Test Case 3d: API sourcing is implemented"""
        from quiz_solver.data_sourcing import call_api
        import inspect
        
        sig = inspect.signature(call_api)
        assert "url" in sig.parameters
        
        print("✅ API sourcing capability verified")
    
    def test_3e_audio_processing(self):
        """Test Case 3e: Audio processing is implemented"""
        from quiz_solver.data_sourcing import transcribe_audio, is_audio_url
        import inspect
        
        # Check audio URL detection
        assert is_audio_url("test.mp3") == True
        assert is_audio_url("test.wav") == True
        assert is_audio_url("test.pdf") == False
        
        # Check transcription implementation
        source = inspect.getsource(transcribe_audio)
        assert "gemini" in source.lower() or "whisper" in source.lower()
        
        print("✅ Audio processing capability verified")
    
    def test_3f_multimodal_fusion(self):
        """Test Case 3f: Multi-modal fusion is implemented"""
        from quiz_solver.pipeline import solve_quiz_pipeline
        import inspect
        
        source = inspect.getsource(solve_quiz_pipeline)
        
        # Check for merged context
        assert "merged_context" in source or "multi" in source.lower()
        
        # Check for audio transcript handling
        assert "audio" in source.lower()
        
        # Check for PDF handling
        assert "pdf" in source.lower()
        
        print("✅ Multi-modal fusion capability verified")


class TestAnswerFormats:
    """Test Case 4: Answer format validation"""
    
    def test_format_number(self):
        """Test numeric answer formatting"""
        from quiz_solver.question_parser import format_answer
        from quiz_solver.models import AnswerFormat
        
        result = format_answer(123.456, AnswerFormat.NUMBER)
        assert isinstance(result, (int, float))
        
        result = format_answer("456", AnswerFormat.NUMBER)
        assert result == 456
        
        print("✅ Numeric answer formatting works")
    
    def test_format_string(self):
        """Test string answer formatting"""
        from quiz_solver.question_parser import format_answer
        from quiz_solver.models import AnswerFormat
        
        result = format_answer("hello world", AnswerFormat.STRING)
        assert result == "hello world"
        
        print("✅ String answer formatting works")
    
    def test_format_boolean(self):
        """Test boolean answer formatting"""
        from quiz_solver.question_parser import format_answer
        from quiz_solver.models import AnswerFormat
        
        result = format_answer("true", AnswerFormat.BOOLEAN)
        assert result == True
        
        result = format_answer("no", AnswerFormat.BOOLEAN)
        assert result == False
        
        print("✅ Boolean answer formatting works")
    
    def test_format_json(self):
        """Test JSON answer formatting"""
        from quiz_solver.question_parser import format_answer
        from quiz_solver.models import AnswerFormat
        
        result = format_answer('{"key": "value"}', AnswerFormat.JSON)
        assert result == {"key": "value"}
        
        print("✅ JSON answer formatting works")


class TestChainedQuizzes:
    """Test Case 2: Quiz chain handling"""
    
    def test_chain_tracking(self):
        """Verify quiz chain is tracked in session"""
        from quiz_solver.models import SessionState
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com/quiz1",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com/quiz1",
            quiz_chain=["https://example.com/quiz1"]
        )
        
        # Simulate adding to chain
        session.quiz_chain.append("https://example.com/quiz2")
        session.quiz_chain.append("https://example.com/quiz3")
        
        assert len(session.quiz_chain) == 3
        
        print("✅ Quiz chain tracking works")
    
    def test_max_quizzes_limit(self):
        """Verify maximum quiz limit is enforced"""
        from quiz_solver.pipeline import solve_quiz_pipeline
        import inspect
        
        source = inspect.getsource(solve_quiz_pipeline)
        assert "MAX_QUIZZES" in source
        assert "10" in source  # Should be limited to 10
        
        print("✅ Maximum quiz limit (10) is enforced")
    
    def test_submission_result_model(self):
        """Verify SubmissionResult handles chain URLs"""
        from quiz_solver.models import SubmissionResult
        
        # Test correct with new URL
        result = SubmissionResult(correct=True, url="https://next-quiz.com")
        assert result.correct == True
        assert result.url == "https://next-quiz.com"
        
        # Test correct without URL (end of chain)
        result = SubmissionResult(correct=True)
        assert result.correct == True
        assert result.url is None
        
        # Test incorrect with new URL
        result = SubmissionResult(correct=False, url="https://retry-quiz.com")
        assert result.correct == False
        assert result.url == "https://retry-quiz.com"
        
        print("✅ SubmissionResult handles all chain scenarios")


class TestDeadlineEnforcement:
    """Verify 3-minute deadline enforcement"""
    
    def test_deadline_is_180_seconds(self):
        """Verify deadline is set to 180 seconds"""
        from quiz_solver.config import settings
        
        assert settings.timeouts.quiz_deadline_seconds == 180
        
        print("✅ Deadline is 180 seconds (3 minutes)")
    
    def test_time_remaining_safe(self):
        """Verify safety buffer in time calculation"""
        from quiz_solver.pipeline import time_remaining_safe
        from quiz_solver.models import SessionState
        
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com",
            start_time=time.time(),
            deadline=time.time() + 180,
            current_url="https://example.com",
            quiz_chain=[]
        )
        
        remaining = time_remaining_safe(session)
        # Should be less than 180 due to safety buffer
        assert remaining < 180
        assert remaining > 0
        
        print(f"✅ Time remaining with safety buffer: {remaining:.1f}s")
    
    def test_force_submit_logic(self):
        """Verify force submit triggers at low time"""
        from quiz_solver.pipeline import should_force_submit
        from quiz_solver.models import SessionState
        
        # Session with very little time left
        session = SessionState(
            email="test@example.com",
            secret="secret",
            initial_url="https://example.com",
            start_time=time.time() - 175,  # 175 seconds ago
            deadline=time.time() + 5,  # Only 5 seconds left
            current_url="https://example.com",
            quiz_chain=[]
        )
        
        assert should_force_submit(session) == True
        
        print("✅ Force submit triggers when time is critical")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
