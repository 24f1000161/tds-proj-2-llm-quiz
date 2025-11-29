#!/usr/bin/env python3
"""
Test the exact base64-encoded quiz format from the real quiz pages.
"""

import base64
from quiz_solver.browser import decode_content_multi_layer
from quiz_solver.question_parser import extract_question_components
from quiz_solver.models import SessionState, AnswerFormat
import time

# The exact HTML from the quiz page
QUIZ_HTML = '''<div id="result"></div>

<script>
  document.querySelector("#result").innerHTML = atob(`
UTgzNC4gRG93bmxvYWQgPGEgaHJlZj0iaHR0cHM6Ly9leGFtcGxlLmNvbS9kYXRhLXE4MzQucGRmIj5
maWxlPC9hPi4KV2hhdCBpcyB0aGUgc3VtIG9mIHRoZSAidmFsdWUiIGNvbHVtbiBpbiB0aGUgdGFibG
Ugb24gcGFnZSAyPwoKUG9zdCB5b3VyIGFuc3dlciB0byBodHRwczovL2V4YW1wbGUuY29tL3N1Ym1pd
CB3aXRoIHRoaXMgSlNPTiBwYXlsb2FkOgoKPHByZT4KewogICJlbWFpbCI6ICJ5b3VyLWVtYWlsIiwK
ICAic2VjcmV0IjogInlvdXIgc2VjcmV0IiwKICAidXJsIjogImh0dHBzOi8vZXhhbXBsZS5jb20vcXV
pei04MzQiLAogICJhbnN3ZXIiOiAxMjM0NSAgLy8gdGhlIGNvcnJlY3QgYW5zd2VyCn0KPC9wcmU+`);
</script>
'''

# The base64 content extracted from the script
BASE64_CONTENT = """UTgzNC4gRG93bmxvYWQgPGEgaHJlZj0iaHR0cHM6Ly9leGFtcGxlLmNvbS9kYXRhLXE4MzQucGRmIj5maWxlPC9hPi4KV2hhdCBpcyB0aGUgc3VtIG9mIHRoZSAidmFsdWUiIGNvbHVtbiBpbiB0aGUgdGFibGUgb24gcGFnZSAyPwoKUG9zdCB5b3VyIGFuc3dlciB0byBodHRwczovL2V4YW1wbGUuY29tL3N1Ym1pdCB3aXRoIHRoaXMgSlNPTiBwYXlsb2FkOgoKPHByZT4KewogICJlbWFpbCI6ICJ5b3VyLWVtYWlsIiwKICAic2VjcmV0IjogInlvdXIgc2VjcmV0IiwKICAidXJsIjogImh0dHBzOi8vZXhhbXBsZS5jb20vcXVpei04MzQiLAogICJhbnN3ZXIiOiAxMjM0NSAgLy8gdGhlIGNvcnJlY3QgYW5zd2VyCn0KPC9wcmU+"""

def test_base64_decoding():
    """Test that we can decode the base64 content"""
    print("=" * 60)
    print("TEST 1: Base64 Decoding")
    print("=" * 60)
    
    decoded = base64.b64decode(BASE64_CONTENT).decode('utf-8')
    print(f"Decoded content:\n{decoded}")
    
    assert "Q834" in decoded
    assert "https://example.com/data-q834.pdf" in decoded
    assert "https://example.com/submit" in decoded
    
    print("\n✅ Base64 decoding works correctly")
    return decoded


def test_decode_content_multi_layer():
    """Test our multi-layer decode function"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-layer Decode Function")
    print("=" * 60)
    
    # Test with base64 content
    decoded, layers = decode_content_multi_layer(BASE64_CONTENT)
    print(f"Layers decoded: {layers}")
    print(f"Content preview: {decoded[:200]}...")
    
    assert layers >= 1, "Should decode at least 1 layer"
    assert "Q834" in decoded
    
    print("\n✅ Multi-layer decode works")
    return decoded


def test_question_parsing():
    """Test question parsing with decoded content"""
    print("\n" + "=" * 60)
    print("TEST 3: Question Parsing")
    print("=" * 60)
    
    # Decode the content first
    decoded = base64.b64decode(BASE64_CONTENT).decode('utf-8')
    
    # Create session
    session = SessionState(
        email="test@example.com",
        secret="test_secret",
        initial_url="https://example.com/quiz-834",
        start_time=time.time(),
        deadline=time.time() + 180,
        current_url="https://example.com/quiz-834",
        quiz_chain=[]
    )
    
    # Parse question
    components = extract_question_components(decoded, session)
    
    print(f"Question ID: {components.question_id}")
    print(f"Data sources: {components.data_sources}")
    print(f"Submit URL: {components.submit_url}")
    print(f"Answer format: {components.answer_format}")
    
    assert components.question_id == "Q834", f"Expected Q834, got {components.question_id}"
    assert "https://example.com/data-q834.pdf" in components.data_sources, f"PDF not found in {components.data_sources}"
    assert components.submit_url == "https://example.com/submit", f"Wrong submit URL: {components.submit_url}"
    assert components.answer_format == AnswerFormat.NUMBER, f"Wrong format: {components.answer_format}"
    
    print("\n✅ Question parsing works correctly")
    return components


def test_full_flow():
    """Test complete flow from HTML to parsed question"""
    print("\n" + "=" * 60)
    print("TEST 4: Full Flow Simulation")
    print("=" * 60)
    
    # Simulate what Playwright would do:
    # 1. Load page with script
    # 2. Script executes atob() and sets innerHTML
    # 3. We extract from #result div
    
    # After JS execution, #result would contain:
    rendered_content = base64.b64decode(BASE64_CONTENT).decode('utf-8')
    print(f"Simulated #result innerHTML (first 200 chars):\n{rendered_content[:200]}...")
    
    # Our pipeline would then:
    # 1. Try decode_content_multi_layer (in case of additional encoding)
    decoded, layers = decode_content_multi_layer(rendered_content)
    print(f"\nAfter decode_content_multi_layer: {layers} additional layers")
    
    # 2. Parse the question
    session = SessionState(
        email="test@example.com",
        secret="test_secret",
        initial_url="https://example.com/quiz-834",
        start_time=time.time(),
        deadline=time.time() + 180,
        current_url="https://example.com/quiz-834",
        quiz_chain=[]
    )
    
    components = extract_question_components(decoded, session)
    
    print(f"\nExtracted components:")
    print(f"  - Question ID: {components.question_id}")
    print(f"  - PDF URL: {components.data_sources}")
    print(f"  - Submit URL: {components.submit_url}")
    print(f"  - Answer format: {components.answer_format}")
    
    # Verify everything
    assert components.question_id == "Q834"
    assert "https://example.com/data-q834.pdf" in components.data_sources
    assert components.submit_url == "https://example.com/submit"
    assert components.answer_format == AnswerFormat.NUMBER
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Ready to handle real quiz pages!")
    print("=" * 60)


if __name__ == "__main__":
    test_base64_decoding()
    test_decode_content_multi_layer()
    test_question_parsing()
    test_full_flow()
