#!/usr/bin/env python3
"""
Direct test of the quiz solver pipeline with the demo endpoint.
Bypasses the API layer for direct testing.
"""

import asyncio
import time
from quiz_solver.pipeline import solve_quiz_pipeline
from quiz_solver.logging_utils import logger


async def main():
    """Run the quiz solver directly."""
    
    email = "24f100161@ds.study.iitm.ac.in"
    secret = "iamdivyam"
    initial_url = "https://tds-llm-analysis.s-anand.net/demo"
    
    print("=" * 60)
    print("🧪 Direct Quiz Solver Test - Demo Endpoint")
    print("=" * 60)
    print(f"Email: {email}")
    print(f"URL: {initial_url}")
    print(f"Deadline: 180 seconds")
    print("=" * 60)
    print()
    
    start_time = time.time()
    deadline = start_time + 180  # 3 minutes
    
    try:
        result = await solve_quiz_pipeline(
            email=email,
            secret=secret,
            initial_url=initial_url,
            deadline=deadline
        )
        
        elapsed = time.time() - start_time
        
        print()
        print("=" * 60)
        print("📊 RESULT")
        print("=" * 60)
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Success: {result.get('success', False)}")
        print(f"Quizzes solved: {result.get('quizzes_solved', 0)}")
        print(f"Final answer: {result.get('final_answer', 'N/A')}")
        
        if result.get('error'):
            print(f"Error: {result.get('error')}")
        
        if result.get('submissions'):
            print(f"Submissions: {len(result.get('submissions', []))}")
            for i, sub in enumerate(result.get('submissions', []), 1):
                print(f"  {i}. {sub}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        print(f"\n❌ Pipeline failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
