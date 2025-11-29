"""
End-to-end test for Quiz 3 (demo-audio) using the pipeline directly.
"""

import asyncio
import time
import sys

# Add project to path
sys.path.insert(0, ".")

from quiz_solver.pipeline import solve_quiz_pipeline


async def test_quiz3_pipeline():
    """Test Quiz 3 directly through the pipeline"""
    
    email = "24f100161@ds.study.iitm.ac.in"
    secret = "iamdivyam"
    url = "https://tds-llm-analysis.s-anand.net/demo-audio?email=24f100161@ds.study.iitm.ac.in&id=33536"
    deadline = time.time() + 180  # 3 minutes
    
    print("=" * 70)
    print("Testing Quiz 3 Pipeline")
    print("=" * 70)
    
    await solve_quiz_pipeline(email, secret, url, deadline)
    
    print("\n" + "=" * 70)
    print("Test Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_quiz3_pipeline())
