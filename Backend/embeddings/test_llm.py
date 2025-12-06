#!/usr/bin/env python3
"""
Test script to debug LLM answer generation
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def test_llm_script():
    """Test the llm_answer.py script with sample data"""
    
    print("=" * 70)
    print("LLM ANSWER SCRIPT DIAGNOSTIC TEST")
    print("=" * 70)
    
    # Check .env file
    env_path = Path('.env')
    print(f"\n1. Checking .env file...")
    if env_path.exists():
        print(f"   ✓ .env file found at: {env_path.absolute()}")
        with open(env_path, 'r') as f:
            lines = f.readlines()
            has_together = any('TOGETHER_API_KEY' in line and not line.strip().startswith('#') for line in lines)
            has_openrouter = any('OPENROUTER_API_KEY' in line and not line.strip().startswith('#') for line in lines)
            print(f"   TOGETHER_API_KEY present: {has_together}")
            print(f"   OPENROUTER_API_KEY present: {has_openrouter}")
    else:
        print(f"   ✗ .env file NOT FOUND at: {env_path.absolute()}")
        print(f"   Create a .env file with your API keys!")
        return
    
    # Check environment variables are loaded
    print(f"\n2. Checking environment variables...")
    from dotenv import load_dotenv
    load_dotenv()
    
    together_key = os.getenv('TOGETHER_API_KEY')
    openrouter_key = os.getenv('OPENROUTER_API_KEY')
    
    print(f"   TOGETHER_API_KEY loaded: {bool(together_key)} ({len(together_key) if together_key else 0} chars)")
    print(f"   OPENROUTER_API_KEY loaded: {bool(openrouter_key)} ({len(openrouter_key) if openrouter_key else 0} chars)")
    
    if not together_key and not openrouter_key:
        print(f"   ✗ No API keys found!")
        return
    
    # Check llm_answer.py exists
    print(f"\n3. Checking llm_answer.py script...")
    script_path = Path('llm_answer.py')
    if not script_path.exists():
        print(f"   ✗ llm_answer.py NOT FOUND at: {script_path.absolute()}")
        return
    print(f"   ✓ Script found at: {script_path.absolute()}")
    
    # Test with sample chunks
    print(f"\n4. Testing llm_answer.py with sample data...")
    
    query = "What is supervised learning?"
    chunk1 = """Supervised learning is a machine learning approach where the algorithm 
    learns from labeled training data. In supervised learning, each training example 
    consists of an input and a corresponding desired output (label). The algorithm 
    learns to map inputs to outputs by finding patterns in the training data."""
    
    chunk2 = """The goal of supervised learning is to learn a function that can predict 
    the output for new, unseen inputs. Common examples include classification tasks 
    (predicting categories) and regression tasks (predicting continuous values)."""
    
    try:
        # Run the script
        cmd = [sys.executable, str(script_path), query, chunk1, chunk2]
        print(f"\n   Running: {' '.join(cmd[:2])} <query> <chunks>")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        print(f"\n   Return code: {result.returncode}")
        print(f"\n   === STDOUT ({len(result.stdout)} chars) ===")
        print(result.stdout[:1000])
        
        if result.stderr:
            print(f"\n   === STDERR ({len(result.stderr)} chars) ===")
            print(result.stderr[:1000])
        
        if result.returncode == 0:
            print(f"\n   ✓ SUCCESS! Script executed properly")
            
            # Try to parse JSON
            import json
            try:
                data = json.loads(result.stdout)
                if 'answer' in data:
                    print(f"\n   ✓ Valid JSON response with answer field")
                    print(f"\n   Answer preview:")
                    print(f"   {data['answer'][:200]}...")
                elif 'error' in data:
                    print(f"\n   ✗ Script returned error: {data.get('error')}")
                    print(f"   Message: {data.get('message', 'N/A')}")
            except json.JSONDecodeError as e:
                print(f"\n   ✗ Could not parse JSON response: {e}")
        else:
            print(f"\n   ✗ FAILED with return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        print(f"\n   ✗ Script timeout (60s)")
    except Exception as e:
        print(f"\n   ✗ Error running script: {e}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    test_llm_script()