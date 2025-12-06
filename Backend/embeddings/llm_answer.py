#!/usr/bin/env python3
"""
LLM Answer Generation Script - Fixed Version with Enhanced Debugging
Generates clean, well-structured answers from textbook excerpts
Usage: python llm_answer.py "user query" "chunk1" "chunk2" "chunk3" "chunk4" "chunk5"
"""

import sys
import json
import requests
from typing import List
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API KEYS - Get from environment variables
TOGETHER_API_KEY = os.getenv('TOGETHER_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')


def log_debug(message):
    """Log debug message to stderr so it doesn't interfere with JSON output"""
    print(f"[LLM_DEBUG] {message}", file=sys.stderr)


def call_together_ai(chunks: List[str], user_query: str) -> str:
    """Call Together.ai API for answer generation"""
    try:
        log_debug(f"Calling Together.ai with {len(chunks)} chunks")
        
        excerpts = '\n\n'.join([f"[Excerpt {i+1}]\n{chunk[:1000]}" for i, chunk in enumerate(chunks)])

        prompt = f"""You are an expert academic tutor. Answer the student's question using ONLY the textbook excerpts provided below.

CRITICAL INSTRUCTIONS:
1. Write a clear, comprehensive, well-structured answer
2. Use ONLY information from the excerpts - do NOT add external knowledge
3. Break down complex concepts into understandable parts
4. Use proper academic language and terminology
5. Include relevant examples from the excerpts when available
6. If multiple excerpts discuss the same topic, synthesize them coherently
7. If the excerpts don't contain enough information, state: "The provided excerpts do not contain sufficient information to fully answer this question. Based on what's available, [provide partial answer]."

FORMAT YOUR ANSWER:
- Start with a clear, direct answer to the question
- Follow with detailed explanation in well-organized paragraphs
- Use bullet points ONLY for lists or key points
- End with a brief summary if the answer is long

TEXTBOOK EXCERPTS:
{excerpts}

STUDENT'S QUESTION:
{user_query}

YOUR ANSWER:"""

        headers = {
            'Authorization': f'Bearer {TOGETHER_API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            'model': 'mistralai/Mistral-7B-Instruct-v0.1',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            'temperature': 0.2,
            'top_p': 0.9,
            'repetition_penalty': 1.1
        }

        log_debug("Sending request to Together.ai...")
        response = requests.post(
            'https://api.together.xyz/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        log_debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        answer = data['choices'][0]['message']['content'].strip()
        
        log_debug(f"Received answer: {len(answer)} chars")
        
        # Clean up the answer
        answer = clean_answer(answer)
        return answer
    
    except requests.exceptions.RequestException as e:
        log_debug(f"Together.ai request error: {str(e)}")
        raise Exception(f"Together.ai API error: {str(e)}")
    except Exception as e:
        log_debug(f"Together.ai error: {str(e)}")
        raise Exception(f"Together.ai error: {str(e)}")


def call_openrouter(chunks: List[str], user_query: str) -> str:
    """Call OpenRouter API for answer generation"""
    try:
        log_debug(f"Calling OpenRouter with {len(chunks)} chunks")
        
        excerpts = '\n\n'.join([f"[Excerpt {i+1}]\n{chunk[:1000]}" for i, chunk in enumerate(chunks)])

        prompt = f"""You are an expert academic tutor helping a student understand textbook concepts.

YOUR TASK:
Answer the student's question using ONLY the textbook excerpts provided below. Create a clear, well-structured explanation that helps the student fully understand the concept.

REQUIREMENTS:
✓ Use ONLY information from the excerpts below
✓ Write in clear, engaging prose with proper paragraphs
✓ Break complex concepts into digestible parts
✓ Define key terms when first introduced
✓ Use examples from the excerpts to illustrate points
✓ Maintain academic rigor while being accessible
✓ If information is incomplete, acknowledge it and provide what you can

FORMAT:
- Begin with a concise, direct answer (1-2 sentences)
- Follow with detailed explanation in organized paragraphs
- Use subheadings (with ###) if explaining multiple aspects
- Use bullet points sparingly, only for actual lists
- Conclude with a brief summary for longer answers

TEXTBOOK EXCERPTS:
{excerpts}

STUDENT'S QUESTION:
{user_query}

YOUR ANSWER:"""

        headers = {
            'Authorization': f'Bearer {OPENROUTER_API_KEY}',
            'Content-Type': 'application/json',
            'HTTP-Referer': 'http://localhost:3000',
            'X-Title': 'Textbook Search Assistant'
        }

        payload = {
            'model': 'mistralai/mistral-7b-instruct',
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'max_tokens': 1000,
            'temperature': 0.2,
            'top_p': 0.9
        }

        log_debug("Sending request to OpenRouter...")
        response = requests.post(
            'https://openrouter.ai/api/v1/chat/completions',
            headers=headers,
            json=payload,
            timeout=60
        )
        
        log_debug(f"Response status: {response.status_code}")
        response.raise_for_status()
        
        data = response.json()
        answer = data['choices'][0]['message']['content'].strip()
        
        log_debug(f"Received answer: {len(answer)} chars")
        
        # Clean up the answer
        answer = clean_answer(answer)
        return answer
    
    except requests.exceptions.RequestException as e:
        log_debug(f"OpenRouter request error: {str(e)}")
        raise Exception(f"OpenRouter API error: {str(e)}")
    except Exception as e:
        log_debug(f"OpenRouter error: {str(e)}")
        raise Exception(f"OpenRouter error: {str(e)}")


def clean_answer(answer: str) -> str:
    """
    Clean and format the LLM answer for better presentation.
    
    Args:
        answer: Raw answer from LLM
        
    Returns:
        Cleaned and formatted answer
    """
    import re
    
    # Remove common LLM artifacts
    answer = answer.strip()
    
    # Remove "Answer:", "Response:", etc. if at the start
    prefixes_to_remove = [
        "Answer:", "ANSWER:", "Response:", "RESPONSE:",
        "Here's the answer:", "Here is the answer:",
        "Based on the excerpts:", "According to the excerpts:"
    ]
    
    for prefix in prefixes_to_remove:
        if answer.startswith(prefix):
            answer = answer[len(prefix):].strip()
    
    # Remove excessive newlines (max 2 consecutive)
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    
    # Remove trailing/leading whitespace from each line
    lines = [line.rstrip() for line in answer.split('\n')]
    answer = '\n'.join(lines)
    
    # Ensure proper spacing after punctuation
    answer = re.sub(r'([.!?])([A-Z])', r'\1 \2', answer)
    
    # Remove redundant spacing
    answer = re.sub(r' +', ' ', answer)
    
    return answer.strip()


def format_answer_with_metadata(answer: str, chunks_count: int, query: str, api_used: str) -> dict:
    """
    Format the final answer with helpful metadata.
    
    Args:
        answer: The generated answer
        chunks_count: Number of chunks used
        query: Original query
        api_used: Which API was used
        
    Returns:
        Formatted result dictionary
    """
    # Calculate answer metrics
    word_count = len(answer.split())
    has_structure = any(marker in answer for marker in ['###', '**', '•', '-', '1.', '2.'])
    
    return {
        "answer": answer,
        "metadata": {
            "api_used": api_used,
            "model": "mistralai/mistral-7b-instruct" if api_used == 'openrouter' else "mistralai/Mistral-7B-Instruct-v0.1",
            "chunks_processed": chunks_count,
            "word_count": word_count,
            "has_formatting": has_structure,
            "query": query
        },
        "status": "success"
    }


def main():
    try:
        log_debug(f"Script started with {len(sys.argv)} arguments")
        
        if len(sys.argv) < 3:
            result = {
                "error": "Insufficient arguments",
                "message": "Usage: python llm_answer.py 'user_query' 'chunk1' 'chunk2' ...",
                "received_args": len(sys.argv) - 1
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)

        user_query = sys.argv[1].strip()
        chunks = [chunk.strip() for chunk in sys.argv[2:] if chunk.strip()]

        log_debug(f"Query: {user_query[:50]}...")
        log_debug(f"Chunks received: {len(chunks)}")

        if not user_query:
            result = {"error": "Empty query provided"}
            print(json.dumps(result, indent=2))
            sys.exit(1)

        if not chunks:
            result = {"error": "No valid content chunks provided"}
            print(json.dumps(result, indent=2))
            sys.exit(1)

        # Limit to top 5 chunks for quality
        chunks = chunks[:5]
        log_debug(f"Using top {len(chunks)} chunks")

        answer = None
        api_used = None
        error_details = []

        # Check API keys
        log_debug(f"Together API key present: {bool(TOGETHER_API_KEY)}")
        log_debug(f"OpenRouter API key present: {bool(OPENROUTER_API_KEY)}")

        # Try Together.ai first
        if TOGETHER_API_KEY:
            try:
                log_debug("Attempting Together.ai...")
                answer = call_together_ai(chunks, user_query)
                api_used = 'together.ai'
                log_debug("Together.ai succeeded!")
            except Exception as e:
                error_msg = f"Together.ai failed: {str(e)}"
                log_debug(error_msg)
                error_details.append(error_msg)
        else:
            error_msg = "Together.ai API key not found in environment"
            log_debug(error_msg)
            error_details.append(error_msg)

        # If Together.ai fails, try OpenRouter
        if not answer and OPENROUTER_API_KEY:
            try:
                log_debug("Attempting OpenRouter...")
                answer = call_openrouter(chunks, user_query)
                api_used = 'openrouter'
                log_debug("OpenRouter succeeded!")
            except Exception as e:
                error_msg = f"OpenRouter failed: {str(e)}"
                log_debug(error_msg)
                error_details.append(error_msg)
        elif not answer:
            error_msg = "OpenRouter API key not found in environment"
            log_debug(error_msg)
            error_details.append(error_msg)

        if not answer:
            log_debug("All APIs failed, returning error")
            result = {
                "error": "All LLM APIs failed",
                "details": error_details,
                "query": user_query,
                "chunks_provided": len(chunks),
                "suggestion": "Please check your API keys in the .env file"
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)

        # Format and return success response
        log_debug("Formatting final response...")
        result = format_answer_with_metadata(answer, len(chunks), user_query, api_used)
        
        # Print ONLY the JSON to stdout
        print(json.dumps(result, indent=2))
        log_debug("Success! JSON output sent to stdout")

    except Exception as e:
        log_debug(f"Unexpected error: {str(e)}")
        result = {
            "error": "Script execution error",
            "message": str(e),
            "query": sys.argv[1] if len(sys.argv) > 1 else "unknown",
            "traceback": str(e)
        }
        print(json.dumps(result, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()