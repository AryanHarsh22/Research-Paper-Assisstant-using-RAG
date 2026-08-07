"""
Verifies the Generation pipeline components:
  1. PromptBuilder prompt construction logic
  2. CitationProcessor citation verification (validated vs unverified matches)
  3. LLMClient (OllamaClient and OpenAIClient) API request formatting using mocked urlopen calls
"""
import os
import sys
import urllib.request
from io import BytesIO
import json

# Ensure UTF-8 output on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.generation.prompt import PromptBuilder
from src.generation.llm import get_llm_client, OllamaClient, OpenAIClient
from src.generation.citations import CitationProcessor


def section(title: str):
    print(f"\n{'=' * 55}")
    print(f"  {title}")
    print(f"{'=' * 55}")


def ok(msg: str):   print(f"  [OK]   {msg}")
def fail(msg: str): print(f"  [FAIL] {msg}")


# --- MOCKING URLOPEN ---
class MockURLResponse:
    def __init__(self, data: dict):
        self.data = data

    def read(self):
        return json.dumps(self.data).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def test_prompt_builder():
    section("STAGE 1 — PromptBuilder")
    
    mock_chunks = [
        {
            "chunk_id": "doc1_p1_c0",
            "text": "Self-attention mechanism is a key component.",
            "metadata": {"source": "attention.pdf", "page": 1}
        },
        {
            "chunk_id": "doc1_p4_c2",
            "text": "Multi-head attention allows joint attention to different projection subspaces.",
            "metadata": {"source": "attention.pdf", "page": 4}
        }
    ]
    query = "What is attention?"
    
    prompt = PromptBuilder.build_rag_prompt(query, mock_chunks)
    
    assert "System Instructions:" in prompt, "Prompt missing system instructions"
    assert "attention.pdf" in prompt, "Prompt missing context sources"
    assert "Page: 4" in prompt, "Prompt missing page numbers"
    assert "Multi-head attention" in prompt, "Prompt missing context text"
    assert query in prompt, "Prompt missing user question"
    
    ok("PromptBuilder successfully structured the prompt.")


def test_citation_processor():
    section("STAGE 2 — CitationProcessor")

    mock_chunks = [
        {
            "chunk_id": "attention_p4_c0",
            "text": "Multi-head attention maps queries to keys and values.",
            "metadata": {"source": "attention_is_all_you_need.pdf", "page": 4}
        },
        {
            "chunk_id": "vit_p2_c1",
            "text": "Vision Transformers apply self-attention directly to image patches.",
            "metadata": {"source": "vit_paper.pdf", "page": 2}
        }
    ]

    # Test cases containing inline citations:
    # 1. Valid citation: [attention_is_all_you_need.pdf, p. 4]
    # 2. Valid citation (fuzzy filename check): [vit_paper, p. 2]
    # 3. Invalid page: [vit_paper.pdf, p. 99]
    # 4. Invalid document: [random_paper.pdf, p. 4]
    raw_response = (
        "We use multi-head attention [attention_is_all_you_need.pdf, p. 4]. "
        "Also, ViTs split images into patches [vit_paper, p. 2] and process them. "
        "Some details are described in [vit_paper.pdf, p. 99] and [random_paper.pdf, p. 4]."
    )

    result = CitationProcessor.verify_citations(raw_response, mock_chunks)
    
    processed_text = result["text"]
    citations = result["citations"]

    assert len(citations) == 4, f"Expected 4 citations, found {len(citations)}"
    
    # 1. Check verified
    c1 = citations[0]
    assert c1["verified"] is True, "First citation should be verified"
    assert c1["chunk_id"] == "attention_p4_c0", "Incorrect chunk_id matched"
    assert "attention_is_all_you_need.pdf, p. 4" in processed_text, "Citation text missing"
    assert "**[attention_is_all_you_need.pdf, p. 4]**" in processed_text, "Verified citation should be bolded"

    # 2. Check fuzzy verified
    c2 = citations[1]
    assert c2["verified"] is True, "Second fuzzy citation should be verified"
    assert c2["chunk_id"] == "vit_p2_c1", "Incorrect chunk_id matched"
    assert "**[vit_paper.pdf, p. 2]**" in processed_text or "**[vit_paper, p. 2]**" in processed_text, "Fuzzy verified citation should be bolded"

    # 3. Check invalid page unverified
    c3 = citations[2]
    assert c3["verified"] is False, "Third citation with invalid page should be unverified"
    assert "**[vit_paper.pdf, p. 99] [UNVERIFIED]**" in processed_text, "Unverified citation mismatch in output"

    # 4. Check invalid document unverified
    c4 = citations[3]
    assert c4["verified"] is False, "Fourth citation with invalid doc should be unverified"
    assert "**[random_paper.pdf, p. 4] [UNVERIFIED]**" in processed_text, "Unverified citation mismatch in output"

    ok("CitationProcessor correctly validated and marked citations.")


def test_llm_clients():
    section("STAGE 3 — LLM Clients (Mocked Network)")

    # Save original urlopen to restore later
    original_urlopen = urllib.request.urlopen
    
    last_requested_url = None
    last_requested_payload = None

    def mock_urlopen(req, timeout=None):
        nonlocal last_requested_url, last_requested_payload
        last_requested_url = req.full_url
        if req.data:
            last_requested_payload = json.loads(req.data.decode("utf-8"))
        
        # Determine client type based on URL
        if "localhost" in req.full_url:
            return MockURLResponse({"response": "Hello from Ollama!"})
        else:
            return MockURLResponse({
                "choices": [
                    {
                        "message": {
                            "content": "Hello from OpenAI!"
                        }
                    }
                ]
            })

    # Apply monkeypatch
    urllib.request.urlopen = mock_urlopen

    try:
        # Test Ollama
        client_ollama = get_llm_client("ollama", "llama3", temperature=0.2)
        response_ollama = client_ollama.generate("Test prompt")
        assert response_ollama == "Hello from Ollama!", "Ollama response parsing failed"
        assert last_requested_url.endswith("/api/generate"), "Incorrect Ollama endpoint url"
        assert last_requested_payload["model"] == "llama3", "Incorrect model in payload"
        assert last_requested_payload["options"]["temperature"] == 0.2, "Incorrect temperature in payload"
        ok("OllamaClient works and parses responses correctly.")

        # Test OpenAI
        client_openai = get_llm_client("openai", "gpt-4o-mini", api_key="sk-testkey", temperature=0.7)
        response_openai = client_openai.generate("Test prompt")
        assert response_openai == "Hello from OpenAI!", "OpenAI response parsing failed"
        assert last_requested_url == "https://api.openai.com/v1/chat/completions", "Incorrect OpenAI endpoint"
        assert last_requested_payload["model"] == "gpt-4o-mini", "Incorrect model in payload"
        assert last_requested_payload["temperature"] == 0.7, "Incorrect temperature in payload"
        ok("OpenAIClient works and parses responses correctly.")

    finally:
        # Restore original urlopen
        urllib.request.urlopen = original_urlopen


def main():
    try:
        test_prompt_builder()
        test_citation_processor()
        test_llm_clients()
        section("ALL GENERATION TESTS PASSED ✓")
    except AssertionError as e:
        fail(f"Test assertion failed: {e}")
    except Exception as e:
        fail(f"Unexpected exception during verification: {e}")


if __name__ == "__main__":
    main()
