import re
from typing import List, Dict, Any

class CitationProcessor:
    """
    Parses and validates inline citations from LLM generated responses,
    cross-referencing them against retrieved context chunks.
    """
    # Regex to match [Filename, p. X] or [Filename, p.X] (with or without spacing)
    CITATION_PATTERN = re.compile(r'\[([^\]]+?),\s*p\.\s*(\d+)\]', re.IGNORECASE)

    @staticmethod
    def _normalize_filename(name: str) -> str:
        """Normalizes filename for fuzzy matching (removes extensions, spaces, punctuation)."""
        name = name.lower()
        # Remove common extensions
        if name.endswith('.pdf'):
            name = name[:-4]
        # Retain only alphanumeric characters
        return "".join(c for c in name if c.isalnum())

    @classmethod
    def verify_citations(cls, response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts, parses, and validates inline citations.
        
        Args:
            response_text (str): The raw response text from the LLM.
            retrieved_chunks (List[Dict[str, Any]]): List of chunk dicts from retrieval.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - "text" (str): Response text with citations marked as verified or unverified.
                - "citations" (List[Dict]): Detailed records of each citation parsed.
                    [
                        {
                            "raw": "[attention.pdf, p. 4]",
                            "document": "attention_is_all_you_need.pdf",
                            "page": 4,
                            "verified": True,
                            "chunk_id": "...",
                            "snippet": "..."
                        },
                        ...
                    ]
        """
        # Build index of retrieved chunks for fast lookup
        # Key: (normalized_source_name, page_number) -> chunk_dict
        retrieved_index = {}
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "")
            page = meta.get("page")
            
            if source and page is not None:
                norm_source = cls._normalize_filename(source)
                try:
                    page_num = int(page)
                except ValueError:
                    continue
                retrieved_index[(norm_source, page_num)] = chunk

        citations_list = []
        processed_text = response_text
        
        # We search and replace each citation in response_text
        # Use finditer to work with match positions
        offset = 0
        for match in cls.CITATION_PATTERN.finditer(response_text):
            raw_match = match.group(0)
            cited_doc = match.group(1).strip()
            cited_page_str = match.group(2).strip()
            
            try:
                cited_page = int(cited_page_str)
            except ValueError:
                continue

            norm_cited_doc = cls._normalize_filename(cited_doc)
            
            # Find matching chunk in index
            verified = False
            matched_chunk = None
            
            # Check for exact or substring match in our retrieved chunks
            for (norm_source, page), chunk in retrieved_index.items():
                if page == cited_page and (norm_cited_doc in norm_source or norm_source in norm_cited_doc):
                    verified = True
                    matched_chunk = chunk
                    break
            
            citation_info = {
                "raw": raw_match,
                "document": cited_doc,
                "page": cited_page,
                "verified": verified,
                "chunk_id": matched_chunk.get("chunk_id") if matched_chunk else None,
                "snippet": matched_chunk.get("text") if matched_chunk else None
            }
            citations_list.append(citation_info)

            # Highlight citation status in output text
            status_suffix = "" if verified else " [UNVERIFIED]"
            replacement = f"**{raw_match}{status_suffix}**"
            
            # Update processed_text using the offset-corrected index
            start = match.start() + offset
            end = match.end() + offset
            processed_text = processed_text[:start] + replacement + processed_text[end:]
            offset += len(replacement) - len(raw_match)

        return {
            "text": processed_text,
            "citations": citations_list
        }
