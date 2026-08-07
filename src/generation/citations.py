import re
from typing import List, Dict, Any

class CitationProcessor:
    """
    Parses and validates inline citations from LLM generated responses,
    cross-referencing them against retrieved context chunks.
    """
    # Regex to match [Filename, p. X], [Filename, p X], [Filename, page X], [Filename, pp. X]
    CITATION_PATTERN = re.compile(r'\[([^\]]+?),\s*(?:p\.|p|page|pp\.)\s*(\d+)\]', re.IGNORECASE)

    @staticmethod
    def _normalize_filename(name: str) -> str:
        """Normalizes filename for fuzzy matching (removes extensions, spaces, punctuation)."""
        name = name.lower()
        if name.endswith('.pdf'):
            name = name[:-4]
        return "".join(c for c in name if c.isalnum())

    @classmethod
    def verify_citations(cls, response_text: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extracts, parses, and validates inline citations.
        
        Args:
            response_text (str): The raw response text from the LLM.
            retrieved_chunks (List[Dict[str, Any]]): List of chunk dicts from retrieval.
            
        Returns:
            Dict[str, Any]: Verified citations result dictionary.
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
            
            verified = False
            matched_chunk = None
            resolved_doc_name = cited_doc
            
            # 1. Direct filename matching (exact or substring)
            for (norm_source, page), chunk in retrieved_index.items():
                if page == cited_page and (norm_cited_doc in norm_source or norm_source in norm_cited_doc):
                    verified = True
                    matched_chunk = chunk
                    resolved_doc_name = chunk.get("metadata", {}).get("source", cited_doc)
                    break
            
            # 2. Fallback matching for "Context 1", "Passage 2", "Chunk 3" references
            if not verified:
                passage_match = re.match(r'^(?:context|passage|chunk|doc)\s*(\d+)$', norm_cited_doc, re.IGNORECASE)
                if passage_match:
                    passage_idx = int(passage_match.group(1)) - 1  # 1-indexed -> 0-indexed
                    if 0 <= passage_idx < len(retrieved_chunks):
                        target_chunk = retrieved_chunks[passage_idx]
                        chunk_meta = target_chunk.get("metadata", {})
                        chunk_page = chunk_meta.get("page")
                        try:
                            # Verify page number or match passage chunk
                            if chunk_page is not None and (int(chunk_page) == cited_page or cited_page >= 1):
                                verified = True
                                matched_chunk = target_chunk
                                resolved_doc_name = chunk_meta.get("source", cited_doc)
                                cited_page = int(chunk_page) if chunk_page is not None else cited_page
                        except (ValueError, TypeError):
                            pass
            
            citation_info = {
                "raw": raw_match,
                "document": resolved_doc_name,
                "page": cited_page,
                "verified": verified,
                "chunk_id": matched_chunk.get("chunk_id") if matched_chunk else None,
                "snippet": matched_chunk.get("text") if matched_chunk else None
            }
            citations_list.append(citation_info)

            # Highlight citation status in output text
            status_suffix = "" if verified else " [UNVERIFIED]"
            replacement = f"**[{resolved_doc_name}, p. {cited_page}]**" if (verified and resolved_doc_name != cited_doc) else f"**{raw_match}{status_suffix}**"
            
            start = match.start() + offset
            end = match.end() + offset
            processed_text = processed_text[:start] + replacement + processed_text[end:]
            offset += len(replacement) - len(raw_match)

        return {
            "text": processed_text,
            "citations": citations_list
        }
