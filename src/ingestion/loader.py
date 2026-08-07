import os
from typing import List, Dict, Any
import fitz  # PyMuPDF
from collections import Counter
import re
import unicodedata

class PDFLoader:
    """
    Loader class to extract and clean text from PDF files.
    Identifies common headers/footers dynamically using line frequencies.
    """
    def __init__(self, remove_headers_footers: bool = True, header_threshold: float = 0.3):
        self.remove_headers_footers = remove_headers_footers
        self.header_threshold = header_threshold

    def load_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Loads a PDF and extracts cleaned text from each page.
        
        Args:
            file_path (str): Path to the PDF file.
            
        Returns:
            List[Dict[str, Any]]: List of pages with cleaned text and metadata:
                [
                    {
                        "text": "...",
                        "metadata": {"source": "filename.pdf", "page": 1}
                    },
                    ...
                ]
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        doc = fitz.open(file_path)
        filename = os.path.basename(file_path)
        
        # Pre-pass to find candidate headers/footers if enabled
        common_lines = set()
        if self.remove_headers_footers and len(doc) > 2:
            line_counts = Counter()
            for page in doc:
                page_lines = [line.strip() for line in page.get_text("text").split("\n") if line.strip()]
                # Use a set per page to count document-wide page frequency of lines
                for unique_line in set(page_lines):
                    # Avoid filtering very short common words or empty strings
                    if len(unique_line) > 5:
                        line_counts[unique_line] += 1
            
            # Lines appearing on more than header_threshold fraction of pages
            threshold_count = max(2, int(len(doc) * self.header_threshold))
            common_lines = {line for line, count in line_counts.items() if count >= threshold_count}

        documents = []
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text_layout = page.get_text("text")
            cleaned_text = self._clean_page_text(text_layout, common_lines)
            
            if cleaned_text:
                documents.append({
                    "text": cleaned_text,
                    "metadata": {
                        "source": filename,
                        "page": page_num
                    }
                })
                
        doc.close()
        return documents

    def _clean_page_text(self, text: str, common_lines: set) -> str:
        # Normalize unicode characters
        text = unicodedata.normalize("NFKC", text)
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue
            
            # Filter out common lines (headers/footers)
            if line_str in common_lines:
                continue
                
            # Filter out standalone page numbers
            if re.match(r'^\d+$', line_str):
                continue
            if re.match(r'^page\s+\d+(\s+of\s+\d+)?$', line_str, re.IGNORECASE):
                continue
                
            cleaned_lines.append(line_str)
            
        # Reconstruct text handling hyphenations at end of lines
        reconstructed = []
        for i, line in enumerate(cleaned_lines):
            if line.endswith('-') and i < len(cleaned_lines) - 1:
                reconstructed.append(line[:-1])
            else:
                if line.endswith(' ') or line.endswith('-'):
                    reconstructed.append(line)
                else:
                    reconstructed.append(line + " ")
                    
        full_text = "".join(reconstructed)
        # Replace multiple spaces with a single space
        full_text = re.sub(r'\s+', ' ', full_text)
        return full_text.strip()
 