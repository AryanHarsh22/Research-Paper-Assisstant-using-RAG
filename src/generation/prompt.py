from typing import List, Dict, Any

class PromptBuilder:
    """
    Formats the system instructions, retrieved context chunks, and user query
    into a structured prompt for retrieval-augmented generation.
    """

    @staticmethod
    def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Assembles the RAG prompt.
        
        Args:
            query (str): The user's query.
            retrieved_chunks (List[Dict[str, Any]]): Retrieved chunks from the retriever.
            
        Returns:
            str: The structured prompt for the LLM.
        """
        if not retrieved_chunks:
            return (
                "System Instructions:\n"
                "Answer the user query. Note that no document context is available.\n\n"
                f"User Question:\n{query}\n\n"
                "Answer:"
            )

        context_str = ""
        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "?")
            text = chunk.get("text", "").strip()
            
            # Format each chunk with clear document source label
            context_str += f"[Passage {i + 1}] Document: {source} | Page: {page}\n{text}\n\n"

        prompt = (
            "System Instructions:\n"
            "You are an expert research assistant. Answer the user's question strictly based "
            "on the provided context passages below. For every claim or statement you make, "
            "you MUST cite the exact source document filename and page number in line using the format [Document_Name.pdf, p. X]. "
            "For example: 'Attention mechanisms allow modeling dependencies [attention_is_all_you_need.pdf, p. 4].'\n"
            "IMPORTANT: Always use the exact document filename (e.g. attention_is_all_you_need.pdf) in citations, NOT 'Passage 1' or 'Context 1'.\n"
            "Do not cite if the claim is not directly supported by that specific passage. "
            "If the answer cannot be found in the provided context, state: 'I could not find the answer in the provided documents.' "
            "Do NOT use external knowledge or make up claims.\n\n"
            "Provided Context:\n"
            f"{context_str}"
            "User Question:\n"
            f"{query}\n\n"
            "Grounded Answer:"
        )
        return prompt
