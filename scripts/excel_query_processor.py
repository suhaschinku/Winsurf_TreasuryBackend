from typing import Optional
from logger_setup import get_logger
from gen_ai_hub.proxy.native.amazon.clients import Session
from dotenv import load_dotenv
from llm_client import execute_final_analysis
load_dotenv()

logger = get_logger()

def process_excel_only_query(user_prompt: str, excel_non_transcript_store) -> str:
    """Process a query using only the Excel non-transcript vector store."""
    if not user_prompt:
        logger.error("No user prompt provided")
        return "Error: No user prompt provided."
    if excel_non_transcript_store is None:
        logger.error("Excel non-transcript store is not provided")
        return "Error: Excel non-transcript store is required."

    try:
        # Embed the user prompt using the same embedding model as the store
        query_embedding = excel_non_transcript_store.embedding.embed_query(user_prompt)
        logger.info(f"Embedded user prompt for Excel search: {user_prompt}")

        # Search the Excel non-transcript vector store
        k = 10  # Number of relevant chunks to retrieve
        results = excel_non_transcript_store.similarity_search_with_score(
            query=user_prompt,
            k=k
        )

        # Retrieve relevant chunks
        retrieved_chunks = [doc.page_content for doc, score in results]
        context = "\n".join(retrieved_chunks) if retrieved_chunks else "No relevant data found in Excel non-transcript store."
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks from Excel non-transcript store")

        # Combine prompt with retrieved context
        rag_prompt = f"Context from Excel files:\n{context}\n\nUser query: {user_prompt}"

        # Call Claude 3.5 Sonnet via excelAnalysis
        result = execute_final_analysis(rag_prompt)
        logger.info("Successfully generated response from Claude 3.5 Sonnet for Excel query")
        return result

    except Exception as e:
        logger.error(f"Error processing Excel query: {e}")
        return f"Error processing Excel query: {str(e)}"