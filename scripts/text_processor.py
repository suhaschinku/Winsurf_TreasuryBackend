from logger_setup import get_logger
from typing import Tuple, List

logger = get_logger()

def format_documents(docs):
    """Format retrieved documents into a single string with metadata."""
    logger.info(f"Starting document formatting for {len(docs)} documents")
    formatted = "\n\n".join(
        f"{doc.page_content} (Source: {doc.metadata.get('file_name', 'Unknown')}, Page: {doc.metadata.get('Page', 'Unknown')})"
        for doc in docs
    )
    logger.info(f"Documents formatted, total length: {len(formatted)} characters")
    logger.debug(f"Documents formatted, total length: {len(formatted)} characters")
    return formatted

def parse_query(query: str) -> Tuple[str, List[str]]:
    """Identify analysis types and clean the query."""
    logger.info(f"Starting query parsing for: '{query[:50]}...'")
    query_lower = query.lower()
    analysis_terms = {
        "general": ["summarize", "summary", "summarization"],
        "financial": ["finance", "financial", "revenue", "profit", "earnings", "income", "balance sheet", "impairments", "highlights"],
        "trend": ["trend", "trends", "growth", "decline", "change over time", "outlook"],
        "topics": ["topics", "themes", "breakdown", "categorize", "categorization", "topic breakdown", "topic analysis"],
        "quotes": ["quote", "quotes", "statement", "statements"],
        "callouts": ["callouts", "major callouts"],
        "consensus": ["consensus"],
        "Stock": ["Stock, Share Price Analysis"]
    }
    detected_types = []
    for analysis_type, terms in analysis_terms.items():
        for term in terms:
            if term in query_lower and len(term.split()) > 1:
                if analysis_type not in detected_types:
                    detected_types.append(analysis_type)
    if not detected_types:
        for analysis_type, terms in analysis_terms.items():
            if any(term in query_lower.split() for term in terms if len(term.split()) == 1):
                if analysis_type not in detected_types:
                    detected_types.append(analysis_type)
    if not detected_types:
        logger.info("No specific analysis type detected, defaulting to 'general'")
        return query, ["general"]
    cleaned_query = query
    for term in sum(analysis_terms.values(), []):
        cleaned_query = cleaned_query.replace(term, "").strip()
    logger.info(f"Query parsed, detected types: {detected_types}, cleaned query: '{cleaned_query[:50]}...'")
    return cleaned_query, detected_types