from typing import Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from llm_client import execute_final_analysis, data_formatter, extract_data_requirements, run_orchestration
from text_processor import format_documents, parse_query
from coda_analyzer import generate_coda_prompt
from logger_setup import get_logger
from langchain_community.vectorstores import HanaDB
from image_processor import process_images
import os

logger = get_logger()

def generate_summary_template(context: str, query: str, analysis_type: str = "general", transcript_context: str = None) -> str:
    """Create a concise summary template based on analysis type."""
    logger.info(f"Generating summary template for query: '{query[:50]}...' with analysis type: {analysis_type}")
    template = "Key points for '{query}': Use {context} only; Provide a small section of writeup max 2 sentences on how the answer is interpreted"
    formatted_template = template.format(
        query=query, 
        context=context, 
        transcript_context=transcript_context or "No transcript context"
    )
    logger.info(f"Summary template generated, length: {len(formatted_template)} characters")
    return formatted_template

def fetch_context(retriever, query: str, k: int = 30) -> str:
    """Retrieve documents with minimal memory usage."""
    logger.info(f"Fetching context for query: '{query[:50]}...'")
    try:
        docs = retriever.invoke(query)
        context = format_documents(docs) if docs else "No documents found."
        logger.info(f"Context fetched, length: {len(context)} characters")
        logger.debug(f"Fetched context for query '{query[:50]}...': {context[:100]}... (length: {len(context)})")
        return context
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}")
        return f"Error: {str(e)}"

def process_analysis_type(
    analysis_type: str, user_query: str, transcript_store, non_transcript_store, transcript_context: str
) -> Tuple[str, str, str]:
    """Process an individual analysis type."""
    logger.info(f"Starting analysis type processing: {analysis_type} for query: '{user_query[:50]}...'")
    try:
        logger.debug(f"Processing analysis type '{analysis_type}' with user_query: '{user_query[:50]}...' and transcript_context: {transcript_context[:100]}... (length: {len(transcript_context)})")
        # Use non-transcript store for general analysis
        store = non_transcript_store
        retriever = store.as_retriever(search_kwargs={"k": 30})
        context = fetch_context(retriever, user_query)
        task = generate_summary_template(context, user_query, analysis_type, transcript_context)
        logger.info(f"Completed analysis type processing: {analysis_type}")
        return analysis_type, task, context
    except Exception as e:
        logger.error(f"Processing {analysis_type} failed: {str(e)}")
        return analysis_type, f"Error: {str(e)}", f"Error: {str(e)}"

def process_excel_only_query(query: str, excel_non_transcript_store: HanaDB) -> str:
    """Process Excel query efficiently."""
    logger.info(f"Processing Excel query: '{query[:50]}...'")
    try:
        retriever = excel_non_transcript_store.as_retriever(search_kwargs={"k": 30})
        context = fetch_context(retriever, query)
        logger.info(f"Excel query processed, result: {context[:100]}... (length: {len(context)})")
        return context if context and not context.startswith("Error") else "No Excel data"
    except Exception as e:
        logger.error(f"Excel query failed: {str(e)}")
        return "No Excel data"

def process_query(
    query: str,
    transcript_store: Optional[HanaDB] = None,
    non_transcript_store: Optional[HanaDB] = None,
    excel_non_transcript_store: Optional[HanaDB] = None
) -> str:
    """Handle query processing with parallel retrieval, image processing, and Excel query execution."""
    logger.info(f"Starting query processing for: '{query[:50]}...'")
    if not query:
        logger.error("No query provided")
        return "Error: No query provided."
    
    if not all([transcript_store, non_transcript_store, excel_non_transcript_store]):
        logger.error("Missing vector stores")
        return "Error: All vector stores required."

    try:
        # Parse query and generate analysis types
        clean_query, analysis_types = parse_query(query)
        logger.info(f"Query parsed, clean query: '{clean_query[:50]}...', analysis types: {analysis_types}")

        # Fetch transcript context
        transcript_context = fetch_context(transcript_store.as_retriever(search_kwargs={"k": 30}), clean_query)
        logger.info(f"Transcript context fetched, length: {len(transcript_context)} characters")

        # General case: Process all analysis types
        coda_prompt = generate_coda_prompt(query)
        logger.info(f"CODA prompt generated for query: '{query[:50]}...'")
        data_requirements = extract_data_requirements(coda_prompt)
        logger.info(f"Data requirements extracted: {data_requirements}")
        refined_query = f"{clean_query} {data_requirements}"
        logger.info(f"Refined query created: '{refined_query[:50]}...'")

        # Check for stock-related query
        stock_keywords = ["stock", "share price", "stock analysis", "stock insights"]
        is_stock_query = any(keyword.lower() in query.lower() for keyword in stock_keywords)
        logger.info(f"Stock query check: {'Stock query detected' if is_stock_query else 'Not a stock query'}")

        # Parallel processing of image processing, analysis types, and Excel query
        image_results = []
        tasks = {}
        contexts = {}
        excel_result = "No Excel data"

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            logger.info("Starting parallel processing with ThreadPoolExecutor")

            # Submit image processing task if stock query
            if is_stock_query:
                base_path = os.getenv('LOCALPATH', '')
                folder_path = os.path.join(base_path, "Images")
                user_prompt = query
                logger.info(f"Submitting image processing task for folder: {folder_path}")
                if not os.path.exists(folder_path):
                    logger.error(f"Image folder does not exist: {folder_path}")
                    image_results = [{"image_path": "N/A", "analysis": f"Error: Image folder {folder_path} does not exist"}]
                else:
                    futures[executor.submit(process_images, folder_path, user_prompt)] = "image"

            # Submit analysis type tasks
            for atype in analysis_types:
                logger.info(f"Submitting analysis task for type: {atype}")
                futures[executor.submit(
                    process_analysis_type, 
                    atype, 
                    query,
                    transcript_store, 
                    non_transcript_store, 
                    transcript_context
                )] = f"analysis_{atype}"

            # Submit Excel query task
            logger.info("Submitting Excel query task")
            futures[executor.submit(process_excel_only_query, refined_query, excel_non_transcript_store)] = "excel"

            # Collect results as they complete
            for future in as_completed(futures):
                task_type = futures[future]
                try:
                    if task_type == "image":
                        image_results = future.result()
                        logger.info(f"Processed {len(image_results)} images for stock query")
                    elif task_type.startswith("analysis_"):
                        atype, task, context = future.result()
                        tasks[atype] = task
                        contexts[atype] = context
                        logger.info(f"Completed analysis task for type: {atype}")
                    elif task_type == "excel":
                        excel_result = future.result()
                        logger.info("Completed Excel query task")
                except Exception as e:
                    if task_type == "image":
                        logger.error(f"Image processing failed: {str(e)}")
                        image_results = [{"image_path": "N/A", "analysis": f"Error processing images: {str(e)}"}]
                    elif task_type.startswith("analysis_"):
                        logger.error(f"Processing {task_type} failed: {str(e)}")
                        atype = task_type.split("_")[1]
                        tasks[atype] = f"Error: {str(e)}"
                        contexts[atype] = f"Error: {str(e)}"
                    elif task_type == "excel":
                        logger.error(f"Excel query failed: {str(e)}")
                        excel_result = f"Error: {str(e)}"

        # Create integrated prompt
        logger.info("Creating integrated prompt")
        integrated_prompt = f"""
        Tasks: {', '.join(tasks.values())}
        CODA Data: {data_requirements}
        Contexts: {str(contexts)}
        Transcript Context: {transcript_context}
        Query: {query}
        Requirements:
        - Do not hallucinate. If there is no data, STRICTLY state 'Data not Available'.
        - Integrate tasks with CODA analysis
        - Strictly Support data with consised and important details.
        - Format cohesively with confidence metrics
        """
        logger.info(f"Integrated prompt created, length: {len(integrated_prompt)} characters")

        # Execute and format final response
        logger.info("Executing final analysis")
        final_analysis = execute_final_analysis(integrated_prompt)
        logger.info("Final analysis completed")
        response = data_formatter(final_analysis, excel_result, image_results if is_stock_query else None)
        logger.info("Final response formatted")
        return response or "No response due to insufficient data."

    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        return f"Error: {str(e)}"