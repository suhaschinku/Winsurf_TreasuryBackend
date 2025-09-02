import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from langchain.docstore.document import Document
import pdfplumber
import camelot
import tabula
from logger_setup import get_logger
import re
from env_config import TABLE_NAMES, EMBEDDING_MODEL
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details,extract_aicore_credentials
import requests


logger = get_logger()
load_dotenv()

#$$$ SOC: 28.05.25 -- Initialize AIC Credentials --- $$$#
logger.info ("====> pdf_processor.py -> AIC CREDENTIALS <====")

# Load VCAP_SERVICES from environment
vcap_services = os.environ.get("VCAP_SERVICES")

# Extract destination service credentials
destination_service_credentials = get_destination_service_credentials(vcap_services)

# Generate OAuth token for destination service
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
except requests.exceptions.HTTPError as e:
    # Handle HTTP 500 error for invalid client secret
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise


#-------------------------------- READ AIC Configuration -------------------------------------

# variables for AIC credentials
global AIC_CREDENTIALS

# Get AIC details from Dest Services
dest_AIC = "GENAI_AI_CORE"
aicore_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
    
# Extract AIC Details
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
logger.info("pdf_processor -> AIC Credential", AIC_CREDENTIALS)

#$$$ EOC: 28.05.25 -- Initialize AIC Credentials --- $$$#



def clean_text(text):
    """Clean OCR artifacts and normalize text."""
    if not text:
        return ""
    text = re.sub(r"JPHORGAN", "JPMORGAN", text, flags=re.IGNORECASE)
    text = re.sub(r"excape", "except", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()  # Normalize whitespace
    return text

def table_to_text(table_data, source_file, page_num, source_tool="pdfplumber"):
    """Convert table data to natural language text with improved header detection."""
    if not table_data:  # Handle None or empty input
        return ""
    # Handle different table formats based on source tool
    if source_tool == "tabula-py":
        if not isinstance(table_data, list) and hasattr(table_data, "empty"):  # Check if it's a DataFrame
            if table_data.empty:  # Explicitly check for empty DataFrame
                return ""
            table_data = table_data.values.tolist()  # Convert DataFrame to list of lists
        elif not isinstance(table_data, list):  # Invalid tabula output
            return ""
    elif source_tool == "camelot":
        if not table_data or len(table_data.df) < 1:  # Check camelot table
            return ""
        table_data = table_data.df.values.tolist()  # Convert camelot DataFrame to list
    if len(table_data) < 2:  # Need at least header + 1 row
        return ""
    text = []
    headers = table_data[0]
    if all(re.match(r"^\$?\d[\d,.]*$", str(cell).strip()) for cell in headers if str(cell).strip()):
        headers = ["Column " + str(i) for i in range(len(headers))]  # Fallback if no clear header
    else:
        table_data = table_data[1:]  # Skip header row
    for row in table_data:
        if len(row) != len(headers):
            continue  # Skip malformed rows
        row_text = " ".join(f"{headers[i]} is {cell}" for i, cell in enumerate(row) if cell and str(cell).strip())
        if row_text:
            text.append(row_text)
    return "\n".join(text) if text else ""

def process_pdf(pdf_path):
    """Extract text and tables from a PDF file, returning chunks."""
    logger.info(f"Processing PDF: {pdf_path}")
    try:
        # Step 1: Extract text with PyPDFLoader
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        filename = os.path.basename(pdf_path)
        if not documents:
            logger.warning(f"No content extracted from {pdf_path} with PyPDFLoader")
        else:
            for doc in documents:
                doc.metadata['source_file'] = filename
            logger.info(f"Extracted {len(documents)} pages from {filename}, Sample: {documents[0].page_content[:50]}")
        # Step 2: Extract tables with pdfplumber, camelot, and tabula-py
        table_docs = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract tables with pdfplumber
                tables = page.extract_tables()
                for table_idx, table_data in enumerate(tables):
                    table_text = table_to_text(table_data, filename, page_num, source_tool="pdfplumber")
                    if table_text:
                        metadata = {
                            "source_file": filename,
                            "page": page_num,
                            "table_idx": table_idx,
                            "type": "table",
                            "extractor": "pdfplumber"
                        }
                        table_docs.append(Document(page_content=table_text, metadata=metadata))
        # Fallback 1: Camelot
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
            for table in tables:
                table_text = table_to_text(table, filename, table.page, source_tool="camelot")
                if table_text:
                    metadata = {
                        "source_file": filename,
                        "page": table.page,
                        "type": "table",
                        "extractor": "camelot"
                    }
                    table_docs.append(Document(page_content=table_text, metadata=metadata))
        except Exception as e:
            logger.warning(f"Camelot failed for {pdf_path}: {e}")
        # Fallback 2: Tabula-py
        try:
            tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, lattice=True)
            if not tables:
                tables = tabula.read_pdf(pdf_path, pages="all", multiple_tables=True, stream=True)
            for table_idx, table_data in enumerate(tables):
                table_text = table_to_text(table_data, filename, page_num, source_tool="tabula-py")
                if table_text:
                    metadata = {
                        "source_file": filename,
                        "page": page_num,  # Approximate; refine if needed
                        "table_idx": table_idx,
                        "type": "table",
                        "extractor": "tabula-py"
                    }
                    table_docs.append(Document(page_content=table_text, metadata=metadata))
        except Exception as e:
            logger.warning(f"Tabula-py failed for {pdf_path}: {e}")
        # Combine text and table documents
        all_docs = documents + table_docs
        if not all_docs:
            logger.warning(f"No chunks created from {filename}")
            return []
        # Step 3: Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(all_docs)
        if not chunks:
            logger.warning(f"No chunks created from {filename}")
            return []

        #Append filename to each chunk's content
        for chunk in chunks:
            chunk.page_content = f"{chunk.page_content}\n\nSource: {filename}"

        logger.info(f"Processed {len(chunks)} chunks from {filename}")
        return chunks
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}", exc_info=True)
        return []

def create_embeddings(docs, model_name=EMBEDDING_MODEL, batch_size=400):
    """Generate embeddings for document chunks in batches."""
    if not docs:
        logger.warning("No documents provided for embedding")
        return [], 1
    logger.info(f"Creating embeddings for {len(docs)} docs with {model_name}")
    try:
    ### SOC: Initialize Embedding Models ### 
        from gen_ai_hub.proxy import GenAIHubProxyClient
        logger.info("PDF Processor: AIC, {AIC_CREDENTIALS}")

        proxy_client = GenAIHubProxyClient(
                                    base_url = AIC_CREDENTIALS['aic_base_url'],
                                    auth_url = AIC_CREDENTIALS['aic_auth_url'],
                                    client_id = AIC_CREDENTIALS['clientid'],
                                    client_secret = AIC_CREDENTIALS['clientsecret'],
                                    resource_group = AIC_CREDENTIALS['resource_group']
                                    )

        embedding_model = init_embedding_model(model_name = EMBEDDING_MODEL, proxy_client=proxy_client)
    ### EOC: Initialize Embedding Models ### 

        logger.info(f"Embedding model initialized: {embedding_model}")
        results = []
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_texts = [doc.page_content for doc in batch]
            if not batch_texts or any(not isinstance(text, str) for text in batch_texts):
                logger.error(f"Invalid batch content at index {i}: {batch_texts[:50]}")
                return [], 1
            logger.info(f"Processing batch {i//batch_size + 1}, size: {len(batch_texts)}, Sample: {batch_texts[0][:50]}")
            embeddings = embedding_model.embed_documents(batch_texts)
            if not embeddings or len(embeddings) != len(batch_texts):
                logger.error(f"Embedding mismatch: {len(embeddings)} embeddings for {len(batch_texts)} texts")
                return [], 1
            results.extend(zip(batch, embeddings))
            logger.info(f"Batch {i//batch_size + 1} completed")
            if i + batch_size < len(docs):
                time.sleep(10)  # Rate limiting delay
        logger.info(f"Created embeddings for {len(results)} docs")
        return results, 0
    except Exception as e:
        logger.error(f"Embedding error: {str(e)}", exc_info=True)
        return [], 1

def process_pdf_with_embeddings(pdf_path, model=EMBEDDING_MODEL):
    """Process a single PDF and generate embeddings."""
    chunks = process_pdf(pdf_path)
    if not chunks:
        logger.warning(f"No chunks to embed for {pdf_path}")
        return []
    filename = os.path.basename(pdf_path)
    logger.info(f"Using model {model} for {filename}")
    embeddings, error = create_embeddings(chunks, model)
    if error:
        print(f"❌ Embedding creation failed for {filename}!")
    else:
        print(f"✅ Created {len(embeddings)} embeddings for {filename}")
    return embeddings

def process_all_pdfs(directory, model=EMBEDDING_MODEL):
    """Process all PDFs in a directory, separating transcript and non-transcript embeddings."""
    
    print(f"\n{'='*50}")
    print(f"STARTING PDF PROCESSING")
    print(f"{'='*50}")
    print(f"Directory: {directory}")
    print(f"Model: {model}")
    
    # Check directory existence
    if not os.path.exists(directory):
        error_msg = f"Directory not found: {directory}"
        logger.info(error_msg)
        print(f"❌ {error_msg}")
        return [], []
    else:
        logger.info(f"Directory exists: {directory}")
        print(f"✅ Directory exists: {directory}")
    
    # Find PDF files
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    if not pdf_files:
        warning_msg = f"No PDFs found in {directory}"
        logger.info(warning_msg)
        print(f"⚠️  {warning_msg}")
        return [], []
    
    logger.info(f"Found {len(pdf_files)} PDF files: {pdf_files}")
    print(f"📁 Found {len(pdf_files)} PDF files:")
    for i, pdf in enumerate(pdf_files, 1):
        print(f"  {i}. {pdf}")
    
    transcript_embeddings = []
    non_transcript_embeddings = []
    processing_stats = {
        'total_files': len(pdf_files),
        'successful': 0,
        'failed': 0,
        'empty_results': 0,
        'transcript_files': 0,
        'non_transcript_files': 0
    }

    def process_single_pdf(pdf):
        """Process a single PDF file with detailed logging."""
        print(f"\n{'─'*30}")
        print(f"📄 Processing: {pdf}")
        print(f"{'─'*30}")
        
        pdf_path = os.path.join(directory, pdf)
        logger.info(f"Processing PDF: {pdf}")
        print(f"📍 Full path: {pdf_path}")
        
        # Check if file exists and is readable
        if not os.path.exists(pdf_path):
            logger.info(f"File not found: {pdf_path}")
            print(f"❌ File not found: {pdf_path}")
            return None, False
        
        if not os.access(pdf_path, os.R_OK):
            logger.info(f"File not readable: {pdf_path}")
            print(f"❌ File not readable: {pdf_path}")
            return None, False
        
        file_size = os.path.getsize(pdf_path)
        filename = os.path.basename(pdf_path)
        is_transcript = 'transcript' in filename.lower()
        logger.info(f"Processing {pdf} - Size: {file_size} bytes, Type: {'Transcript' if is_transcript else 'Non-transcript'}")
        print(f"📊 File size: {file_size:,} bytes")
        
        filename = os.path.basename(pdf_path)
        is_transcript = 'transcript' in filename.lower()
        print(f"🏷️  File type: {'Transcript' if is_transcript else 'Non-transcript'}")
        
        try:
            print(f"🔄 Calling process_pdf_with_embeddings...")
            embeddings = process_pdf_with_embeddings(pdf_path, model)
            
            # Detailed analysis of returned embeddings
            logger.info(f"Embeddings result for {pdf}: type={type(embeddings)}, length={len(embeddings) if hasattr(embeddings, '__len__') else 'N/A'}")
            print(f"🔍 Analyzing returned embeddings...")
            print(f"   Type of embeddings: {type(embeddings)}")
            
            if embeddings is None:
                logger.info(f"No embeddings returned for {pdf} - result is None")
                print(f"❌ Embeddings is None")
                return None, is_transcript
            elif isinstance(embeddings, list):
                print(f"📋 Embeddings is a list with {len(embeddings)} items")
                if len(embeddings) > 0:
                    print(f"   First item type: {type(embeddings[0])}")
                    if hasattr(embeddings[0], '__len__'):
                        try:
                            logger.info(f"Successfully generated {len(embeddings)} embeddings for {pdf}, first item length: {len(embeddings[0])}")
                            print(f"   First item length: {len(embeddings[0])}")
                        except:
                            logger.info(f"Successfully generated {len(embeddings)} embeddings for {pdf}, could not get first item length")
                            print(f"   Could not get length of first item")
                else:
                    logger.info(f"Empty embeddings list returned for {pdf}")
                    print(f"⚠️  Embeddings list is empty")
                return embeddings, is_transcript
            else:
                logger.info(f"Unexpected embeddings type for {pdf}: {type(embeddings)}")
                print(f"⚠️  Embeddings is not a list: {type(embeddings)}")
                return embeddings, is_transcript
                
        except Exception as e:
            logger.info(f"Exception processing {pdf}: {str(e)} (Type: {type(e).__name__})")
            print(f"❌ Exception occurred: {str(e)}")
            print(f"   Exception type: {type(e).__name__}")
            return None, is_transcript

    # Process each PDF sequentially
    print(f"\n{'='*50}")
    print(f"PROCESSING FILES")
    print(f"{'='*50}")
    
    for i, pdf in enumerate(pdf_files, 1):
        print(f"\n🔄 Processing file {i}/{len(pdf_files)}")
        
        embeddings, is_transcript = process_single_pdf(pdf)
        
        # Update stats
        if is_transcript:
            processing_stats['transcript_files'] += 1
        else:
            processing_stats['non_transcript_files'] += 1
        
        # Process results
        if embeddings is None:
            logger.info(f"No embeddings returned for {pdf}")
            print(f"❌ No embeddings returned for {pdf}")
            processing_stats['failed'] += 1
        elif isinstance(embeddings, list) and len(embeddings) == 0:
            logger.info(f"Empty embeddings list returned for {pdf}")
            print(f"⚠️  Empty embeddings list returned for {pdf}")
            processing_stats['empty_results'] += 1
        elif embeddings:  # Non-empty, non-None result
            embeddings_count = len(embeddings) if isinstance(embeddings, list) else 1
            logger.info(f"Successfully processed {pdf}: {embeddings_count} embeddings, type: {'transcript' if is_transcript else 'non-transcript'}")
            print(f"✅ Successfully got {embeddings_count} embeddings from {pdf}")
            
            if is_transcript:
                transcript_embeddings.extend(embeddings)
                logger.info(f"Added {embeddings_count} transcript embeddings (total transcript: {len(transcript_embeddings)})")
                print(f"📊 Added to transcript embeddings (now {len(transcript_embeddings)} total)")
            else:
                non_transcript_embeddings.extend(embeddings)
                logger.info(f"Added {embeddings_count} non-transcript embeddings (total non-transcript: {len(non_transcript_embeddings)})")
                print(f"📊 Added to non-transcript embeddings (now {len(non_transcript_embeddings)} total)")
            
            processing_stats['successful'] += 1
        else:
            logger.info(f"Unexpected result type for {pdf}: {type(embeddings)}")
            print(f"❓ Unexpected result type for {pdf}: {type(embeddings)}")
            processing_stats['failed'] += 1

    # Final summary
    print(f"\n{'='*50}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*50}")
    print(f"📊 PROCESSING STATISTICS:")
    print(f"   Total files: {processing_stats['total_files']}")
    print(f"   Successful: {processing_stats['successful']}")
    print(f"   Failed: {processing_stats['failed']}")
    print(f"   Empty results: {processing_stats['empty_results']}")
    print(f"   Transcript files: {processing_stats['transcript_files']}")
    print(f"   Non-transcript files: {processing_stats['non_transcript_files']}")
    print(f"\n📈 EMBEDDING RESULTS:")
    print(f"   Total transcript embeddings: {len(transcript_embeddings)}")
    print(f"   Total non-transcript embeddings: {len(non_transcript_embeddings)}")
    print(f"   Grand total embeddings: {len(transcript_embeddings) + len(non_transcript_embeddings)}")
    
    # Log final results
    logger.info(f"Processing complete - Total files: {processing_stats['total_files']}, Successful: {processing_stats['successful']}, Failed: {processing_stats['failed']}, Empty: {processing_stats['empty_results']}")
    logger.info(f"Final results - Transcript embeddings: {len(transcript_embeddings)}, Non-transcript embeddings: {len(non_transcript_embeddings)}, Total: {len(transcript_embeddings) + len(non_transcript_embeddings)}")
    
    return transcript_embeddings, non_transcript_embeddings