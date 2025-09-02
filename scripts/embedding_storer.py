from db_connection import get_db_connection, release_db_connection
from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from logger_setup import get_logger
import os
import hashlib
from excel_processor import process_all_excel
from word_processor import process_doc_with_embeddings
from concurrent.futures import ThreadPoolExecutor
from env_config import TABLE_NAMES, EMBEDDING_MODEL
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_aicore_credentials
from pdf_processor import process_pdf_with_embeddings
import json
import requests

logger = get_logger()

# Initialize AIC Credentials
logger.info("====> embeddings_storer.py -> AIC CREDENTIALS <====")

vcap_services = os.environ.get("VCAP_SERVICES")
destination_service_credentials = get_destination_service_credentials(vcap_services)

try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
except requests.exceptions.HTTPError as e:
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise

AIC_CREDENTIALS = None
dest_AIC = "GENAI_AI_CORE"
aicore_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
logger.info("AIC Credential", AIC_CREDENTIALS)

def get_existing_file_info_from_db():
    """Retrieve unique file info from three HANA tables."""
    logger.info("Fetching existing file info from all tables")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"""
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['transcript']}
        WHERE VEC_META IS NOT NULL
        UNION ALL
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['non_transcript']}
        WHERE VEC_META IS NOT NULL
        UNION ALL
        SELECT 
            DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {TABLE_NAMES['excel_non_transcript']}
        WHERE VEC_META IS NOT NULL
        """
        cursor.execute(query)
        file_info = {row[0]: row[1] for row in cursor.fetchall()}
        logger.info(f"Found {len(file_info)} unique files across all tables")
        return file_info
    except Exception as e:
        logger.info(f"Error fetching file info from database: {e}")
        return {}
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)

def compute_file_hash(file_path):
    """Calculate MD5 hash of a file's content."""
    logger.debug(f"Computing hash for file: {file_path}")
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    file_hash = hash_md5.hexdigest()
    logger.debug(f"Hash computed: {file_hash}")
    return file_hash

def store_embeddings(vector_store, texts, embeddings, metadatas):
    """Store embeddings and metadata in the HANA database."""
    logger.info(f"Storing {len(embeddings)} embeddings in HANA DB table {vector_store.table_name}")

    def validate_metadata(meta):
        """Validate and clean metadata dictionary."""
        source_file = meta.get("source_file", "unknown")
        content_hash = meta.get("content_hash", "unknown")
        page = meta.get("page", 0)

        if not isinstance(source_file, str):
            logger.info(f"source_file is not a string, converting: {source_file} (type: {type(source_file)})")
            source_file = str(source_file)
        if not isinstance(content_hash, str):
            logger.info(f"content_hash is not a string, converting: {content_hash} (type: {type(content_hash)})")
            content_hash = str(content_hash)
        if not isinstance(page, int):
            try:
                page = int(page)
            except Exception:
                logger.info(f"page is not an integer, defaulting to 0: {page} (type: {type(page)})")
                page = 0

        return {"source_file": source_file, "content_hash": content_hash, "page": page}

    # Validate inputs
    if not texts or not embeddings or not metadatas:
        logger.info("Empty inputs provided to store_embeddings")
        return
    
    if len(texts) != len(embeddings) or len(texts) != len(metadatas):
        logger.info(f"Mismatched input lengths: texts={len(texts)}, embeddings={len(embeddings)}, metadatas={len(metadatas)}")
        return

    filtered_metadatas = []
    filtered_texts = []
    filtered_embeddings = []
    
    for i, meta in enumerate(metadatas):
        try:
            logger.debug(f"Processing metadata at index {i}: {meta}")
            clean_meta = validate_metadata(meta)
            filtered_metadatas.append(clean_meta)
            filtered_texts.append(texts[i])
            filtered_embeddings.append(embeddings[i])
        except Exception as e:
            logger.info(f"Skipping metadata at index {i} due to validation error: {e}. Metadata: {meta}")
            continue
    
    if not filtered_texts:
        logger.warning("No valid embeddings to store after validation. Skipping DB insert.")
        return
    
    try:
        logger.info(f"About to store {len(filtered_embeddings)} embeddings in {vector_store.table_name}")
        logger.debug(f"Sample text: {filtered_texts[0] if filtered_texts else 'None'}")
        logger.debug(f"Sample metadata: {filtered_metadatas[0] if filtered_metadatas else 'None'}")
        logger.debug(f"Sample embedding shape: {len(filtered_embeddings[0]) if filtered_embeddings else 'None'}")
        
        # CRITICAL FIX: Convert embeddings to proper format if needed
        processed_embeddings = []
        for emb in filtered_embeddings:
            if hasattr(emb, 'tolist'):  # numpy array
                processed_embeddings.append(emb.tolist())
            elif isinstance(emb, list):
                processed_embeddings.append(emb)
            else:
                logger.info(f"Unexpected embedding type: {type(emb)}")
                processed_embeddings.append(list(emb))
        
        vector_store.add_texts(
            texts=filtered_texts,
            embeddings=processed_embeddings,
            metadatas=filtered_metadatas
        )
        logger.info(f"✅ Successfully stored {len(filtered_embeddings)} embeddings in {vector_store.table_name}")
        
        # Verify insertion by counting records
        conn = vector_store.connection
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {vector_store.table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"Table {vector_store.table_name} now contains {count} total records")
        cursor.close()
        
    except Exception as e:
        logger.info(f"❌ Error storing embeddings in {vector_store.table_name}: {e}", exc_info=True)
        # Add more detailed error information
        logger.info(f"Error details - texts count: {len(filtered_texts)}, embeddings count: {len(filtered_embeddings)}")
        raise  # Re-raise to see the full error

def delete_embeddings_for_file(table_name, source_file):
    """Remove embeddings for a specific file from a table."""
    logger.info(f"Deleting embeddings for {source_file} from {table_name}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        query = f"""
        DELETE FROM {table_name}
        WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) = ?
        """
        cursor.execute(query, (source_file,))
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️ Deleted {deleted_count} embeddings for {source_file} from {table_name}")
        return deleted_count
    except Exception as e:
        logger.info(f"❌ Error deleting embeddings for {source_file} from {table_name}: {e}")
        if conn:
            conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)

def remove_duplicates(table_name):
    """Eliminate duplicate entries in a specified table."""
    logger.info(f"Removing duplicate entries from vector store table: {table_name}")
    conn = None
    cursor = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        select_query = f"""
        SELECT 
            VEC_TEXT, 
            VEC_VECTOR, 
            VEC_META,
            JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file,
            JSON_VALUE(VEC_META, '$.page' RETURNING NVARCHAR(5000)) as page,
            JSON_VALUE(VEC_META, '$.content_hash' RETURNING NVARCHAR(32)) as content_hash
        FROM {table_name}
        """
        cursor.execute(select_query)
        all_records = cursor.fetchall()
        logger.info(f"Retrieved {len(all_records)} total records from {table_name}")
        
        # Early return if no records to deduplicate
        if not all_records:
            logger.info(f"No records to deduplicate in {table_name}")
            return 0
        
        # Proceed with deduplication logic
        unique_records = {}
        for record in all_records:
            vec_text, vec_vector, vec_meta, source_file, page, content_hash = record
            source_file = source_file or "unknown"
            page = int(page) if page and page.isdigit() else 0
            content_hash = content_hash or "unknown"
            key = (str(vec_text), source_file, page, content_hash)
            if key not in unique_records:
                unique_records[key] = (vec_text, vec_vector, vec_meta)
        
        records_removed = len(all_records) - len(unique_records)
        
        # Skip truncate/re-insert if no duplicates found
        if records_removed == 0:
            logger.info(f"No duplicates found in {table_name}. Skipping truncate/re-insert operation.")
            return 0
        
        logger.info(f"Found {records_removed} duplicates. Proceeding with cleanup.")
        logger.info(f"Truncating table {table_name}")
        cursor.execute(f"TRUNCATE TABLE {table_name}")
        
        logger.info(f"Inserting {len(unique_records)} unique records into {table_name}")
        insert_query = f"""
        INSERT INTO {table_name} (VEC_TEXT, VEC_VECTOR, VEC_META)
        VALUES (?, ?, ?)
        """
        for vec_text, vec_vector, vec_meta in unique_records.values():
            cursor.execute(insert_query, (vec_text, vec_vector, vec_meta))
        
        conn.commit()
        logger.info(f"🧹 Removed {records_removed} duplicates from {table_name}")
        return records_removed
        
    except Exception as e:
        logger.error(f"❌ Error removing duplicates from {table_name}: {e}", exc_info=True)
        if conn:
            conn.rollback()
        return 0
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)


# Step 9.1: Process and store embeddings from a single directory
def process_and_store_embeddings(directory_path, force_overwrite_files=None, model_name=EMBEDDING_MODEL):
    """Process files from a single directory and store embeddings based on file type."""
    logger.info("EmbStorer: 9.1 - Starting process_and_store_embeddings")
    logger.info(f"Processing files from {directory_path} with model: {model_name}")
    
    # Step 9.1: Validate directory and initialize variables
    logger.info("EmbStorer: 9.1 - Validating directory and initializing variables")
    if force_overwrite_files is None:
        force_overwrite_files = set()

    if not os.path.exists(directory_path):
        logger.info(f"Directory {directory_path} does not exist")
        return

    logger.debug(f"Directory path received: {directory_path}")
    files_in_dir = os.listdir(directory_path)
    logger.debug(f"Files in directory: {files_in_dir}")

    if not files_in_dir:
        logger.warning(f"No files found in directory: {directory_path}")
        return

    existing_file_info = get_existing_file_info_from_db()

    # Step 9.2: Categorize files
    logger.info("EmbStorer: 9.2 - Categorizing files")
    pdf_files_info = {}
    pdf_files_to_process = set()
    excel_files_info = {}
    excel_files_to_process = set()
    doc_files_info = {}
    doc_files_to_process = set()    

    for f in files_in_dir:
        if not isinstance(f, str):
            logger.warning(f"Skipping non-string filename in directory: {f} (type: {type(f)})")
            continue

        file_path = os.path.join(directory_path, f)
        if not os.path.isfile(file_path):
            logger.debug(f"Skipping non-file: {file_path}")
            continue

        current_hash = compute_file_hash(file_path)

        if f.lower().endswith('.pdf'):
            pdf_files_info[f] = current_hash
            if f not in existing_file_info:
                logger.info(f"New PDF file detected: {f}")
                pdf_files_to_process.add(f)
            elif existing_file_info[f] != current_hash:
                logger.info(f"Content changed for PDF {f}: old hash {existing_file_info[f]}, new hash {current_hash}")
                pdf_files_to_process.add(f)
            elif f in force_overwrite_files:
                logger.info(f"Forced overwrite requested for PDF {f}")
                pdf_files_to_process.add(f)

        elif f.lower().endswith(('.docx', '.doc')):
            doc_files_info[f] = current_hash
            if f not in existing_file_info:
                logger.info(f"New doc file detected: {f}")
                doc_files_to_process.add(f)
            elif existing_file_info[f] != current_hash:
                logger.info(f"Content changed for word {f}: old hash {existing_file_info[f]}, new hash {current_hash}")
            elif f in force_overwrite_files:
                logger.info(f"Forced overwrite requested for Word {f}")
                doc_files_to_process.add(f)
        elif f.lower().endswith(('.xlsx', '.xls')):
            excel_files_info[f] = current_hash
            if f not in existing_file_info:
                logger.info(f"New Excel file detected: {f}")
                excel_files_to_process.add(f)
            elif existing_file_info[f] != current_hash:
                logger.info(f"Content changed for Excel {f}: old hash {existing_file_info[f]}, new hash {current_hash}")
                excel_files_to_process.add(f)
            elif f in force_overwrite_files:
                logger.info(f"Forced overwrite requested for Excel {f}")
                excel_files_to_process.add(f)

    logger.info(f"Found {len(pdf_files_info)} PDF files, {len(pdf_files_to_process)} need processing")
    logger.info(f"Found {len(excel_files_info)} Excel files, {len(excel_files_to_process)} need processing")
    logger.info(f"Found {len(doc_files_info)} Doc files, {len(doc_files_to_process)} need processing")

    if not pdf_files_to_process and not excel_files_to_process and not doc_files_to_process:
        logger.info("No new or changed files to process")
        return

    # Step 9.3: Initialize embedding model
    logger.info("EmbStorer: 9.3 - Initializing embedding model")
    try:
        from gen_ai_hub.proxy import GenAIHubProxyClient
        logger.info(f"AIC Credentials: {json.dumps(AIC_CREDENTIALS)}")

        proxy_client = GenAIHubProxyClient(
            base_url=AIC_CREDENTIALS['aic_base_url'],
            auth_url=AIC_CREDENTIALS['aic_auth_url'],
            client_id=AIC_CREDENTIALS['clientid'],
            client_secret=AIC_CREDENTIALS['clientsecret'],
            resource_group=AIC_CREDENTIALS['resource_group']
        )

        embedding_model = init_embedding_model(model_name=EMBEDDING_MODEL, proxy_client=proxy_client)
        logger.info("Embedding model initialized successfully")
    except Exception as e:
        logger.info(f"Failed to initialize embedding model: {e}")
        raise

    # Process files
    def process_pdf_task():
        logger.info("EmbStorer: 9.19 - Starting PDF processing task")
        transcript_embeddings = []
        non_transcript_embeddings = []
        successful_files = 0
        failed_files = 0

        def process_single_pdf(pdf):
            pdf_path = os.path.join(directory_path, pdf)
            logger.info(f"Processing PDF: {pdf}")
            print(f"Processing: {pdf}")
            filename = os.path.basename(pdf_path)
            is_transcript = 'transcript' in filename.lower()
            try:
                embeddings = process_pdf_with_embeddings(pdf_path, model_name)
                if embeddings is None:
                    print(f"  WARNING: process_pdf_with_embeddings returned None for {pdf}")
                    return None, is_transcript
                elif not embeddings:
                    print(f"  WARNING: process_pdf_with_embeddings returned empty list for {pdf}")
                    return [], is_transcript
                else:
                    print(f"  SUCCESS: Got {len(embeddings)} embeddings from {pdf}")
                    return embeddings, is_transcript
            except Exception as e:
                logger.info(f"Error processing {pdf}: {str(e)}")
                print(f"  ERROR processing {pdf}: {str(e)}")
                return None, is_transcript

        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf') and f in pdf_files_to_process]
        if not pdf_files:
            print(f"No PDFs in {directory_path} to process")
            return [], []

        print(f"Found {len(pdf_files)} PDFs to process")

        for pdf in pdf_files:
            embeddings, is_transcript = process_single_pdf(pdf)
            if embeddings is not None and len(embeddings) > 0:
                if is_transcript:
                    transcript_embeddings.extend(embeddings)
                    print(f"  Added {len(embeddings)} transcript embeddings")
                else:
                    non_transcript_embeddings.extend(embeddings)
                    print(f"  Added {len(embeddings)} non-transcript embeddings")
                successful_files += 1
            else:
                failed_files += 1
                print(f"  Skipped {pdf} (no embeddings returned)")

        print(f"\n=== PROCESSING SUMMARY ===")
        print(f"Files processed successfully: {successful_files}")
        print(f"Files failed/skipped: {failed_files}")
        print(f"Total transcript embeddings: {len(transcript_embeddings)}")
        print(f"Total non-transcript embeddings: {len(non_transcript_embeddings)}")
        print(f"Grand total embeddings: {len(transcript_embeddings) + len(non_transcript_embeddings)}")

        logger.info("PDF processing task completed")
        logger.info(f"Filtered to {len(transcript_embeddings)} PDF transcript embeddings and "
                    f"{len(non_transcript_embeddings)} PDF non-transcript embeddings for processing")
        return transcript_embeddings, non_transcript_embeddings

######### WORD DOCUMENT PROCESSING ############

# Process word in parallel

    def process_doc_task():
        logger.info("EmbStorer: 9.20 - Starting DOC processing task")
        transcript_embeddings = []
        non_transcript_embeddings = []
        successful_files = 0
        failed_files = 0

        def process_single_doc(doc):
            doc_path = os.path.join(directory_path, doc)
            logger.info(f"Processing DOC: {doc}")
            print(f"Processing: {doc}")
            filename = os.path.basename(doc_path)
            is_transcript = 'transcript' in filename.lower()
            try:
                embeddings = process_doc_with_embeddings(doc_path, model_name)
                if embeddings is None:
                    print(f"  WARNING: process_doc_with_embeddings returned None for {doc}")
                    return None, is_transcript
                elif not embeddings:
                    print(f"  WARNING: process_doc_with_embeddings returned empty list for {doc}")
                    return [], is_transcript
                else:
                    print(f"  SUCCESS: Got {len(embeddings)} embeddings from {doc}")
                    return embeddings, is_transcript
            except Exception as e:
                logger.info(f"Error processing {doc}: {str(e)}")
                print(f"  ERROR processing {doc}: {str(e)}")
                return None, is_transcript

        doc_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.doc', '.docx')) and f in doc_files_to_process]
        if not doc_files:
            print(f"No DOC/DOCX files in {directory_path} to process")
            return [], []

        print(f"Found {len(doc_files)} DOC/DOCX files to process")

        for doc in doc_files:
            embeddings, is_transcript = process_single_doc(doc)
            if embeddings is not None and len(embeddings) > 0:
                if is_transcript:
                    transcript_embeddings.extend(embeddings)
                    print(f"  Added {len(embeddings)} transcript embeddings")
                else:
                    non_transcript_embeddings.extend(embeddings)
                    print(f"  Added {len(embeddings)} non-transcript embeddings")
                successful_files += 1
            else:
                failed_files += 1
                print(f"  Skipped {doc} (no embeddings returned)")

        print(f"\n=== DOC PROCESSING SUMMARY ===")
        print(f"Files processed successfully: {successful_files}")
        print(f"Files failed/skipped: {failed_files}")
        print(f"Total transcript embeddings: {len(transcript_embeddings)}")
        print(f"Total non-transcript embeddings: {len(non_transcript_embeddings)}")
        print(f"Grand total embeddings: {len(transcript_embeddings) + len(non_transcript_embeddings)}")

        logger.info("DOC processing task completed")
        logger.info(f"Filtered to {len(transcript_embeddings)} DOC transcript embeddings and "
                    f"{len(non_transcript_embeddings)} DOC non-transcript embeddings for processing")
        return transcript_embeddings, non_transcript_embeddings


######### END #########

    def process_excel_task():
        logger.info("EmbStorer: 9.21 - Starting Excel processing task")
        if not excel_files_to_process:
            return []
        all_excel_embeddings = process_all_excel(directory_path, model_name)
        filtered_excel_embeddings = [(doc, emb) for doc, emb in all_excel_embeddings 
                                    if doc.metadata.get("source_file") in excel_files_to_process]
        logger.info("Excel processing task completed")
        logger.info(f"Filtered to {len(filtered_excel_embeddings)} Excel embeddings for processing")
        return filtered_excel_embeddings

    # Process files in parallel
    logger.info("EmbStorer: 9.23 - Starting parallel processing of files")

    # SOC Niladri 

    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     future_pdf = executor.submit(process_pdf_task)
    #     future_excel = executor.submit(process_excel_task)
        
    #     filtered_transcript_embeddings, filtered_non_transcript_embeddings = future_pdf.result()
    #     filtered_excel_embeddings = future_excel.result()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future_pdf = executor.submit(process_pdf_task)
        filtered_transcript_embeddings, filtered_non_transcript_embeddings = future_pdf.result()
 
        future_excel = executor.submit(process_excel_task)
        filtered_excel_embeddings = future_excel.result()
        future_doc = executor.submit(process_doc_task)
        filtered_transcript_doc_embeddings, filtered_non_transcript_doc_embeddings = future_doc.result()

# EOC -- Niladri 


    logger.info("Parallel processing of files completed")

    # Get database connection
    logger.info("EmbStorer: 9.25 - Acquiring database connection")
    try:
        conn = get_db_connection()
    except Exception as e:
        logger.info(f"EmbStorer: 9.25 - Failed to acquire database connection: {e}", exc_info=True)
        logger.info("EmbStorer: ERROR - Database connection failed. Embedding process will not continue. This is a critical error that prevents reaching Step 10 in server.py.")
        return
    
    def clean_metadata(meta, file_hash_dict):
        """Clean metadata to ensure proper format"""
        logger.info("EmbStorer: 9.26 - Cleaning metadata")
        cleaned = dict(meta)
        sf = cleaned.get("source_file")
        if isinstance(sf, dict):
            logger.info("Fixing source_file metadata from dict to string")
            logger.info(f"Fixing source_file metadata from dict to string: {sf}")
            cleaned["source_file"] = json.dumps(sf)
        elif not isinstance(sf, str):
            cleaned["source_file"] = str(sf)
        
        # Add content hash
        cleaned["content_hash"] = file_hash_dict.get(cleaned.get("source_file"), "unknown")
        return cleaned

    try:
        # Store PDF transcript embeddings
        if filtered_transcript_embeddings:
            logger.info("EmbStorer: 9.28 - Storing PDF transcript embeddings")
            pdf_transcript_table = TABLE_NAMES['transcript']
            transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=pdf_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            
            # Delete existing embeddings for files being reprocessed
            for source_file in pdf_files_to_process:
                if source_file in existing_file_info or source_file in force_overwrite_files:
                    logger.info("Deleting existing embeddings for PDF")
                    delete_embeddings_for_file(pdf_transcript_table, source_file)
            
            transcript_texts = [doc.page_content for doc, _ in filtered_transcript_embeddings]
            transcript_embeds = [embedding for _, embedding in filtered_transcript_embeddings]
            transcript_metadatas = [
                clean_metadata(doc.metadata, pdf_files_info)
                for doc, _ in filtered_transcript_embeddings
            ]
            
            logger.info(f"About to store {len(transcript_texts)} transcript embeddings")
            store_embeddings(transcript_store, transcript_texts, transcript_embeds, transcript_metadatas)

        # Store PDF non-transcript embeddings
        if filtered_non_transcript_embeddings:
            logger.info("EmbStorer: 9.30 - Storing PDF non-transcript embeddings")
            pdf_non_transcript_table = TABLE_NAMES['non_transcript']
            non_transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=pdf_non_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            
            # Delete existing embeddings for files being reprocessed
            for source_file in pdf_files_to_process:
                if source_file in existing_file_info or source_file in force_overwrite_files:
                    logger.info("Deleting existing embeddings for PDF non-transcript")
                    delete_embeddings_for_file(pdf_non_transcript_table, source_file)
            
            non_transcript_texts = [doc.page_content for doc, _ in filtered_non_transcript_embeddings]
            non_transcript_embeds = [embedding for _, embedding in filtered_non_transcript_embeddings]
            non_transcript_metadatas = [
                clean_metadata(doc.metadata, pdf_files_info)
                for doc, _ in filtered_non_transcript_embeddings
            ]
            
            logger.info(f"About to store {len(non_transcript_texts)} non-transcript embeddings")
            store_embeddings(non_transcript_store, non_transcript_texts, non_transcript_embeds, non_transcript_metadatas)
    ############# DOCS PROCESSING ################
        doc_transcript_table = TABLE_NAMES['transcript']
        if filtered_transcript_doc_embeddings:
            transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=doc_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            for source_file in doc_files_to_process:
                if source_file in existing_file_info or source_file in force_overwrite_files:
                    delete_embeddings_for_file(doc_transcript_table, source_file)
            transcript_doc_texts = [doc.page_content for doc, _ in filtered_transcript_doc_embeddings]
            transcript_doc_embeds = [embedding for _, embedding in filtered_transcript_doc_embeddings]
            transcript_doc_metadatas = [
                {**doc.metadata, "content_hash": doc_files_info.get(doc.metadata.get("source_file"))}
                for doc, _ in filtered_transcript_doc_embeddings
            ]
            if transcript_doc_texts:
                store_embeddings(transcript_store, transcript_doc_texts, transcript_doc_embeds, transcript_doc_metadatas)
        # Store Docs non-transcript embeddings
        doc_non_transcript_table = TABLE_NAMES['non_transcript']
        if filtered_non_transcript_doc_embeddings:
            non_transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=doc_non_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            for source_file in doc_files_to_process:
                if source_file in existing_file_info or source_file in force_overwrite_files:
                    delete_embeddings_for_file(doc_non_transcript_table, source_file)
            non_transcript_doc_texts = [doc.page_content for doc, _ in filtered_non_transcript_doc_embeddings]
            non_transcript_doc_embeds = [embedding for _, embedding in filtered_non_transcript_doc_embeddings]
            non_transcript_doc_metadatas = [
                {**doc.metadata, "content_hash": doc_files_info.get(doc.metadata.get("source_file"))}
                for doc, _ in filtered_non_transcript_doc_embeddings
            ]
            if non_transcript_doc_texts:
                store_embeddings(non_transcript_store, non_transcript_doc_texts, non_transcript_doc_embeds, non_transcript_doc_metadatas)

        # Store Excel embeddings
        if filtered_excel_embeddings:
            logger.info("EmbStorer: 9.32 - Storing Excel embeddings")
            excel_non_transcript_table = TABLE_NAMES['excel_non_transcript']
            excel_non_transcript_store = HanaDB(
                connection=conn,
                embedding=embedding_model,
                table_name=excel_non_transcript_table,
                content_column="VEC_TEXT",
                metadata_column="VEC_META",
                vector_column="VEC_VECTOR"
            )
            
            # Delete existing embeddings for files being reprocessed
            for source_file in excel_files_to_process:
                if source_file in existing_file_info or source_file in force_overwrite_files:
                    logger.info("Deleting existing embeddings for Excel")
                    delete_embeddings_for_file(excel_non_transcript_table, source_file)
            
            excel_non_transcript_texts = [doc.page_content for doc, _ in filtered_excel_embeddings]
            excel_non_transcript_embeds = [embedding for _, embedding in filtered_excel_embeddings]
            excel_non_transcript_metadatas = [
                clean_metadata(doc.metadata, excel_files_info)
                for doc, _ in filtered_excel_embeddings
            ]
            
            logger.info(f"About to store {len(excel_non_transcript_texts)} Excel embeddings")
            store_embeddings(excel_non_transcript_store, excel_non_transcript_texts, excel_non_transcript_embeds, excel_non_transcript_metadatas)

    except Exception as e:
        logger.info(f"EmbStorer: ERROR - Exception during embedding storage: {e}", exc_info=True)
        logger.info("EmbStorer: ERROR - Embedding storage failed. This may prevent reaching Step 10 in server.py.")
        return
    finally:
        logger.info("EmbStorer: 9.34 - Releasing database connection")
        try:
            release_db_connection(conn)
        except Exception as e:
            logger.info(f"EmbStorer: ERROR - Failed to release database connection: {e}", exc_info=True)

    # Remove duplicates
    logger.info("EmbStorer: 9.35 - Starting duplicate removal process")
    try:
        pdf_transcript_removed = remove_duplicates(TABLE_NAMES['transcript'])
        pdf_non_transcript_removed = remove_duplicates(TABLE_NAMES['non_transcript'])
        excel_non_transcript_removed = remove_duplicates(TABLE_NAMES['excel_non_transcript'])      
        total_removed = pdf_transcript_removed + pdf_non_transcript_removed + excel_non_transcript_removed
        logger.info(f"Total duplicates removed: {total_removed}")
        logger.info("EmbStorer: 9.37 - Embedding storage process completed successfully")
    except Exception as e:
        logger.info(f"EmbStorer: ERROR - Exception during duplicate removal: {e}", exc_info=True)
        logger.info("EmbStorer: ERROR - Duplicate removal failed. This may prevent reaching Step 10 in server.py.")
        return