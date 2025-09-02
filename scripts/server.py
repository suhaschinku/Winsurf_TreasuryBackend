import os
import re
import time
import logging
import traceback
from logging.handlers import RotatingFileHandler
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from markupsafe import Markup
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from env_config import DEF_SCHEMA
from hdbcli import dbapi  # SAP HANA client library
import requests
from db_connection import get_db_connection, load_vector_stores, cleanup_managers, get_all_file_names, get_all_file_ids,delete_file_from_all_tables,retrieve_metadata_as_json
from query_processor import process_query
from env_config import get_documents_dir_path
from embedding_storer import process_and_store_embeddings
from gen_ai_hub.proxy.native.amazon.clients import Session
from Intellibase.finalresponse import combine_function_results, generate_aika_prompt
from Documents_Summary.summary import create_text_summary, retrieve_summary_blobs
import pdfplumber
from Intellibase.DANS_Upload import insert_excel_to_hana
from Intellibase.Data_Dictionary import process_dictionary_sheet
from llm_client import execute_aika_analysis
from file_name_extractor import extract_filename_from_input
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials
from xsuaa_srv import get_xsuaa_credentials, verify_jwt_token, require_auth
from logger_setup import get_logger
from content_scanner_orchestrator import SimpleDocumentProcessor, cleanup_processing_files
from Dublin_Core.metadata_fetch import get_metadata_by_filename


logger = get_logger()
# Load environment variables
load_dotenv()

# Define improved error handling system
class ErrorCategory:
    """Enum-like class to categorize errors for frontend interpretation"""
    INPUT_VALIDATION = "input_validation"
    SECURITY = "security"
    FILE_NOT_FOUND = "file_not_found"
    FILE_ACCESS = "file_access"
    FILE_FORMAT = "file_format"
    RATE_LIMIT = "rate_limit"
    DATABASE = "database"
    PROCESSING = "processing"
    INTERNAL = "internal"
    METHOD_NOT_ALLOWED = "method_not_allowed"

class AppError(Exception):
    """Enhanced application error with standardized structure"""
    def __init__(self, error_type, message, user_friendly=True, status_code=400, details=None):
        super().__init__(message)
        self.error_type = error_type
        self.user_friendly = user_friendly
        self.status_code = status_code
        self.details = details or {}
    
    def to_dict(self):
        """Convert error to standardized dictionary format"""
        error_dict = {
            "error": True,
            "error_type": self.error_type,
            "message": str(self) if self.user_friendly else "An unexpected error occurred. Please try again later.",
            "status_code": self.status_code
        }
        if self.details and self.user_friendly:
            error_dict["details"] = self.details
        return error_dict

# Initialize Flask app
app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# Configure CORS
CORS(app)

# Detect environment
IS_PRODUCTION = 'VCAP_SERVICES' in os.environ
if IS_PRODUCTION:
    logger.info("Running in production mode")
else:
    logger.info("Running in Local mode")

# Set base path and directories
LOCALPATH = os.getenv('LOCALPATH', os.getcwd())
logger.info(f"Base path set: {LOCALPATH}")
base_path = os.getenv('LOCALPATH', os.path.abspath(os.getcwd()))
logger.info(f"Base path configured: {base_path}")
documents_dir = os.path.abspath(os.path.join(base_path, "Documents"))
logger.info(f"Document directory set: {documents_dir}")
images_dir = os.path.join(base_path, "Images")
logger.info(f"Image directory set: {images_dir}")
logs_dir = os.path.join(base_path, "logs")
logger.info(f"Logs directory set: {logs_dir}")
documents_summary_dir = os.path.join(base_path, "Documents_Summary")
logger.info(f"Documents summary directory set: {documents_summary_dir}")
dictionary_dir = os.path.abspath(os.path.join(base_path, "Intellibase"))
dictionary_path = os.path.join(dictionary_dir, "Data_Dictionary.txt")

# Ensure directories exist
for directory in [documents_dir, images_dir, logs_dir, documents_summary_dir]:
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Ensured directory exists: {directory}")

def clean_folders():
    """Remove all files from Documents folders."""
    folders = [documents_dir]
    
    for folder in folders:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleaned folder: {folder}")
        else:
            print(f"Folder not found: {folder}")


# Configure logging with rotation
log_file_path = os.path.join(logs_dir, "TreasuryAnalysis.log")
logger = logging.getLogger('TreasuryAnalysis')
logger.setLevel(logging.INFO)
handler = RotatingFileHandler(log_file_path, maxBytes=50 * 1024 * 1024, backupCount=5)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info("Logging configured with rotation")

# Rate limiter
limiter = Limiter(app=app, key_func=get_remote_address)
logger.info("Rate limiter initialized")

# Configuration
ALLOWED_EXTENSIONS = {'.pdf', '.xlsx', '.txt', '.jpg', '.png', '.jpeg','.docx','.doc'}
ALLOWED_SUMMARY_EXTENSIONS = {'.pdf', '.xlsx', '.txt','.docx','.doc'}
IMAGE_EXTENSIONS = {'.jpg', '.png', '.jpeg'}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
UPLOAD_LIMIT = 500
upload_counts = {}

# ---------------------------- XSUAA Authentication Setup ----------------------------
"""
XSUAA authentication is enforced on protected endpoints using the @require_auth decorator.
- The XSUAA credentials are loaded from VCAP_SERVICES and stored in the Flask app context as 'uaa_xsuaa_credentials'.
- The decorator (from xsuaa_srv.py) checks for a Bearer token in the Authorization header and validates it using the credentials.
- If the token is missing, invalid, or lacks the required scope, a 401/403 error is returned.
"""
vcap_services = os.environ.get("VCAP_SERVICES")
uaa_xsuaa_credentials = get_xsuaa_credentials(vcap_services)
logger.info(f"XSUAA credentials loaded: {uaa_xsuaa_credentials}")
app.uaa_xsuaa_credentials = uaa_xsuaa_credentials

# ---------------------------- LOAD CF VCAP_SERVICES Variables -----------------------------
logger.info("Loading HANA and AIC credentials from destination services")
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination service credentials loaded: {destination_service_credentials}")
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully")
except requests.exceptions.HTTPError as e:
    if e.response is not None and e.response.status_code == 500:
        raise Exception("HTTP 500: Check if the client secret is correct.") from e
    else:
        raise

logger.info(f"OAuth token: {oauth_token}")

# -------------------------------- READ HANA DB Configuration -------------------------------------
dest_HDB = 'GENAI_HDB'
hana_dest_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    name=dest_HDB,
    token=oauth_token
)
logger.info(f"HANA destination details fetched: {hana_dest_details}")

HANA_CONN = None
GV_HANA_CREDENTIALS = None

def initialize_hana_connection():
    """Initialize HANA DB connection using extracted credentials"""
    global HANA_CONN, GV_HANA_CREDENTIALS
    logger.info("Initializing HANA database connection")
    GV_HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f"HANA credentials extracted: {GV_HANA_CREDENTIALS}")
    try:
        HANA_CONN = dbapi.connect(
            address=GV_HANA_CREDENTIALS['address'],
            port=GV_HANA_CREDENTIALS['port'],
            user=GV_HANA_CREDENTIALS['user'],
            password=GV_HANA_CREDENTIALS['password'],
            encrypt=True,
            sslValidateCertificate=False
        )
        logger.info("HANA database connection established successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing HANA connection: {str(e)}")
        return False

initialize_hana_connection()

def store_metadata_in_hana(filename, file_path, file_type, upload_time):
    """Store file metadata in HANA database"""
    logger.info(f"Storing metadata for file: {filename}")
    try:
        if not HANA_CONN:
            logger.warning("HANA connection not initialized, skipping metadata storage")
            return False
        cursor = HANA_CONN.cursor()
        cursor.execute(f"SET SCHEMA {os.getenv('DEF_SCHEMA', 'DEFAULT_SCHEMA')}")
        logger.info(f"Schema set to: {os.getenv('DEF_SCHEMA', 'DEFAULT_SCHEMA')}")
        query = """
            INSERT INTO "FILE_METADATA" (filename, file_path, file_type, upload_time)
            VALUES (?, ?, ?, ?)
        """
        cursor.execute(query, (filename, file_path, file_type, upload_time))
        HANA_CONN.commit()
        logger.info(f"Metadata stored successfully for {filename}")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Error storing metadata in HANA: {str(e)}")
        return False

# -------------------------------- READ AIC Configuration -------------------------------------
AIC_CREDENTIALS = None

def initialize_aic_credentials():
    """Initialize AIC credentials from VCAP_SERVICES"""
    global GV_AIC_CREDENTIALS
    logger.info("Initializing AIC credentials")
    try:
        dest_AIC = "GENAI_AI_CORE"
        aicore_details = fetch_destination_details(
            destination_service_credentials['dest_base_url'],
            dest_AIC,
            oauth_token
        )
        logger.info("AIC destination details fetched successfully")
        GV_AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
        logger.info(f"AIC credentials extracted: {GV_AIC_CREDENTIALS}")
        return True
    except Exception as e:
        logger.error(f"Error initializing AIC credentials: {str(e)}")
        return False

initialize_aic_credentials()

# Initialize default vector stores
logger.info("Loading default HANA DB vector stores")
default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store = load_vector_stores(AIC_CREDENTIALS=GV_AIC_CREDENTIALS)
if default_transcript_store is None or default_non_transcript_store is None:
    logger.error("Failed to load default vector stores")
else:
    logger.info("Default vector stores loaded successfully")

def allowed_file(filename):
    """Check if file extension is allowed"""
    logger.info(f"Checking if file is allowed: {filename}")
    is_allowed = os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS
    logger.info(f"File {filename} is {'allowed' if is_allowed else 'not allowed'}")
    return is_allowed

# Log all incoming requests
@app.before_request
def log_request_info():
    logger.info(f"Incoming request: {request.method} {request.url} from {request.remote_addr}")

# Enhanced error handlers
@app.errorhandler(AppError)
def handle_app_error(error):
    """Global error handler for AppError exceptions"""
    logger.info(f"Handling AppError: [{error.error_type}] {str(error)}")
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(404)
def not_found_error(error):
    logger.warning(f"Route not found: {request.path}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.FILE_NOT_FOUND,
        "message": "The requested resource was not found.",
        "status_code": 404
    }), 404

@app.errorhandler(500)
def internal_server_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.INTERNAL,
        "message": "An internal server error occurred. Please try again later.",
        "status_code": 500
    }), 500

@app.errorhandler(429)
def rate_limit_error(error):
    logger.warning(f"Rate limit exceeded: {request.remote_addr}")
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.RATE_LIMIT,
        "message": "Rate limit exceeded. Please slow down your requests.",
        "status_code": 429
    }), 429

@app.errorhandler(405)
def method_not_allowed_error(error):
    logger.warning(f"Method not allowed: {request.method} on {request.path}")
    endpoint = app.url_map._rules_by_endpoint.get(request.endpoint, [{}])[0]
    allowed_methods = endpoint.get('methods', [])
    return jsonify({
        "error": True,
        "error_type": ErrorCategory.METHOD_NOT_ALLOWED,
        "message": f"Method {request.method} not allowed for {request.path}.",
        "status_code": 405,
        "details": {"allowed_methods": allowed_methods}
    }), 405

# Enhanced Input validation
def validate_user_input(user_input):
    """Validate user input for length and security"""
    logger.info(f"Validating user input: '{user_input[:50]}...'")
    try:
        user_input = user_input.strip()
        logger.info("User input stripped")
        if len(user_input) < 3:
            raise AppError(
                ErrorCategory.INPUT_VALIDATION, 
                "Your query is too short. Please provide at least 3 characters.", 
                status_code=400
            )
        if len(user_input) > 500:
            raise AppError(
                ErrorCategory.INPUT_VALIDATION, 
                "Your query is too long. Please limit your input to 500 characters.", 
                status_code=400
            )
        
        sql_patterns = re.compile(
            r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|alternatives|UNION|EXEC|EXECUTE|TRUNCATE|CREATE|RENAME|DESCRIBE|GRANT|REVOKE)\b'
            r'|\b(OR|AND)\s+\d+\s*=\s*\d+|\b(--|#|\/\*|\*\/)|\b(WAITFOR\s+DELAY)\b)',
            re.IGNORECASE
        )
        if sql_patterns.search(user_input):
            raise AppError(
                ErrorCategory.SECURITY, 
                "Your query contains potentially harmful content. Please revise your input and avoid SQL-like syntax.", 
                status_code=400
            )
        logger.info("User input validated successfully")
        return True
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in input validation: {str(e)}")
        raise AppError(
            ErrorCategory.INTERNAL, 
            "We encountered an issue processing your input.", 
            user_friendly=False, 
            status_code=500
        )

# Prompt processing function
def promptlaunchpad(chatquery,user_id):
    """Handle prompt processing with error handling"""
    logger.info(f"Starting prompt processing for query: '{chatquery[:50]}...'")
    try:
        validate_user_input(chatquery)
        logger.info("User input validated for prompt")
        aika_prompt = generate_aika_prompt(chatquery)
        logger.info("AIKA prompt generated")
        chatquery = execute_aika_analysis(aika_prompt)
        logger.info("AIKA analysis executed")
        logger.info(f"Processing query: {chatquery}")
        response = combine_function_results(user_id,chatquery)
        logger.info("Prompt processing completed successfully")
        logger.debug(f"Query processed successfully")
        return response
    except AppError as e:
        error_dict = e.to_dict()
        logger.info(f"Prompt processing failed with AppError: {error_dict['message']}")
        return {
            "analysis_result": f"Error: {error_dict['message']}", 
            "sql_query": "N/A", 
            "error": error_dict
        }
    except Exception as e:
        logger.error(f"Unexpected error in prompt processing: {str(e)}")
        error_dict = AppError(
            ErrorCategory.PROCESSING, 
            "We couldn't process your query at this time.", 
            user_friendly=True, 
            status_code=500
        ).to_dict()
        return {
            "analysis_result": f"Error: {error_dict['message']}", 
            "sql_query": "N/A", 
            "error": error_dict
        }

def get_vector_stores_for_query(user_input):
    """
    Extract filename from user input and load appropriate vector stores.
    Returns tuple of (transcript_store, non_transcript_store, excel_non_transcript_store, extracted_filename)
    """
    logger.info(f"Loading vector stores for query: '{user_input[:50]}...'")
    try:
        extracted_filename = extract_filename_from_input(user_input)
        if extracted_filename:
            if isinstance(extracted_filename, list):
                bank_name = ','.join(extracted_filename)
                logger.info(f"Extracted multiple filenames: {extracted_filename}")
            else:
                bank_name = extracted_filename
                logger.info(f"Extracted filename: {extracted_filename}")
            logger.info(f"Loading filtered vector stores for: {bank_name}")
            transcript_store, non_transcript_store, excel_non_transcript_store = load_vector_stores(bank_name, AIC_CREDENTIALS=GV_AIC_CREDENTIALS)
            if transcript_store is None or non_transcript_store is None or excel_non_transcript_store is None:
                logger.warning(f"Failed to load filtered vector stores for {bank_name}, falling back to default stores")
                return default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store, extracted_filename
            logger.info(f"Filtered vector stores loaded successfully for: {bank_name}")
            return transcript_store, non_transcript_store, excel_non_transcript_store, extracted_filename
        else:
            logger.info("No filename extracted, using default vector stores")
            return default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store, None
    except Exception as e:
        logger.error(f"Error loading vector stores for query: {str(e)}")
        logger.info("Falling back to default vector stores")
        return default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store, None


@app.route('/api/chat', methods=['POST'])
@require_auth
def chat():
    """Process chat queries and return responses"""
    logger.info("Processing chat request")
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            logger.warning("Invalid request data")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "No message provided in your request",
                status_code=400
            )
        raw_message = data.get('message','')
        user_id = None
        user_input = raw_message
        
        # Check if message contains user_id pattern: "user_id: <value> : <actual_message>"
        user_id_pattern = r'^user_id\s*:\s*([^:]+)\s*:\s*(.*)$'
        match = re.match(user_id_pattern, raw_message.strip())
        if match:
            user_id = match.group(1).strip()
            user_input = match.group(2).strip()
            logger.info(f"Extracted user_id: {user_id}")
            logger.info(f"Extracted message: {user_input}")
        else:
            # Fallback to IP address if no user_id is provided
            user_id = request.remote_addr
            logger.info(f"No user_id found in message, using IP address: {user_id}")
        
        # First validate input
        validate_user_input(user_input)
        
        # Check for Research Summary request - ONLY RETRIEVE, DON'T GENERATE
        research_summary_match = re.search(r'Research Summary\s*:\s*([^\n]+)', user_input, re.IGNORECASE)
        if research_summary_match:
            logger.info("Detected 'Research Summary' keyword in prompt")
            filename = research_summary_match.group(1).strip()
            if not filename:
                logger.warning("No filename provided in Research Summary prompt")
                raise AppError(
                    ErrorCategory.INPUT_VALIDATION,
                    "No filename provided for Research Summary. Please specify a file.",
                    status_code=400
                )
            logger.info(f"Checking for existing summaries for file: {filename}")
            
            # Validate file extension
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext not in ALLOWED_SUMMARY_EXTENSIONS:
                logger.warning(f"Invalid file type for summary: {filename}")
                raise AppError(
                    ErrorCategory.FILE_FORMAT,
                    f"Invalid file type for '{filename}'. Only PDF, XLSX, TXT, DOC, and DOCX files are supported for summaries.",
                    status_code=400,
                    details={"filename": filename, "extension": file_ext, "allowed": list(ALLOWED_SUMMARY_EXTENSIONS)}
                )
            
            # Try to retrieve existing summaries
            summaries = retrieve_summary_blobs(filename)
            if summaries:
                logger.info(f"Found existing summaries for {filename}")
                return jsonify({
                    "success": True,
                    "FINAL_RESULT": "Summaries retrieved",
                    "summary_files": summaries
                }), 200
            else:
                # No summaries found - instruct user to use generate-embeddings
                logger.info(f"No existing summaries found for {filename}")
                return jsonify({
                    "success": False,
                    "FINAL_RESULT": f"No summaries found for '{filename}'. Please use the 'Generate Embeddings' feature to create summaries first.",
                    "message": f"Summaries for '{filename}' need to be generated. Please click 'Generate Embeddings' to create them.",
                    "requires_generation": True,
                    "filename": filename
                }), 200
        
        # Process regular queries (non-summary requests)
        transcript_store, non_transcript_store, excel_non_transcript_store, extracted_filename = get_vector_stores_for_query(user_input)
        if extracted_filename:
            if isinstance(extracted_filename, list):
                logger.info(f"Using filtered vector stores for files: {extracted_filename}")
            else:
                logger.info(f"Using filtered vector stores for file: {extracted_filename}")
        else:
            logger.info("Using default vector stores (no specific filename detected)")
        
        # Check for Intellibase or standard Query
        use_intellibase = "intellibase" in user_input.lower()
        logger.info(f"Processing method: {'Intellibase' if use_intellibase else 'Standard query processor'}")
        
        if use_intellibase:
            try:
                logger.info("Processing query with Intellibase")
                result = promptlaunchpad(user_input, user_id)
                if "error" in result:
                    error_data = result["error"]
                    logger.info(f"Intellibase processing failed: {error_data['message']}")
                    return jsonify({
                        "error": True,
                        "error_type": error_data["error_type"],
                        "message": error_data["message"],
                        "FINAL_RESULT": result["analysis_result"],
                        "SQL_QUERY": result["sql_query"]
                    }), error_data["status_code"]
                formatted_response = {
                    "success": True,
                    "FINAL_RESULT": str(result["analysis_result"]),
                    "SQL_QUERY": result["sql_query"]
                }
                if extracted_filename:
                    formatted_response["filtered_by"] = extracted_filename
                logger.info("Intellibase response generated successfully")
                return jsonify(formatted_response), 200
            except Exception as e:
                logger.error(f"Intellibase processing failed: {str(e)}")
                raise AppError(
                    ErrorCategory.PROCESSING,
                    "Intellibase processing failed",
                    status_code=500,
                    details={"error": str(e)}
                )
        else:
            try:
                logger.info("Processing query with standard query processor")
                result = process_query(user_input, transcript_store=transcript_store, 
                                       non_transcript_store=non_transcript_store, 
                                       excel_non_transcript_store=excel_non_transcript_store)
                html_content = Markup(result)
                formatted_response = {
                    "success": True,
                    "FINAL_RESULT": str(html_content)
                }
                if extracted_filename:
                    formatted_response["filtered_by"] = extracted_filename
                logger.info("Standard query response generated successfully")
                return jsonify(formatted_response), 200
            except Exception as e:
                logger.error(f"Standard query processing failed: {str(e)}")
                raise AppError(
                    ErrorCategory.PROCESSING,
                    "Query processing failed",
                    status_code=500,
                    details={"error": str(e)}
                )
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred while processing your request",
            user_friendly=True,
            status_code=500
        )

@app.route('/api/generate-embeddings', methods=['POST'])
@require_auth
def generate_embeddings():
    """Generate embeddings, refresh Intellibase, and create summaries based on available files"""
    logger.info("Processing generate-embeddings request")
    global default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store
    connection = None
    cursor = None
    
    try:
        connection = get_db_connection()
        # Check for DANS_DATASET.xlsx first for Intellibase refresh
        dans_file = os.path.join(documents_dir, "DANS_DATASET.xlsx")
        if os.path.exists(dans_file):
            logger.info("Found DANS_DATASET.xlsx, starting Intellibase data refresh")
            start_time = time.time()
            try:
                #insert_excel_to_hana(dans_file)
                process_dictionary_sheet(dans_file,dictionary_path,connection)
                logger.info("DANS_DATASET.xlsx processed for Intellibase refresh")
            except Exception as e:
                logger.error(f"Error processing DANS_DATASET.xlsx: {str(e)}")
                raise AppError(
                    ErrorCategory.PROCESSING,
                    "Failed to process DANS_DATASET.xlsx.",
                    status_code=500,
                    details={"detail": str(e)}
                )
            cleanup_managers()
            logger.info("Vector store managers cleaned up")
            default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store = load_vector_stores(AIC_CREDENTIALS=GV_AIC_CREDENTIALS)
            logger.info(f"Intellibase data refreshed in {time.time() - start_time:.2f} seconds")
            return jsonify({
                "success": True,
                "message": "Intellibase data refreshed successfully",
                "filename": "DANS_DATASET.xlsx"
            }), 200

        # If no DANS_DATASET.xlsx, proceed with embedding generation and summary creation
        logger.info("No DANS_DATASET.xlsx found, proceeding with embedding generation and summary creation")
        uploaded_files = []
        for directory in [documents_dir, images_dir]:
            files = [f for f in os.listdir(directory) if f.lower().endswith(tuple(ALLOWED_EXTENSIONS))]
            for f in files:
                uploaded_files.append((os.path.join(directory, f), os.path.getmtime(os.path.join(directory, f))))
        
        if not uploaded_files:
            logger.error("No files available to process")
            raise AppError(
                ErrorCategory.FILE_NOT_FOUND,
                "No files available to process. Please upload files first.",
                status_code=400
            )
        
        latest_file = sorted(uploaded_files, key=lambda x: x[1], reverse=True)[0][0]
        latest_filename = os.path.basename(latest_file)
        logger.info(f"Selected latest file for processing: {latest_filename}")
        
        # Determine which directory the latest file is in
        file_directory = os.path.dirname(latest_file)
        logger.info(f"Processing embeddings for directory: {file_directory}")
        
        # Track processing results
        processing_results = {
            "embeddings_processed": False,
            "summaries_created": [],
            "summary_errors": []
        }
        
        # Step 1: Process embeddings
        try:
            force_overwrite_files = []
            # Pass the directory path, not the file path
            process_and_store_embeddings(file_directory, force_overwrite_files)
            logger.info(f"Embeddings processed for directory: {file_directory}")
            processing_results["embeddings_processed"] = True
        except Exception as e:
            logger.error(f"Error processing embeddings in {file_directory}: {str(e)}")
            raise AppError(
                ErrorCategory.PROCESSING,
                "Error processing embeddings",
                status_code=500,
                details={"detail": str(e), "directory": file_directory}
            )
        
        # Step 2: Create summaries for eligible files
        logger.info("Starting summary generation for eligible files")
        summary_eligible_files = []
        
        # Find all files eligible for summary generation
        for directory in [documents_dir]:  # Only check documents_dir for summaries
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) 
                        if os.path.splitext(f)[1].lower() in ALLOWED_SUMMARY_EXTENSIONS]
                for filename in files:
                    file_path = os.path.join(directory, filename)
                    if os.path.isfile(file_path):
                        summary_eligible_files.append(filename)
        
        logger.info(f"Found {len(summary_eligible_files)} files eligible for summary generation: {summary_eligible_files}")
        
        # Generate summaries for each eligible file
        for filename in summary_eligible_files:
            try:
                logger.info(f"Checking/generating summaries for: {filename}")
                
                # Check if summaries already exist
                existing_summaries = retrieve_summary_blobs(filename)
                if existing_summaries:
                    logger.info(f"Summaries already exist for {filename}, skipping generation")
                    processing_results["summaries_created"].append({
                        "filename": filename,
                        "status": "already_exists",
                        "summary_count": len(existing_summaries)
                    })
                    continue
                
                # Generate new summaries
                logger.info(f"Generating new summaries for: {filename}")
                output_paths = create_text_summary(filename)
                logger.info(f"Summaries generated for {filename}: {output_paths}")
                
                # Verify summaries were created and stored
                new_summaries = retrieve_summary_blobs(filename)
                if new_summaries:
                    logger.info(f"Successfully created and retrieved {len(new_summaries)} summaries for {filename}")
                    processing_results["summaries_created"].append({
                        "filename": filename,
                        "status": "newly_created",
                        "summary_count": len(new_summaries),
                        "output_paths": output_paths
                    })
                else:
                    logger.warning(f"Failed to retrieve summaries after generation for {filename}")
                    processing_results["summary_errors"].append({
                        "filename": filename,
                        "error": "Failed to retrieve summaries after generation"
                    })
                    
            except Exception as e:
                logger.error(f"Failed to generate summaries for {filename}: {str(e)}")
                processing_results["summary_errors"].append({
                    "filename": filename,
                    "error": str(e)
                })
        
        # Step 3: Cleanup and reload vector stores
        try:
            cleanup_managers()
            logger.info("Vector store managers cleaned up after processing")
            default_transcript_store, default_non_transcript_store, default_excel_non_transcript_store = load_vector_stores(AIC_CREDENTIALS=GV_AIC_CREDENTIALS)
            logger.info("Default vector stores reloaded after processing")
        except Exception as e:
            logger.error(f"Error during cleanup and reload: {str(e)}")
            # Don't fail the entire operation for cleanup issues
        
        # Prepare response message
        success_message_parts = []
        
        if processing_results["embeddings_processed"]:
            success_message_parts.append("Embeddings processed successfully")
        
        total_summaries = len(processing_results["summaries_created"])
        if total_summaries > 0:
            newly_created = len([s for s in processing_results["summaries_created"] if s["status"] == "newly_created"])
            already_existing = total_summaries - newly_created
            
            if newly_created > 0:
                success_message_parts.append(f"{newly_created} new summaries created")
            if already_existing > 0:
                success_message_parts.append(f"{already_existing} summaries already existed")
        
        if processing_results["summary_errors"]:
            success_message_parts.append(f"{len(processing_results['summary_errors'])} summary generation errors")
        
        final_message = "; ".join(success_message_parts) if success_message_parts else "Processing completed"
        
        response_data = {
            "success": True,
            "message": final_message,
            "latest_file": latest_filename,
            "processed_directory": os.path.basename(file_directory),
            "processing_summary": {
                "embeddings_processed": processing_results["embeddings_processed"],
                "summaries_created": len(processing_results["summaries_created"]),
                "summary_errors": len(processing_results["summary_errors"])
            }
        }
        
        # Add detailed results if there were summary operations
        if processing_results["summaries_created"] or processing_results["summary_errors"]:
            response_data["summary_details"] = {
                "created": processing_results["summaries_created"],
                "errors": processing_results["summary_errors"]
            }
        
        logger.info(f"Generate embeddings completed: {final_message}")
        return jsonify(response_data), 200
        
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error in generate-embeddings endpoint: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred while processing request",
            status_code=500
        )
    finally:
        clean_folders()

@app.route('/api/', methods=['GET'])
def home():
    """Return server status"""
    logger.info("Accessing root status endpoint")
    try:
        if default_transcript_store is None or default_non_transcript_store is None:
            logger.error("Vector stores not loaded")
            return jsonify({"status": "failed", "message": "Server is running but vector stores failed to load"}), 500
        logger.info("Server status check completed")
        return jsonify({"status": "running"}), 200
    except Exception as e:
        logger.error(f"Server status check failed: {str(e)}")
        return jsonify({"status": "failed", "message": f"Server error: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    logger.info("Accessing health check endpoint")
    return jsonify({
        "status": "Server is running",
        "aic_credentials_loaded": GV_AIC_CREDENTIALS is not None,
        "hana_connected": HANA_CONN is not None
    }), 200

@app.route('/api/test-get', methods=['GET'])
def test_get():
    """Test endpoint to verify GET requests"""
    logger.info("Accessing test GET endpoint")
    return jsonify({"success": True, "message": "GET request successful"}), 200

@app.route('/api/debug/routes', methods=['GET'])
def debug_routes():
    """Debug endpoint to list all registered routes"""
    logger.info("Accessing debug routes endpoint")
    routes = [str(rule) for rule in app.url_map.iter_rules()]
    logger.info(f"Retrieved {len(routes)} routes")
    return jsonify({
        "success": True,
        "routes": routes
    }), 200

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get detailed status information"""
    logger.info("Accessing status endpoint")
    status = {
        "server_status": "running",
        "vector_stores": {
            "transcript_store": default_transcript_store is not None,
            "non_transcript_store": default_non_transcript_store is not None,
            "excel_non_transcript_store": default_excel_non_transcript_store is not None
        },
        "aic_configuration": {
            "credentials_loaded": GV_AIC_CREDENTIALS is not None,
            "base_url_configured": GV_AIC_CREDENTIALS['aic_base_url'] is not None,
            "auth_url_configured": GV_AIC_CREDENTIALS['aic_auth_url'] is not None
        },
        "hana_configuration": {
            "connected": HANA_CONN is not None,
            "credentials_loaded": GV_HANA_CREDENTIALS is not None,
        }
    }
    logger.info("Status information retrieved")
    return jsonify(status), 200

@app.route('/api/static/<path:filename>', methods=['GET'])
def serve_static(filename):
    """Serve static files"""
    logger.info(f"Serving static file: {filename}")
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        raise AppError(
            ErrorCategory.FILE_NOT_FOUND,
            f"The requested file '{filename}' could not be found",
            status_code=404
        )

@app.route('/api/get-files', methods=['GET'])
def get_files():
    """Get all file names from vector stores for frontend dropdown"""
    logger.info("Processing get-files request")
    try:
        bank_name = request.args.get('bank_name', None)
        if bank_name:
            bank_name = bank_name.strip()
            if not bank_name:
                bank_name = None
        logger.info(f"Fetching files with bank_name filter: {bank_name if bank_name else 'None'}")
        file_names = get_all_file_names(bank_name)
        if not file_names:
            logger.warning(f"No files found in vector stores with filter: {bank_name}")
            return jsonify({
                "success": True,
                "files": [],
                "message": "No files found in the system",
                "count": 0,
                "bank_filter": bank_name
            }), 200
        logger.info(f"Retrieved {len(file_names)} files from vector stores")
        return jsonify({
            "success": True,
            "files": file_names,
            "count": len(file_names),
            "bank_filter": bank_name,
            "message": f"Retrieved {len(file_names)} files successfully"
        }), 200
    except Exception as e:
        logger.error(f"Error in get-files endpoint: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.DATABASE,
            "Failed to retrieve file list from database",
            status_code=500,
            details={"error": str(e)}
        )

@app.route('/api/get-template-files', methods=['GET'])
def get_template_files():
    """Get all file IDs from database and Documents folder for frontend template dropdown"""
    logger.info("Processing get-template-files request")
    try:
        file_ids = get_all_file_ids()
        if not file_ids:
            logger.warning("No file IDs found in database or Documents folder")
            return jsonify({
                "success": True,
                "templates": [],
                "message": "No file IDs found in the system",
                "count": 0
            }), 200
        logger.info(f"Retrieved {len(file_ids)} file IDs")
        return jsonify({
            "success": True,
            "templates": file_ids,
            "count": len(file_ids),
            "message": f"Retrieved {len(file_ids)} file IDs successfully"
        }), 200
    except Exception as e:
        logger.error(f"Error in get-template-files endpoint: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.DATABASE,
            "Failed to retrieve file ID list from database or Documents folder",
            status_code=500,
            details={"error": str(e)}
        )

@app.route('/api/retrieve-summaries', methods=['POST'])
def retrieve_summaries():
    """Retrieve summaries for a given file"""
    logger.info("Processing retrieve-summaries request")
    try:
        data = request.get_json()
        if not data or 'original_filename' not in data:
            logger.warning("Invalid request data: original_filename missing")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "original_filename is required in the request body",
                status_code=400
            )
        original_filename = data.get('original_filename', '').strip()
        if not original_filename:
            logger.warning("Empty original_filename provided")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "original_filename cannot be empty",
                status_code=400
            )
        logger.info(f"Retrieving summaries for file: {original_filename}")
        file_ext = os.path.splitext(original_filename)[1].lower()
        if file_ext not in ALLOWED_SUMMARY_EXTENSIONS:
            logger.warning(f"Invalid file type for summary: {original_filename}")
            raise AppError(
                ErrorCategory.FILE_FORMAT,
                f"Invalid file type for '{original_filename}'. Only PDF, XLSX, and TXT files are supported for summaries.",
                status_code=400,
                details={"filename": original_filename, "extension": file_ext, "allowed": list(ALLOWED_SUMMARY_EXTENSIONS)}
            )
        summaries = retrieve_summary_blobs(original_filename)
        if not summaries:
            logger.warning(f"No summaries found for {original_filename}")
            raise AppError(
                ErrorCategory.FILE_NOT_FOUND,
                f"No summaries found for '{original_filename}'",
                status_code=404,
                details={"filename": original_filename}
            )
        logger.info(f"Retrieved {len(summaries)} summaries for {original_filename}")
        return jsonify({
            "success": True,
            "FINAL_RESULT": "Summaries retrieved",
            "summary_files": summaries
        }), 200
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error in retrieve-summaries endpoint: {str(e)}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred while retrieving summaries",
            user_friendly=True,
            status_code=500,
            details={"error": str(e)}
        )


@app.route('/api/delete-files', methods=['POST'])
@require_auth
def delete_files():
    """Delete a file from all vector store tables and file storage"""
    logger.info("Processing delete-files request")
    try:
        data = request.get_json()
        if not data or 'filename' not in data:
            logger.warning("Invalid request data: filename missing")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "filename is required in the request body",
                status_code=400
            )
        
        filename = data.get('filename', '').strip()
        if not filename:
            logger.warning("Empty filename provided")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "filename cannot be empty",
                status_code=400
            )
        
        logger.info(f"Attempting to delete file: {filename}")
        
        # Delete from vector store tables
        deletion_summary = delete_file_from_all_tables(filename)
        
        # Delete physical file from Documents folder
        physical_file_deleted = False
        physical_file_error = None
        
        try:
            file_path = os.path.join(documents_dir, filename)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
                physical_file_deleted = True
                logger.info(f"Physical file deleted: {file_path}")
            else:
                logger.info(f"Physical file not found: {file_path}")
        except Exception as e:
            physical_file_error = str(e)
        
        # Update deletion summary with physical file deletion status
        deletion_summary['physical_file_deleted'] = physical_file_deleted
        if physical_file_error:
            deletion_summary['physical_file_error'] = physical_file_error
        
        if not deletion_summary['success'] and deletion_summary.get('error'):
            logger.error(f"Failed to delete file {filename}: {deletion_summary['error']}")
            raise AppError(
                ErrorCategory.DATABASE,
                f"Failed to delete file '{filename}': {deletion_summary['error']}",
                status_code=500,
                details={"filename": filename, "errors": deletion_summary['errors']}
            )
        
        logger.info(f"Deletion completed for {filename}: {deletion_summary['total_deleted']} records deleted")
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted successfully",
            "deletion_summary": {
                "total_deleted": deletion_summary['total_deleted'],
                "deleted_counts": deletion_summary['deleted_counts'],
                "physical_file_deleted": deletion_summary.get('physical_file_deleted', False),
                "errors": deletion_summary['errors']
            }
        }), 200
    
    except AppError:
        raise
    except Exception as e:
        logger.error(f"Error in delete-files endpoint: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred while deleting the file",
            user_friendly=True,
            status_code=500,
            details={"error": str(e)}
        )


@app.route('/api/metadata-retreival', methods=['POST'])
@require_auth
def get_metadata():
    """API endpoint to retrieve metadata for a given filename."""
    try:
        filename = request.args.get('filename')
        
        if not filename:
            return jsonify({'error': 'Missing required parameter: filename'}), 400
        
        # Use the same function as in upload API
        metadata = get_metadata_by_filename(filename)
        
        if metadata:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'metadata': metadata
            }), 200
        else:
            return jsonify({
                'status': 'success',
                'filename': filename,
                'metadata': None,
                'message': f'No metadata found for {filename}'
            }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving metadata for {filename}: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
            
@app.route('/api/upload', methods=['POST'])
@require_auth
def upload_file():
    client_ip = request.remote_addr
    logger.info(f"Received upload request from {client_ip}")
    current_time = int(time.time())
    hour_ago = current_time - 3600

    try:
        # Clean up old upload counts
        upload_counts[client_ip] = [t for t in upload_counts.get(client_ip, []) if t > hour_ago]

        # Check upload limit
        if len(upload_counts.get(client_ip, [])) >= UPLOAD_LIMIT:
            logger.warning(f"Upload limit reached for {client_ip}")
            raise AppError(
                ErrorCategory.RATE_LIMIT,
                "Upload limit reached (500 files per hour). Please try again later.",
                status_code=429
            )

        if 'file' not in request.files:
            logger.error("No file part in request")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "No file provided in the request",
                status_code=400
            )

        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                "No file selected",
                status_code=400
            )

        if not allowed_file(file.filename):
            logger.error(f"Unsupported file type: {file.filename}")
            file_ext = os.path.splitext(file.filename)[1].lower() or "unknown"
            raise AppError(
                ErrorCategory.FILE_FORMAT,
                "Unsupported file type. Only PDF, XLSX, .DOC,  .DOCX, TXT, JPG, PNG, and JPEG are allowed.",
                status_code=400,
                details={"filename": file.filename, "extension": file_ext, "allowed": list(ALLOWED_EXTENSIONS)}
            )

        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        # Determine the appropriate directory based on file extension
        target_dir = images_dir if file_ext in IMAGE_EXTENSIONS else documents_dir
        file_path = os.path.join(target_dir, filename)

        if os.path.exists(file_path):
            overwrite = request.form.get('overwrite', 'false').lower() == 'true'
            logger.info(f"File {filename} exists in {target_dir}, overwrite={overwrite}")
            if not overwrite:
                return jsonify({
                    "exists": True, 
                    "message": f"File '{filename}' already exists. Overwrite?",
                    "filename": filename
                }), 200
            # If overwrite is true, continue with saving

        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size > MAX_FILE_SIZE:
            logger.error(f"File {filename} too large: {file_size} bytes")
            raise AppError(
                ErrorCategory.INPUT_VALIDATION,
                f"File too large. Maximum size is {MAX_FILE_SIZE//1024//1024}MB.",
                status_code=400,
                details={"filename": filename, "size": file_size, "max_size": MAX_FILE_SIZE}
            )

        file.save(file_path)
        logger.info(f"File {filename} saved to {file_path}")

        # Record upload
        if client_ip not in upload_counts:
            upload_counts[client_ip] = []
        upload_counts[client_ip].append(current_time)

        target_folder_name = 'Images' if file_ext in IMAGE_EXTENSIONS else 'Documents'
        
        # Initialize base message
        base_message = f"File '{filename}' uploaded and scanning through the Document."
        
        # Process document only if it's not an image
        if file_ext not in IMAGE_EXTENSIONS:
            try:
                logger.info(f"Starting document processing for: {filename}")
                
                # Initialize processor
                processor = SimpleDocumentProcessor()
                
                # Process document
                result = processor.process_document(filename, store_in_database=True)
                
                # Cleanup processing files
                cleanup_processing_files(filename)
                
                if result["status"] == "SUCCESS":
                    logger.info(f"✅ Successfully processed: {filename}")
                    logger.info(f"🎯 Decision: {result['final_decision']}")
                    logger.info(f"📁 Output files: {len(result['output_files'])} files")
                    logger.info(f"💾 Database: {'✅ STORED' if result['database_stored'] else '❌ NOT STORED'}")
                    
                    # Enhanced success message with decision
                    decision = result.get('final_decision', 'Unknown')
                    final_message = f"{base_message} and processed. Decision: {decision}.Reasoning: {result['reasoning']}"
                else:
                    logger.error(f"❌ Processing failed: {filename}")
                    for error in result["errors"]:
                        logger.error(f"   • {error}")
                    
                    # File uploaded but processing failed
                    final_message = f"{base_message}, but processing failed"
                
            except ImportError as e:
                logger.error(f"Failed to import document processor: {str(e)}")
                final_message = f"{base_message}, but document processor is not available"
                
            except Exception as e:
                logger.error(f"Error during document processing for {filename}: {str(e)}")
                final_message = f"{base_message}, but processing encountered an error"
        else:
            # Image file - no processing needed
            final_message = base_message
        
        # Get metadata after successful upload and processing
        metadata = None
        try:
            logger.info(f"Retrieving metadata for file: {filename}")
            metadata = get_metadata_by_filename(filename)
            if metadata:
                logger.info(f"Successfully retrieved metadata for {filename}")
            else:
                logger.warning(f"No metadata found for {filename}")
        except Exception as e:
            logger.error(f"Error retrieving metadata for {filename}: {str(e)}")
            # Don't fail the upload if metadata retrieval fails
        
        # Prepare response
        response_data = {
            "success": True,
            "message": final_message,
            "filename": filename,
            "folder": target_folder_name
        }
        
        # Add metadata to response if available
        if metadata:
            response_data["metadata"] = metadata
        
        return jsonify(response_data), 200
    
    except AppError:
        # Re-raise AppError to be caught by the global handler
        raise
    except Exception as e:
        logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "An unexpected error occurred during file upload",
            status_code=500
        )

@app.route('/api/delete-document', methods=['POST'])
@require_auth
def delete_document():
    try:
        # Get filename from request body or query parameter
        data = request.get_json() if request.is_json else {}
        filename = data.get('filename') or request.args.get('filename')
        
        # If no filename provided, delete all files in documents_dir
        if not filename:
            deleted_files = []
            failed_files = []
            
            # Get all files in the documents directory
            if os.path.exists(documents_dir):
                for file in os.listdir(documents_dir):
                    file_path = os.path.join(documents_dir, file)
                    # Only delete files, not directories
                    if os.path.isfile(file_path):
                        try:
                            os.remove(file_path)
                            deleted_files.append(file)
                        except Exception as e:
                            failed_files.append(file)
            
            # Return response based on results
            if deleted_files:
                message = f"Successfully deleted {len(deleted_files)} file(s)"
                if failed_files:
                    message += f", failed to delete {len(failed_files)} file(s)"
                return jsonify({
                    "success": True,
                    "message": message,
                    "deleted_files": deleted_files,
                    "failed_files": failed_files
                }), 200
            else:
                return jsonify({
                    "success": True,
                    "message": "No files found to delete"
                }), 200
        
        # Original logic for deleting a specific file
        # Construct file path in documents_dir only
        file_path = os.path.join(documents_dir, filename)
        
        # Security check: ensure the path is within documents_dir
        if not file_path.startswith(os.path.abspath(documents_dir)):
            return jsonify({"error": "Invalid file path"}), 400
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": f"File '{filename}' not found"}), 404
            
        # Delete the file
        os.remove(file_path)
        
        return jsonify({
            "success": True,
            "message": f"File '{filename}' deleted successfully"
        }), 200
        
    except Exception as e:
        return jsonify({"error": "An error occurred during file deletion"}), 500

@app.route('/api/approved-file-upload', methods=['POST'])
@require_auth
def approved_upload_file():
    client_ip = request.remote_addr
    logger.info(f"Received upload request from {client_ip}")

    try:
        # Basic file existence check
        if 'file' not in request.files:
            return jsonify({"error": "No file provided in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = file.filename
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Determine the appropriate directory based on file extension
        target_dir = images_dir if file_ext in IMAGE_EXTENSIONS else documents_dir
        file_path = os.path.join(target_dir, filename)

        # Handle file overwrite
        if os.path.exists(file_path):
            overwrite = request.form.get('overwrite', 'false').lower() == 'true'
            logger.info(f"File {filename} exists in {target_dir}, overwrite={overwrite}")
            if not overwrite:
                return jsonify({
                    "exists": True, 
                    "message": f"File '{filename}' already exists. Overwrite?",
                    "filename": filename
                }), 200

        # Save the file
        file.save(file_path)
        logger.info(f"File {filename} saved to {file_path}")

        target_folder_name = 'Images' if file_ext in IMAGE_EXTENSIONS else 'Documents'
        
        # Prepare response
        response_data = {
            "success": True,
            "message": f"File '{filename}' uploaded successfully.",
            "filename": filename,
            "folder": target_folder_name
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred during file upload"}), 500


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - redirect to API status or serve a welcome page"""
    logger.info("Accessing root endpoint")
    try:
        return jsonify({
            "message": "Flask Server is running",
            "status": "active",
            "api_base": "/api/",
            "available_endpoints": [
                "/api/ - Server status",
                "/api/health - Health check", 
                "/api/chat - Chat endpoint",
                "/api/upload - File upload",
                "/api/get-files - Get file list",
                "/api/generate-uploads - Process uploads"
            ]
        }), 200
    except Exception as e:
        logger.error(f"Root endpoint error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    """Clean up resources when the application context is torn down"""
    logger.info("Starting application context cleanup")
    try:
        cleanup_managers()
        logger.info("Vector store managers cleaned up")
        if HANA_CONN:
            HANA_CONN.close()
            logger.info("HANA connection closed")
        logger.debug("Cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during application cleanup: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Starting Flask application on port {port}")
    try:
        for rule in app.url_map.iter_rules():
            methods = rule.methods if rule.methods else set()
            logger.info(f"Registered route: {rule} (Methods: {', '.join(sorted(methods))})")
        app.run(host='0.0.0.0', port=port, debug=not IS_PRODUCTION)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}\n{traceback.format_exc()}")
        raise AppError(
            ErrorCategory.INTERNAL,
            "Failed to start the application",
            status_code=500
        )