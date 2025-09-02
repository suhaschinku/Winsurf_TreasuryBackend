from hdbcli import dbapi
from langchain_community.vectorstores import HanaDB
from gen_ai_hub.proxy.langchain.init_models import init_embedding_model
from logger_setup import get_logger
import os
from dotenv import load_dotenv
from env_config import TABLE_NAMES, EMBEDDING_MODEL,TREASURY_SUMMARY_TABLE,SCHEMA_NAME,DUBLIN_CORE_METADATA
from contextlib import contextmanager
import threading
import atexit
import time
from datetime import datetime, timedelta
import schedule
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_hana_credentials, extract_aicore_credentials
import json
import base64
import typing

load_dotenv()
logger = get_logger()

# Configuration
FILE_STORAGE_TABLE = TREASURY_SUMMARY_TABLE
SCHEMA_NAME = SCHEMA_NAME
base_path = os.getenv('LOCALPATH', '')
DOCUMENTS_FOLDER_PATH = os.path.join(base_path, "Documents")

# --- HANA CREDENTIALS FROM DESTINATION SERVICES ---
vcap_services = os.environ.get("VCAP_SERVICES")
logger.info("===>DB_Connections => GET HANA CREDENTIALS FROM DESTINATION SERVICES<===")

# Extract destination service credentials from VCAP_SERVICES
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")

# Generate OAuth token for destination service
try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully for destination service.")
except Exception as e:
    logger.error(f"Error generating OAuth token: {str(e)}")
    oauth_token = None

# Get the destination details for the HANA DB
HANA_CREDENTIALS = None
AIC_CREDENTIALS = None
if oauth_token:
    dest_HDB = 'GENAI_HDB'  # Destination name for HANA DB
    hana_dest_details = fetch_destination_details(
        uri=destination_service_credentials['dest_base_url'],
        name=dest_HDB,
        token=oauth_token
    )
    logger.info(f"HANA Destination Details: {hana_dest_details}")
    HANA_CREDENTIALS = extract_hana_credentials(hana_dest_details)
    logger.info(f"HANA_CREDENTIALS: {HANA_CREDENTIALS}")
    AIC_CREDENTIALS = extract_aicore_credentials(hana_dest_details)
    logger.info(f"AIC_CREDENTIALS: {AIC_CREDENTIALS}")
else:
    logger.warning("OAuth token not available; HANA credentials not initialized.")

# Custom Connection Pool Implementation
class ConnectionPool:
    def __init__(self, max_connections=20):
        self.max_connections = max_connections
        self.pool = []
        self.lock = threading.Lock()
        self.cleanup_scheduler_started = False

    def get_connection(self):
        """Fetch a connection from the pool or create a new one if necessary."""
        with self.lock:
            if self.pool:
                logger.debug("Reusing connection from pool")
                return self.pool.pop()
            else:
                logger.debug("Creating a new connection")
                return self._create_connection()

    def release_connection(self, conn):
        """Release a connection back to the pool."""
        with self.lock:
            if len(self.pool) < self.max_connections:
                self.pool.append(conn)
                logger.debug("Connection released back to pool")
            else:
                conn.close()
                logger.debug("Connection closed as pool is full")

    def _create_connection(self):
        """Create a new database connection."""
        try:
            if not HANA_CREDENTIALS or not all([HANA_CREDENTIALS.get(k) for k in ['address', 'user', 'password', 'port']]):
                logger.error("HANA credentials not properly initialized")
                raise Exception("HANA credentials not available")
            conn = dbapi.connect(
                address=HANA_CREDENTIALS['address'],
                port=int(HANA_CREDENTIALS['port']),
                user=HANA_CREDENTIALS['user'],
                password=HANA_CREDENTIALS['password'],
                encrypt=True,
                sslValidateCertificate=False
            )
            # Set schema if provided
            if HANA_CREDENTIALS.get('schema'):
                cursor = conn.cursor()
                cursor.execute(f"SET SCHEMA {HANA_CREDENTIALS['schema']}")
                cursor.close()
            logger.info("Database connection established successfully")
            return conn
        except Exception as e:
            logger.error(f"Failed to establish database connection: {e}")
            raise

    def close_all_connections(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                try:
                    conn.close()
                    logger.info("Closed connection from pool")
                except Exception as e:
                    logger.error(f"Error closing connection: {e}")
            self.pool.clear()

# Initialize the global connection pool
connection_pool = ConnectionPool(max_connections=20)

def get_db_connection():
    """Fetch a connection from the global connection pool."""
    logger.debug("Fetching connection from pool")
    return connection_pool.get_connection()

def release_db_connection(conn):
    """Release a connection back to the global connection pool."""
    logger.debug("Releasing connection back to pool")
    connection_pool.release_connection(conn)

def close_all_db_connections():
    """Close all connections in the global connection pool."""
    logger.info("Closing all database connections in the pool")
    connection_pool.close_all_connections()

def cleanup_old_filtered_tables(connection=None, max_age_hours=24):
    """Clean up old filtered tables that might be orphaned"""
    temp_connection = None
    try:
        if connection is None:
            temp_connection = get_db_connection()
            connection = temp_connection
        
        cursor = connection.cursor()
        
        try:
            cursor.execute("""
                SELECT TABLE_NAME FROM SYS.TABLES 
                WHERE TABLE_NAME LIKE '%_FILTERED_%' 
                AND SCHEMA_NAME = ?
            """, (SCHEMA_NAME,))
            
            tables_to_check = cursor.fetchall()
            cleaned_count = 0
            
            for (table_name,) in tables_to_check:
                try:
                    cursor.execute(f"DROP TABLE {SCHEMA_NAME}.{table_name}")
                    connection.commit()
                    cleaned_count += 1
                    logger.info(f"Cleaned up filtered table: {table_name}")
                except Exception as e:
                    logger.debug(f"Could not cleanup table {table_name}: {e}")
                    continue
            
            if cleaned_count > 0:
                logger.info(f"Cleanup completed: {cleaned_count} filtered tables removed")
            else:
                logger.debug("No orphaned filtered tables found to cleanup")
                
        except Exception as e:
            logger.error(f"Error during table cleanup query: {e}")
        finally:
            cursor.close()
            
    except Exception as e:
        logger.error(f"Error in cleanup_old_filtered_tables: {e}")
    finally:
        if temp_connection:
            release_db_connection(temp_connection)

def start_cleanup_scheduler():
    """Start the background cleanup scheduler"""
    def run_cleanup():
        try:
            logger.info("Running scheduled cleanup of old filtered tables")
            cleanup_old_filtered_tables()
        except Exception as e:
            logger.error(f"Scheduled cleanup failed: {e}")
    
    def scheduler_worker():
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    schedule.every(6).hours.do(run_cleanup)
    
    scheduler_thread = threading.Thread(target=scheduler_worker, daemon=True)
    scheduler_thread.start()
    logger.info("Cleanup scheduler started - will run every 6 hours")

def build_bank_filter(bank_name):
    """Build SQL filter condition for bank name filtering"""
    if not bank_name or (isinstance(bank_name, str) and not bank_name.strip()):
        logger.debug("No bank name provided - returning empty filter")
        return ""
    
    try:
        bank_name = bank_name.strip() if isinstance(bank_name, str) else str(bank_name).strip()
        
        if ',' in bank_name:
            bank_names = [b.strip() for b in bank_name.split(',') if b.strip()]
            if not bank_names:
                logger.debug("No valid bank names after splitting - returning empty filter")
                return ""
            
            conditions = []
            for name in bank_names:
                sanitized_name = name.replace("'", "''")
                conditions.append(f"JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) LIKE '%{sanitized_name}%'")
            
            return "WHERE " + "(" + " OR ".join(conditions) + ")"
        else:
            sanitized_name = bank_name.replace("'", "''")
            return f"WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) LIKE '%{sanitized_name}%'"
            
    except Exception as e:
        logger.error(f"Error building bank filter: {e}")
        return ""

def should_create_filtered_table(bank_name):
    """Determine if a filtered table should be created based on bank_name"""
    if not bank_name or (isinstance(bank_name, str) and not bank_name.strip()):
        return False
    return True

class FilteredVectorStore:
    """Filtered vector store that creates a new HanaDB instance with filtered data"""
    
    def __init__(self, original_table_name, embedding_model, connection, bank_name=None, store_type=None):
        self.original_table_name = original_table_name
        self.embedding_model = embedding_model  
        self.connection = connection
        self.bank_name = bank_name
        self.store_type = store_type
        self.filtered_table_name = None
        self.vector_store = None
        self.created_timestamp = datetime.now()
        self.is_filtered = should_create_filtered_table(bank_name)
        
        self._create_filtered_store()
    
    def _create_filtered_store(self):
        """Create a new table with filtered data and initialize HanaDB on it"""
        try:
            if self.is_filtered:
                self.filtered_table_name = f"{self.original_table_name}_FILTERED_{threading.get_ident()}_{int(time.time())}"
                
                cursor = self.connection.cursor()
                
                try:
                    cursor.execute(f"DROP TABLE {SCHEMA_NAME}.{self.filtered_table_name}")
                    logger.debug(f"Dropped existing filtered table: {self.filtered_table_name}")
                except Exception:
                    pass
                
                filter_condition = build_bank_filter(self.bank_name)
                
                create_query = f"""
                    CREATE TABLE {SCHEMA_NAME}.{self.filtered_table_name} AS (
                        SELECT * FROM {SCHEMA_NAME}.{self.original_table_name}
                        {filter_condition}
                    )
                """
                
                cursor.execute(create_query)
                self.connection.commit()
                
                cursor.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{self.filtered_table_name}")
                count = cursor.fetchone()[0]
                logger.info(f"Created filtered table {self.filtered_table_name} with {count} records for {self.store_type}")
                
                cursor.close()
                
                self.vector_store = HanaDB(
                    connection=self.connection,
                    embedding=self.embedding_model,
                    table_name=f"{self.filtered_table_name}",
                    content_column="VEC_TEXT",
                    metadata_column="VEC_META", 
                    vector_column="VEC_VECTOR"
                )
                
                logger.info(f"Successfully created filtered vector store for {self.store_type}")
            else:
                logger.info(f"No bank filter provided - using original table {self.original_table_name} for {self.store_type}")
                
                self.vector_store = HanaDB(
                    connection=self.connection,
                    embedding=self.embedding_model,
                    table_name=f"{self.original_table_name}",
                    content_column="VEC_TEXT",
                    metadata_column="VEC_META", 
                    vector_column="VEC_VECTOR"
                )
                
                cursor = self.connection.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {SCHEMA_NAME}.{self.original_table_name}")
                count = cursor.fetchone()[0]
                cursor.close()
                
                logger.info(f"Using original table {self.original_table_name} with {count} records for {self.store_type}")
                
        except Exception as e:
            logger.error(f"Error creating filtered store for {self.store_type}: {e}")
            raise
    
    def similarity_search(self, query, k=4):
        """Perform similarity search on the vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query, k=4):
        """Perform similarity search with scores on the vector store"""
        if not self.vector_store:
            logger.error("Vector store not initialized")
            return []
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_document_count(self):
        """Get the number of documents in the store"""
        try:
            cursor = self.connection.cursor()
            table_to_count = f"{SCHEMA_NAME}.{self.filtered_table_name}" if self.is_filtered else f"{SCHEMA_NAME}.{self.original_table_name}"
            cursor.execute(f"SELECT COUNT(*) FROM {table_to_count}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            logger.error(f"Error counting documents: {e}")
            return 0
    
    def cleanup(self):
        """Clean up the filtered table (only if one was created)"""
        try:
            if self.is_filtered and self.filtered_table_name:
                cursor = self.connection.cursor() 
                cursor.execute(f"DROP TABLE {SCHEMA_NAME}.{self.filtered_table_name}")
                self.connection.commit()
                cursor.close()
                logger.debug(f"Cleaned up filtered table: {self.filtered_table_name}")
        except Exception as e:
            logger.error(f"Error cleaning up filtered table {self.filtered_table_name}: {e}")

class VectorStoreManager:
    """Manager for multiple filtered vector stores with enhanced cleanup"""
    
    def __init__(self, bank_name=None, AIC_CREDENTIALS=None):
        self.bank_name = bank_name
        self.AIC_CREDENTIALS = AIC_CREDENTIALS
        self.connection = None
        self.embedding_model = None
        self.filtered_stores = {}
        self.store_mapping = {
            'transcript': 'transcript',
            'non_transcript': 'non_transcript', 
            'excel_non_transcript': 'excel_non_transcript'
        }
        self.created_timestamp = datetime.now()
        self.manager_id = f"{threading.get_ident()}_{int(time.time())}"
        
        if not hasattr(VectorStoreManager, '_active_managers'):
            VectorStoreManager._active_managers = {}
        VectorStoreManager._active_managers[self.manager_id] = self
        
    def initialize(self):
        """Initialize the vector store manager"""
        try:
            if not connection_pool.cleanup_scheduler_started:
                start_cleanup_scheduler()
                connection_pool.cleanup_scheduler_started = True
            
            # Initialize embedding model with AIC credentials
            try:
                from gen_ai_hub.proxy import GenAIHubProxyClient
                proxy_client = GenAIHubProxyClient(
                    base_url=self.AIC_CREDENTIALS['aic_base_url'],
                    auth_url=self.AIC_CREDENTIALS['aic_auth_url'],
                    client_id=self.AIC_CREDENTIALS['clientid'],
                    client_secret=self.AIC_CREDENTIALS['clientsecret'],
                    resource_group=self.AIC_CREDENTIALS['resource_group']
                )
                self.embedding_model = init_embedding_model(model_name=EMBEDDING_MODEL, proxy_client=proxy_client)
                logger.info("Embedding model initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {str(e)}")
                raise
            
            self.connection = get_db_connection()
            if not self.connection:
                raise Exception("Failed to get database connection")
            
            if not TABLE_NAMES or not isinstance(TABLE_NAMES, dict):
                raise Exception("TABLE_NAMES not properly configured")
            
            filter_status = "with bank filter" if self.bank_name else "without bank filter"
            logger.info(f"Initializing vector stores {filter_status}: {self.bank_name if self.bank_name else 'None'}")
            
            for store_type, table_name in TABLE_NAMES.items():
                try:
                    logger.info(f"Creating store for {store_type} (table: {table_name})")
                    filtered_store = FilteredVectorStore(
                        original_table_name=table_name,
                        embedding_model=self.embedding_model,
                        connection=self.connection,
                        bank_name=self.bank_name,
                        store_type=store_type
                    )
                    self.filtered_stores[store_type] = filtered_store
                    logger.info(f"Successfully created store for {store_type}")
                    
                except Exception as e:
                    logger.error(f"Failed to create store for {store_type}: {e}")
                    continue
            
            if not self.filtered_stores:
                raise Exception("Failed to create any vector stores")
            
            counts = self.get_document_counts()
            for store_type, count in counts.items():
                logger.info(f"{store_type.capitalize()} vector store loaded with {count} documents")
                if count == 0:
                    logger.warning(f"No documents found in the {store_type} vector store!")
            
            logger.info(f"Successfully initialized {len(self.filtered_stores)} vector stores")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing VectorStoreManager: {e}")
            self.cleanup()
            raise
    
    def get_store(self, store_type):
        """Get a specific vector store"""
        if store_type not in self.filtered_stores:
            logger.error(f"Store type '{store_type}' not available. Available: {list(self.filtered_stores.keys())}")
            return None
        return self.filtered_stores[store_type].vector_store
    
    def get_all_stores(self):
        """Get all vector stores as tuple (transcript, non_transcript, excel_non_transcript)"""
        transcript_store = self.get_store('transcript')
        non_transcript_store = self.get_store('non_transcript') 
        excel_non_transcript_store = self.get_store('excel_non_transcript')
        return transcript_store, non_transcript_store, excel_non_transcript_store
    
    def get_stores_dict(self):
        """Get all vector stores as dictionary"""
        return {store_type: store.vector_store for store_type, store in self.filtered_stores.items()}
    
    def search_store(self, store_type, query, k=4):
        """Search a specific store"""
        if store_type not in self.filtered_stores:
            logger.error(f"Store type '{store_type}' not available")
            return []
        return self.filtered_stores[store_type].similarity_search(query, k=k)
    
    def search_all_stores(self, query, k=4):
        """Search all stores and combine results"""
        all_results = []
        for store_type, store in self.filtered_stores.items():
            try:
                results = store.similarity_search(query, k=k)
                for result in results:
                    if hasattr(result, 'metadata'):
                        result.metadata['store_type'] = store_type
                all_results.extend(results)
                logger.debug(f"Found {len(results)} results in {store_type}")
            except Exception as e:
                logger.error(f"Error searching {store_type}: {e}")
                continue
        
        return all_results[:k]
    
    def get_document_counts(self):
        """Get document counts for all stores"""
        counts = {}
        for store_type, store in self.filtered_stores.items():
            counts[store_type] = store.get_document_count()
        return counts
    
    def cleanup(self):
        """Clean up all resources"""
        try:
            for store in self.filtered_stores.values():
                try:
                    store.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up store: {e}")
            
            if self.connection:
                release_db_connection(self.connection)
                self.connection = None
            
            self.filtered_stores.clear()
            
            if hasattr(VectorStoreManager, '_active_managers') and self.manager_id in VectorStoreManager._active_managers:
                del VectorStoreManager._active_managers[self.manager_id]
            
            logger.debug(f"VectorStoreManager {self.manager_id} cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during VectorStoreManager cleanup: {e}")

    @classmethod
    def cleanup_all_managers(cls):
        """Clean up all active managers (emergency cleanup)"""
        if hasattr(cls, '_active_managers'):
            managers_to_cleanup = list(cls._active_managers.values())
            for manager in managers_to_cleanup:
                try:
                    manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up manager {manager.manager_id}: {e}")
            cls._active_managers.clear()
            logger.info(f"Emergency cleanup completed for {len(managers_to_cleanup)} managers")

def load_vector_stores(bank_name: str = None, AIC_CREDENTIALS=None):
    """Initialize vector stores, maintaining original function signature"""
    try:
        manager = VectorStoreManager(bank_name, AIC_CREDENTIALS)
        manager.initialize()
        
        transcript_store, non_transcript_store, excel_non_transcript_store = manager.get_all_stores()
        
        if not hasattr(load_vector_stores, '_managers'):
            load_vector_stores._managers = []
        load_vector_stores._managers.append(manager)
        
        return transcript_store, non_transcript_store, excel_non_transcript_store
        
    except Exception as e:
        logger.error(f"Failed to load vector stores: {e}")
        return None, None, None

@contextmanager
def get_filtered_vector_stores(bank_name: str = None, AIC_CREDENTIALS=None):
    """Context manager for vector stores - recommended approach"""
    manager = VectorStoreManager(bank_name, AIC_CREDENTIALS)
    try:
        manager.initialize()
        yield manager
    finally:
        manager.cleanup()

def cleanup_managers():
    """Clean up any managers created by load_vector_stores function"""
    if hasattr(load_vector_stores, '_managers'):
        for manager in load_vector_stores._managers:
            try:
                manager.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up manager: {e}")
        load_vector_stores._managers.clear()

def force_cleanup_all():
    """Force cleanup of all resources - use in emergency situations"""
    try:
        cleanup_managers()
        VectorStoreManager.cleanup_all_managers()
        cleanup_old_filtered_tables()
        close_all_db_connections()
        logger.info("Force cleanup completed successfully")
    except Exception as e:
        logger.error(f"Error during force cleanup: {e}")

def get_system_stats():
    """Get statistics about current system state"""
    stats = {
        'active_managers': 0,
        'pool_connections': 0,
        'cleanup_scheduler_running': connection_pool.cleanup_scheduler_started
    }
    
    try:
        if hasattr(VectorStoreManager, '_active_managers'):
            stats['active_managers'] = len(VectorStoreManager._active_managers)
        
        with connection_pool.lock:
            stats['pool_connections'] = len(connection_pool.pool)
            
    except Exception as e:
        logger.error(f"Error getting system stats: {e}")
    
    return stats

def test_filtered_vector_stores(bank_name=None, AIC_CREDENTIALS=None):
    """Test the vector stores functionality with enhanced features"""
    test_name = bank_name if bank_name else "NO_BANK_FILTER"
    logger.info(f"Testing vector stores with bank_name filter: '{test_name}'")
    
    try:
        logger.info("Getting system statistics...")
        stats = get_system_stats()
        logger.info(f"System stats: {stats}")
        
        logger.info("Testing cleanup functionality...")
        cleanup_old_filtered_tables()
        
        logger.info("Testing original load_vector_stores function...")
        transcript_store, non_transcript_store, excel_non_transcript_store = load_vector_stores(bank_name, AIC_CREDENTIALS)
        
        if transcript_store and non_transcript_store and excel_non_transcript_store:
            logger.info("✅ Original function interface: PASSED")
        else:
            logger.error("❌ Original function interface: FAILED")
            return False
        
        logger.info("Testing enhanced context manager interface...")
        with get_filtered_vector_stores(bank_name, AIC_CREDENTIALS) as manager:
            stores = manager.get_stores_dict()
            logger.info(f"Available stores: {list(stores.keys())}")
            
            counts = manager.get_document_counts()
            logger.info(f"Document counts: {counts}")
            
            for store_type in stores.keys():
                store = manager.get_store(store_type)
                if store:
                    logger.info(f"✅ Successfully accessed {store_type} store")
                else:
                    logger.error(f"❌ Failed to access {store_type} store")
            
            test_query = "test query"
            for store_type in stores.keys():
                try:
                    results = manager.search_store(store_type, test_query, k=2)
                    logger.info(f"✅ Search test for {store_type}: {len(results)} results")
                except Exception as e:
                    logger.error(f"❌ Search test failed for {store_type}: {e}")
            
            logger.info("✅ Enhanced context manager interface: PASSED")
        
        stats_after = get_system_stats()
        logger.info(f"System stats after operations: {stats_after}")
        
        cleanup_managers()
        
        logger.info("✅ All tests completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        force_cleanup_all()
        return False

def get_all_file_names(bank_name: str = None):
    """
    Get all unique file names from vector stores.
    """
    connection = None
    try:
        logger.info(f"Fetching file names with bank filter: {bank_name if bank_name else 'None (all banks)'}")
        
        connection = get_db_connection()
        if not connection:
            logger.error("Failed to get database connection")
            return []
        
        cursor = connection.cursor()
        all_filenames = set()
        
        for store_type, table_name in TABLE_NAMES.items():
            try:
                logger.debug(f"Fetching file names from {store_type} table: {table_name}")
                
                if should_create_filtered_table(bank_name):
                    filter_condition = build_bank_filter(bank_name)
                    query = f"""
                        SELECT DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file
                        FROM {SCHEMA_NAME}.{table_name}
                        {filter_condition}
                        AND JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) IS NOT NULL
                        AND JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) != ''
                    """
                else:
                    query = f"""
                        SELECT DISTINCT JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) as source_file
                        FROM {SCHEMA_NAME}.{table_name}
                        WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) IS NOT NULL
                        AND JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) != ''
                    """
                
                cursor.execute(query)
                results = cursor.fetchall()
                
                for (filename,) in results:
                    if filename and filename.strip():
                        clean_filename = filename.strip()
                        all_filenames.add(clean_filename)
                
                logger.debug(f"Found {len(results)} unique files in {store_type}")
                
            except Exception as e:
                logger.error(f"Error fetching file names from {store_type} table {table_name}: {e}")
                continue
        
        cursor.close()
        
        filename_list = sorted(list(all_filenames))
        
        logger.info(f"Successfully retrieved {len(filename_list)} unique file names")
        
        if len(filename_list) == 0:
            if bank_name:
                logger.warning(f"No files found for bank filter: {bank_name}")
            else:
                logger.warning("No files found in any vector store")
        else:
            logger.debug(f"File names preview: {filename_list[:5]}{'...' if len(filename_list) > 5 else ''}")
        
        return filename_list
        
    except Exception as e:
        logger.error(f"Error in get_all_file_names: {e}")
        return []
    finally:
        if connection:
            release_db_connection(connection)

def get_all_file_ids():
    """
    Get all unique file IDs from the database table and local Documents folder.
    """
    connection = None
    try:
        logger.info("Fetching all file IDs from database and Documents folder")
        
        connection = get_db_connection()
        if not connection:
            logger.error("Failed to get database connection")
            return []
        
        cursor = connection.cursor()
        all_file_ids = set()
        
        try:
            logger.debug(f"Fetching file IDs from table: {SCHEMA_NAME}.{FILE_STORAGE_TABLE}")
            
            query = f"""
                SELECT DISTINCT FILE_ID
                FROM {SCHEMA_NAME}.{FILE_STORAGE_TABLE}
                WHERE FILE_ID IS NOT NULL
                AND FILE_ID != ''
            """
            
            cursor.execute(query)
            results = cursor.fetchall()
            
            for (file_id,) in results:
                if file_id and str(file_id).strip():
                    clean_file_id = str(file_id).strip()
                    all_file_ids.add(clean_file_id)
            
            logger.debug(f"Found {len(results)} unique file IDs in database table")
            
        except Exception as e:
            logger.error(f"Error fetching file IDs from database table {SCHEMA_NAME}.{FILE_STORAGE_TABLE}: {e}")
        
        cursor.close()
        
        try:
            if os.path.exists(DOCUMENTS_FOLDER_PATH) and os.path.isdir(DOCUMENTS_FOLDER_PATH):
                logger.debug(f"Scanning Documents folder: {DOCUMENTS_FOLDER_PATH}")
                
                for filename in os.listdir(DOCUMENTS_FOLDER_PATH):
                    file_path = os.path.join(DOCUMENTS_FOLDER_PATH, filename)
                    
                    if os.path.isfile(file_path):
                        clean_filename = filename.strip()
                        all_file_ids.add(clean_filename)
                
                logger.debug(f"Found {len([f for f in os.listdir(DOCUMENTS_FOLDER_PATH) if os.path.isfile(os.path.join(DOCUMENTS_FOLDER_PATH, f))])} files in Documents folder")
            else:
                logger.warning(f"Documents folder not found or not accessible: {DOCUMENTS_FOLDER_PATH}")
                
        except Exception as e:
            logger.error(f"Error scanning Documents folder {DOCUMENTS_FOLDER_PATH}: {e}")
        
        file_ids_list = sorted(list(all_file_ids))
        
        logger.info(f"Successfully retrieved {len(file_ids_list)} unique file IDs")
        
        if len(file_ids_list) == 0:
            logger.warning("No file IDs found in database table or Documents folder")
        else:
            logger.debug(f"File IDs preview: {file_ids_list[:5]}{'...' if len(file_ids_list) > 5 else ''}")
        
        return file_ids_list
        
    except Exception as e:
        logger.error(f"Error in get_all_file_ids: {e}")
        return []
    finally:
        if connection:
            release_db_connection(connection)


def delete_file_from_all_tables(file_name: str):
    """
    Delete a file entry from all vector store tables and file storage table.
    
    Args:
        file_name (str): The name of the file to delete from all tables
        
    Returns:
        dict: Summary of deletion results with counts and status
    """
    if not file_name or not file_name.strip():
        logger.error("File name cannot be empty or None")
        return {
            'success': False,
            'error': 'File name cannot be empty or None',
            'deleted_counts': {},
            'total_deleted': 0
        }
    
    file_name = file_name.strip()
    connection = None
    deletion_summary = {
        'success': False,
        'deleted_counts': {},
        'total_deleted': 0,
        'errors': []
    }
    
    try:
        logger.info(f"Starting deletion process for file: {file_name}")
        
        connection = get_db_connection()
        if not connection:
            error_msg = "Failed to get database connection"
            logger.error(error_msg)
            deletion_summary['error'] = error_msg
            return deletion_summary
        
        cursor = connection.cursor()
        total_deleted = 0
        
        # 1. Delete from vector store tables (TABLE_NAMES)
        logger.info("Deleting from vector store tables...")
        for store_type, table_name in TABLE_NAMES.items():
            try:
                logger.debug(f"Processing {store_type} table: {table_name}")
                
                # First, count how many records will be deleted
                count_query = f"""
                    SELECT COUNT(*) 
                    FROM {SCHEMA_NAME}.{table_name}
                    WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) = ?
                """
                cursor.execute(count_query, (file_name,))
                count_to_delete = cursor.fetchone()[0]
                
                if count_to_delete > 0:
                    # Delete the records
                    delete_query = f"""
                        DELETE FROM {SCHEMA_NAME}.{table_name}
                        WHERE JSON_VALUE(VEC_META, '$.source_file' RETURNING NVARCHAR(5000)) = ?
                    """
                    cursor.execute(delete_query, (file_name,))
                    connection.commit()
                    
                    deletion_summary['deleted_counts'][f'{store_type}_table'] = count_to_delete
                    total_deleted += count_to_delete
                    logger.info(f"✅ Deleted {count_to_delete} records from {store_type} table ({table_name})")
                else:
                    deletion_summary['deleted_counts'][f'{store_type}_table'] = 0
                    logger.debug(f"No records found in {store_type} table ({table_name}) for file: {file_name}")
                    
            except Exception as e:
                error_msg = f"Error deleting from {store_type} table ({table_name}): {str(e)}"
                logger.error(error_msg)
                deletion_summary['errors'].append(error_msg)
                deletion_summary['deleted_counts'][f'{store_type}_table'] = 0
                continue
        
        # 2. Delete from file storage table
        logger.info("Deleting from file storage table...")
        try:
            # First, count how many records will be deleted
            count_query = f"""
                SELECT COUNT(*) 
                FROM {SCHEMA_NAME}.{FILE_STORAGE_TABLE}
                WHERE FILE_ID = ?
            """
            cursor.execute(count_query, (file_name,))
            count_to_delete = cursor.fetchone()[0]
            
            if count_to_delete > 0:
                # Delete the records
                delete_query = f"""
                    DELETE FROM {SCHEMA_NAME}.{FILE_STORAGE_TABLE}
                    WHERE FILE_ID = ?
                """
                cursor.execute(delete_query, (file_name,))
                connection.commit()
                
                deletion_summary['deleted_counts']['file_storage_table'] = count_to_delete
                total_deleted += count_to_delete
                logger.info(f"✅ Deleted {count_to_delete} records from file storage table ({FILE_STORAGE_TABLE})")
            else:
                deletion_summary['deleted_counts']['file_storage_table'] = 0
                logger.debug(f"No records found in file storage table ({FILE_STORAGE_TABLE}) for file: {file_name}")
                
        except Exception as e:
            error_msg = f"Error deleting from file storage table ({FILE_STORAGE_TABLE}): {str(e)}"
            logger.error(error_msg)
            deletion_summary['errors'].append(error_msg)
            deletion_summary['deleted_counts']['file_storage_table'] = 0
        
        # 3. Try to delete physical file from Documents folder (optional)
        try:
            file_path = os.path.join(DOCUMENTS_FOLDER_PATH, file_name)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                os.remove(file_path)
                deletion_summary['physical_file_deleted'] = True
                logger.info(f"✅ Deleted physical file: {file_path}")
            else:
                deletion_summary['physical_file_deleted'] = False
                logger.debug(f"Physical file not found: {file_path}")
        except Exception as e:
            error_msg = f"Error deleting physical file: {str(e)}"
            logger.warning(error_msg)  # Warning, not error, as this is optional
            deletion_summary['errors'].append(error_msg)
            deletion_summary['physical_file_deleted'] = False
        
        cursor.close()
        
        # Summary
        deletion_summary['total_deleted'] = total_deleted
        deletion_summary['success'] = total_deleted > 0 or len(deletion_summary['errors']) == 0
        
        if total_deleted > 0:
            logger.info(f"✅ Successfully deleted file '{file_name}' - Total records deleted: {total_deleted}")
        else:
            logger.warning(f"No records found to delete for file: {file_name}")
        
        # Log detailed summary
        logger.info(f"Deletion summary for '{file_name}':")
        for table_type, count in deletion_summary['deleted_counts'].items():
            logger.info(f"  - {table_type}: {count} records")
        
        if deletion_summary['errors']:
            logger.warning(f"Errors encountered during deletion: {len(deletion_summary['errors'])}")
            for error in deletion_summary['errors']:
                logger.warning(f"  - {error}")
        
        return deletion_summary
        
    except Exception as e:
        error_msg = f"Critical error in delete_file_from_all_tables: {str(e)}"
        logger.error(error_msg)
        deletion_summary['success'] = False
        deletion_summary['error'] = error_msg
        return deletion_summary
        
    finally:
        if connection:
            release_db_connection(connection)


def retrieve_metadata_by_filename(
        filename: str,
        schema_name: str = SCHEMA_NAME,
        metadata_table: str = DUBLIN_CORE_METADATA
    ) -> typing.Dict[str, typing.Any]:
    """
    Retrieve all metadata columns from METADATA table for a given filename.
    
    Args:
        filename: The original filename to search for
        schema_name: Database schema name (default: TREASURY_AI)
        metadata_table: Metadata table name (default: TREASURY_DUBLIN_CORE_METADATA_UI5)
    
    Returns:
        Dictionary containing all metadata columns as JSON, or empty dict if not found
    """
    conn = None
    cursor = None
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query to get all columns from metadata table
        select_query = f"""
        SELECT *
        FROM "{schema_name}"."{metadata_table}"
        WHERE UPPER("ORIGINAL_FILENAME") = UPPER(?)
        """
        
        logger.debug(f"Executing metadata query for filename: {filename}")
        cursor.execute(select_query, (filename,))
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch the result
        result = cursor.fetchone()
        
        if not result:
            logger.warning(f"No metadata found for filename: {filename}")
            print(f"⚠ No metadata found for filename: {filename}")
            return {}
        
        # Convert result to dictionary
        metadata = {}
        for i, value in enumerate(result):
            column_name = column_names[i]
            
            # Handle different data types for JSON serialization
            if value is None:
                metadata[column_name] = None
            elif isinstance(value, (datetime, )):
                metadata[column_name] = value.isoformat()
            elif isinstance(value, bytes):
                # Handle BLOB data by converting to base64
                metadata[column_name] = base64.b64encode(value).decode('utf-8')
            else:
                metadata[column_name] = value
        
        logger.info(f"Retrieved metadata for {filename} with {len(metadata)} columns")
        print(f"✔ Retrieved metadata for {filename} with {len(metadata)} columns")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to retrieve metadata for {filename}: {str(e)}", exc_info=True)
        print(f"✖ Failed to retrieve metadata for {filename}: {str(e)}")
        return {}
        
    finally:
        if cursor:
            cursor.close()
        if conn:
            release_db_connection(conn)


def retrieve_metadata_as_json(
        filename: str,
        schema_name: str = SCHEMA_NAME,
        metadata_table: str = DUBLIN_CORE_METADATA
    ) -> str:
    """
    Retrieve all metadata columns as a JSON string.
    
    Args:
        filename: The original filename to search for
        schema_name: Database schema name (default: TREASURY_AI)
        metadata_table: Metadata table name (default: TREASURY_DUBLIN_CORE_METADATA_UI5)
    
    Returns:
        JSON string containing all metadata columns, or empty JSON object if not found
    """
    metadata = retrieve_metadata_by_filename(filename, schema_name, metadata_table)
    
    try:
        return json.dumps(metadata, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to serialize metadata to JSON: {str(e)}")
        return "{}"




# Enhanced cleanup registration
atexit.register(force_cleanup_all)

