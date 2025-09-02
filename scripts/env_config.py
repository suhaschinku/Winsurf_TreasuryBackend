import os
import json
from dotenv import load_dotenv

# Table names for HANA DB
TABLE_NAMES = {
    "transcript": "TREASURY_EMBEDD_TRANSCRIPT_UI5",
    "non_transcript": "TREASURY_EMBEDD_NON_TRANSCRIPT_UI5",
    "excel_non_transcript": "TREASURY_EMBEDD_EXCEL_NON_TRANSCRIPT_UI5"
}

#Hana Treasury Table
TREASURY_TABLE = "TREASURY_DANS_FACT_TABLE_UI5"
TREASURY_SUMMARY_TABLE = "TREASURY_SUMMARY_FILE_STORAGE_UI5"
SCHEMA_NAME = "TREASURY_AI"
DUBLIN_CORE_METADATA = "TREASURY_DUBLIN_CORE_METADATA_UI5"

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-large"
HANA_DB_API ="3cb8ff87-b67f-4b68-8106-e297566641ef.hana.prod-ap11.hanacloud.ondemand.com"

# Bedrock model configuration
MODEL_ID = "anthropic--claude-3.5-sonnet"

# Known banks as JSON string with code-name pairs
KNOWN_BANKS_JSON = '''
{
    "JPMC": "JP Morgan",
    "MS": "Morgan Stanley",
    "GS": "Goldman Sachs",
    "C": "Citi",
    "BAC": "Bank of America",
    "BNP": "BNP Paribas",
    "DBK": "Deutsche Bank",
    "HSBC": "HSBC",
    "BBVA": "Banco Bilbao Vizcaya Argentaria",
    "BCS": "Barclays",
    "SAN": "Banco Santander",
    "UBSG": "UBS Group",
    "ING": "ING Bank",
    "SCB": "Standard Chartered",
    "DBS": "DBS"
}
'''

# Supported image extensions
IMAGE_EXTENSIONS = ["jpeg", "png"]

def get_known_banks() -> dict:
    """
    Load and parse the KNOWN_BANKS_JSON into a dictionary.
    
    Returns:
        dict: Dictionary with bank codes as keys and names as values.
    """
    try:
        return json.loads(KNOWN_BANKS_JSON)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse KNOWN_BANKS_JSON: {str(e)}")

def load_config():
    """Load environment variables from .env file."""
    load_dotenv()

def get_documents_dir_path():
    """Get default documents directory path."""
    return os.path.join(load_config()['local_path'], "Documents")

def get_default_schema():
    return os.getenv('DEF_SCHEMA', 'TREASURY_AI')

DEF_SCHEMA = get_default_schema()    