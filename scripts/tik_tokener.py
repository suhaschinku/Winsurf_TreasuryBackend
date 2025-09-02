from dotenv import load_dotenv
from logger_setup import get_logger
import tiktoken
from gen_ai_hub.proxy.native.amazon.clients import Session

# Load environment variables (if needed)
load_dotenv()
logger = get_logger()

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a text string using tiktoken.
    """
    try:
        encoder = tiktoken.get_encoding("cl100k_base")
        tokens = encoder.encode(text)
        return len(tokens)
    except Exception as e:
        logger.info(f"Token counting failed: {str(e)}")
        return 0

def print_token_summary(result: dict) -> None:
    """
    Print a formatted summary of token usage.
    """
    print("\n" + "="*50)
    print("TOKEN USAGE SUMMARY")
    print("="*50)
    print(f"Input Tokens:  {result.get('input_tokens', 0):,}")
    print(f"Output Tokens: {result.get('output_tokens', 0):,}")
    print(f"Total Tokens:  {result.get('total_tokens', 0):,}")
    print("="*50 + "\n")