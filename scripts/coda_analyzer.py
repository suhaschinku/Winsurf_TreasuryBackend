from dotenv import load_dotenv
import os
from gen_ai_hub.proxy.native.amazon.clients import Session

load_dotenv()

localpath = os.getenv('LOCALPATH', os.getcwd())
filepath = os.path.join(localpath, "prompt_CODA.txt")

def generate_coda_prompt(input_text):
    """Generate a CODA analysis prompt from a template."""
    
    filename = os.path.abspath(filepath)
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            coda_text = file.read()
        combined_text = "Analyse the prompt using the method and come up with clear steps to analyse and data needed to perform the analysis. <method> " + coda_text + "</method>" + "<prompt> " + input_text + "</prompt> "
        return combined_text
    except Exception as e:
        raise Exception(f"An error occurred: {str(e)}")