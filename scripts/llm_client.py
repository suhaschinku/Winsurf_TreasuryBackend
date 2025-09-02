import os
from dotenv import load_dotenv
from gen_ai_hub.orchestration.models.config import OrchestrationConfig
from gen_ai_hub.orchestration.models.llm import LLM
from gen_ai_hub.orchestration.models.message import UserMessage
from gen_ai_hub.orchestration.models.template import Template, TemplateValue
from gen_ai_hub.orchestration.service import OrchestrationService
from gen_ai_hub.orchestration.models.azure_content_filter import AzureContentFilter 
from logger_setup import get_logger
from typing import Optional, List, Dict
from destination_srv import get_destination_service_credentials, generate_token, fetch_destination_details, extract_aicore_credentials
from tik_tokener import count_tokens, print_token_summary

# Load environment variables
load_dotenv()

# Setup logger
logger = get_logger()

# Initialize AIC Credentials
logger.info("====> llm_client.py -> GET AIC CREDENTIALS <====")
vcap_services = os.environ.get("VCAP_SERVICES")
destination_service_credentials = get_destination_service_credentials(vcap_services)
logger.info(f"Destination Service Credentials: {destination_service_credentials}")

try:
    oauth_token = generate_token(
        uri=destination_service_credentials['dest_auth_url'] + "/oauth/token",
        client_id=destination_service_credentials['clientid'],
        client_secret=destination_service_credentials['clientsecret']
    )
    logger.info("OAuth token generated successfully for destination service.")
except Exception as e:
    logger.error(f"Error generating OAuth token: {str(e)}")
    raise

AIC_CREDENTIALS = None
dest_AIC = "GENAI_AI_CORE"
aicore_details = fetch_destination_details(
    destination_service_credentials['dest_base_url'],
    dest_AIC,
    oauth_token
)
AIC_CREDENTIALS = extract_aicore_credentials(aicore_details)
logger.info(f"AIC Credentials: {AIC_CREDENTIALS}")

# Initialize Orchestration Service
from gen_ai_hub.proxy import GenAIHubProxyClient
proxy_client = GenAIHubProxyClient(
    base_url=AIC_CREDENTIALS['aic_base_url'],
    auth_url=AIC_CREDENTIALS['aic_auth_url'],
    client_id=AIC_CREDENTIALS['clientid'],
    client_secret=AIC_CREDENTIALS['clientsecret'],
    resource_group=AIC_CREDENTIALS['resource_group']
)
ORCHESTRATION_SERVICE_URL = AIC_CREDENTIALS['ORCHESTRATION_SERVICE_URL']
ORCHESTRATION_SERVICE = OrchestrationService(api_url=ORCHESTRATION_SERVICE_URL, proxy_client=proxy_client)


# Define Azure Content Filter thresholds
CONTENT_FILTER = AzureContentFilter(hate=6, sexual=4, self_harm=0, violence=4)

# Model configuration
MODEL_CONFIG = LLM(
    name="anthropic--claude-3.5-sonnet",
    parameters={
        'temperature': 0.5,
        'max_tokens': 200000,
        'top_p': 0.9
    }
)

def run_orchestration(prompt, error_context="orchestration"):
    """Run orchestration service with content filtering."""
    try:
        if ORCHESTRATION_SERVICE is None:
            raise ValueError("OrchestrationService not initialized")
        
        template = Template(messages=[UserMessage("{{ ?extraction_prompt }}")])
        config = OrchestrationConfig(template=template, llm=MODEL_CONFIG)
        config.input_filter = CONTENT_FILTER
        config.output_filter = CONTENT_FILTER
        
        logger.debug(f"Running {error_context} with prompt: {prompt[:100]}...")
        response = ORCHESTRATION_SERVICE.run(
            config=config,
            template_values=[TemplateValue("extraction_prompt", prompt)]
        )
        
        result = response.orchestration_result.choices[0].message.content
        logger.debug(f"Completed {error_context} with result: {result[:100]}...")

        # Token counting
        input_tokens = count_tokens(prompt)
        output_tokens = count_tokens(result)
        total_tokens = input_tokens + output_tokens
        logger.info(f"Token usage for {error_context}: input={input_tokens}, output={output_tokens}, total={total_tokens}")
        print_token_summary({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens
        })

        return result
        
    except Exception as e:
        logger.error(f"Error in {error_context}: {str(e)}", exc_info=True)
        raise Exception(f"Error in {error_context}: {str(e)}")

def execute_coda_analysis(coda_prompt):
    """Execute CODA analysis."""
    return run_orchestration(coda_prompt, error_context="CODA analysis")

def execute_aika_analysis(coda_prompt):
    """Execute aika analysis."""
    return run_orchestration(coda_prompt, error_context="aika analysis")      

def extract_data_requirements(coda_result):
    """Extract data requirements from CODA analysis."""
    prompt = f"<prompt> extract the data requirements portion of the following text </prompt> <text> {coda_result} </text>"
    return run_orchestration(prompt, error_context="data requirements extraction")

def execute_final_analysis(final_prompt):
    """Execute final analysis."""
    return run_orchestration(final_prompt, error_context="final analysis")

def extract_analysis_steps(coda_result):
    """Extract analysis steps from CODA analysis."""
    prompt = f"<prompt> extract STRICTLY the required analysis and data requirements portion of the following text </prompt> <text> {coda_result} </text>"
    return run_orchestration(prompt, error_context="analysis steps extraction")

def extract_topics(transcript_text: str) -> str:
    """Extract top 5 topics with weights and keywords in HTML format."""
    prompt = f"""<prompt> You are tasked with extracting the top 5 topics discussed in the provided transcript text (e.g., earnings call QnA). For each topic, provide a percentage weightage (summing exactly to 100% across all topics) and 3-5 associated keywords (comma-separated). Output the result STRICTLY in HTML format as an ordered list (<ol>), with each list item (<li>) formatted exactly as: "Topic: <topic_name>, Weight: <percentage>%, Keywords: <keyword1>,<keyword2>,<keyword3>,...".

    Strict Requirements:
    - Analyze ONLY the provided transcript text. Do NOT use external information or infer beyond the text.
    - Identify EXACTLY 5 topics unless fewer distinct topics are present, in which case list only those available and adjust percentages to sum to 100%.
    - If no topics can be identified (e.g., text is too short or unclear), return: <ol><li>Topic: None, Weight: 100%, Keywords: none</li></ol>
    - Topics should reflect key discussion themes (e.g., strategy, operations, innovation), NOT financial metrics (e.g., revenue, profit, stock price, earnings).
    - EXCLUDE summaries, introductions, notifications, caveats, or speaker names in topic descriptions.
    - Ensure percentages are integers and sum to 100%.
    - Keywords should be specific, relevant terms from the transcript (e.g., 'cloud computing', 'AI', 'sustainability').
    - Output ONLY the HTML ordered list (<ol>...</ol>) with no additional text, comments, or explanations.
    - Ensure valid HTML syntax.
    - If the transcript is noisy or fragmented, prioritize coherent themes based on frequency and context.

    Example Output:
    <ol>
        <li>Topic: Cloud Computing, Weight: 30%, Keywords: cloud, infrastructure, services, expansion</li>
        <li>Topic: AI Development, Weight: 25%, Keywords: artificial intelligence, machine learning, algorithms</li>
        <li>Topic: Customer Engagement, Weight: 20%, Keywords: user experience, feedback, retention</li>
        <li>Topic: Product Innovation, Weight: 15%, Keywords: new features, development, launches</li>
        <li>Topic: Sustainability, Weight: 10%, Keywords: eco-friendly, green tech, renewable</li>
    </ol>

    </prompt> <text> {transcript_text} </text>"""
    return run_orchestration(prompt, error_context="data topic extraction")

def data_formatter(final_result: str, excel_final_result: str, Image_Result: Optional[List[Dict[str, str]]] = None) -> str:
    """Format final response as HTML with executive summary, main content, stock analysis, and Excel data."""
    stock_section = ""
    if Image_Result:
        stock_section = "<h2>Stock Analysis</h2><div>"
        for result in Image_Result:
            analysis = result.get("analysis", "No stock analysis available")
            stock_section += f"<p>{analysis}</p>"
        stock_section += "</div>"

    prompt = f"""<prompt>  # CRITICAL: LLM Instructions - Follow Every Step

## STEP 1: Output Format Rules
- Output ONLY HTML content - no introductory text
- Do NOT start with phrases like "Here is..." or "Based on..."
- Begin directly with the HTML content

## STEP 2: Required Structure (Must Include Both)

### A) Executive Summary Section
```html
<p style="color:gray;"><em>✨ Generated by FinSight.Intelligence. Please review before use.</em></p>
<h2>Executive Summary</h2>
<p>[Write 1 paragraph of flowing sentences with key numbers. NO bullet points or lists.]</p>
```

### B) Main Content Section
```html
<h2>[Content Title]</h2>
[Include ALL data including Excel data here - use bullet points where appropriate but integrate everything]
```

## STEP 3: Content Rules
- **Executive Summary**: Write as ONE flowing paragraph with complete sentences
- **Main Content**: Include ALL source data (Excel + other) - do NOT create separate Excel section
- **User Instructions**: If user specifies anything explicitly, follow it exactly
- **Scope**: Only include what user asks for - no extra information

## STEP 4: Number Formatting (Always Use)
- Billion = bn
- Million = mn
- Year over Year = YoY
- Quarter over Quarter = QoQ

## STEP 5: Exclusions
- Remove CODA Analysis sections
- No separate Excel data section
- No introductory text before HTML

## VERIFICATION CHECKLIST
Before submitting, confirm:
- [ ] Started directly with HTML (no "Here is..." text)
- [ ] Executive summary is ONE paragraph, not bullet points
- [ ] All Excel data integrated into main content
- [ ] Used bn/mn/YoY/QoQ formatting
- [ ] Followed any explicit user instructions
- [ ] Included the FinSight disclaimer line

## EXAMPLE START:
```html
<p style="color:gray;"><em>✨ Generated by FinSight.Intelligence. Please review before use.</em></p>
<h2>Executive Summary</h2>
<p>The company demonstrated strong performance with revenue growing 15% YoY to $2.3bn, driven primarily by...</p>
```

**REMEMBER: Start your response directly with the HTML - no explanatory text first.** 
    </prompt> <text> {final_result} </text> <stock_section> {stock_section} </stock_section> <excel_section> {excel_final_result} </excel_section>"""   
    return run_orchestration(prompt, error_context="data formatting")