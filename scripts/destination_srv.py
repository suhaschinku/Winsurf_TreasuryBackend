from logger_setup import get_logger
import requests
import json
import os
from dotenv import load_dotenv

# Set up logger
logger = get_logger()

# Step 1: Load environment variables from CF VCAP_SERVICES
def get_destination_service_credentials(vcap_services):
# Function common to both local and cloud environments
# Function to extract destination service credentials from VCAP_SERVICES
    vcap_services = json.loads(vcap_services)
    if not isinstance(vcap_services, dict):
        raise ValueError("VCAP_SERVICES could not be loaded as a dictionary.")
    # Directly access the 'destination' service
    destination_services = vcap_services.get('destination')
    if destination_services and isinstance(destination_services, list) and len(destination_services) > 0:
        creds = destination_services[0].get('credentials', {})
        if all([creds.get('url'), creds.get('clientid'), creds.get('clientsecret'), creds.get('uri')]):
            return {
                'dest_auth_url': creds['url'],
                'clientid': creds['clientid'],
                'clientsecret': creds['clientsecret'],
                'dest_base_url': creds['uri']
            }
    else:
        logger.info("VCAP_SERVICES not found in environment")
        return None

# Step 2: Generate Token for Destination Services
def generate_token(uri, client_id, client_secret):
    response = requests.post(uri, data={'grant_type': 'client_credentials'}, auth=(client_id, client_secret))
    response.raise_for_status()
    logger.info("OAuth token generated successfully.")
    # logger.info(f"Token: {response.json()['access_token']}")
    return response.json()['access_token']

# Step 3: Get Hana DataBase Details by passing Service Name
def fetch_destination_details(uri, name, token):
    url = f"{uri}/destination-configuration/v1/destinations/{name}"
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.get(url, headers=headers)
    # response.raise_for_status()
    return response.json()

# Step 4: Extract HANA connection details
def extract_hana_credentials(config):
    dc = config.get('destinationConfiguration', {})
    url = dc.get('URL', '')
    # Remove http:// or https:// if present
    address = url.replace("https://", "").replace("http://", "")
    return {
        'address': address,
        'user': dc.get('User'),
        'password': dc.get('Password'),
        'port': dc.get('Port'),
        #'schema': dc.get('schema')
        'schema': 'TREASURY_AI'
    }

# Step 5: Extract AI Core connection details
def extract_aicore_credentials(config):
    dc = config.get('destinationConfiguration', {})
    return {
        'aic_base_url': dc.get('URL', 'url'),
        'clientid': dc.get('clientId', 'ClientId'),
        'clientsecret': dc.get('clientSecret', 'ClientSecret'),
        'aic_auth_url': dc.get('tokenServiceURL'),
        'resource_group': dc.get('resourceGroup'),
        'ORCHESTRATION_SERVICE_URL': dc.get('ORCHESTRATION_SERVICE_URL')
   }
# Step 6: Extract CAP Credentials
def extract_cap_credentials(config):
    dc = config.get('destinationConfiguration', {})
    return {
        'cap_base_url': dc.get('URL', 'url'),
        'cap_clientid': dc.get('clientId', 'ClientId'),
        'cap_clientsecret': dc.get('clientSecret', 'ClientSecret'),
        'cap_auth_url': dc.get('tokenServiceURL'),
        'resource_group': dc.get('resourceGroup')
    }