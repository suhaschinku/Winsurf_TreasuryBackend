import os
import json
import requests
from logger_setup import get_logger
from sap import xssec
from fastapi import HTTPException
import jwt
from flask import request, jsonify
from functools import wraps

logger = get_logger()


def get_xsuaa_credentials(vcap_services):
    print(f"VCAP_SERVICES: {vcap_services}")
    vcap_services = json.loads(vcap_services)
    xsuaa = vcap_services["xsuaa"][0]["credentials"]
    xsuaa_clientid = xsuaa.get("clientid")
    xsuaa_clientsecret = xsuaa.get("clientsecret")
    xsuaa_authurl = xsuaa.get("url")
    xsuaa_appname = xsuaa.get("xsappname")

    if not xsuaa_clientid or not xsuaa_clientsecret or not xsuaa_authurl:
        raise ValueError("Missing XSUAA credentials in VCAP_SERVICES.")
    else:
        logger.info("XSUAA credentials successfully retrieved from VCAP_SERVICES.")
        logger.debug(f"Client ID: {xsuaa_clientid}, Auth URL: {xsuaa_authurl}")
        uaa_credentials = {
        "clientid": xsuaa_clientid,
        "clientsecret": xsuaa_clientsecret,
        "url": xsuaa_authurl,
        "xsappname": xsuaa_appname
    }
    return uaa_credentials



def verify_jwt_token(token, uaa_credentials, required_scope="uaa.resource"):
    try:
        decoded = jwt.decode(token, options={"verify_signature": False})
        logger.debug(f"Decoded JWT: {decoded}")
        security_context = xssec.create_security_context(token, uaa_credentials)
        if not security_context.check_scope(required_scope):
            logger.warning(f"Token missing required scope: {required_scope}")
            raise HTTPException(status_code=403, detail="Forbidden: Insufficient scope")
        logger.info("Token successfully verified and scope present.")
        return security_context
    except HTTPException:
        raise
    except Exception as e:
        logger.info(f"Token verification failed: {e}")
        raise HTTPException(status_code=403, detail=f"Token verification failed: {str(e)}")

def require_auth(f):
    """
    Flask decorator to enforce XSUAA JWT authentication and scope check on endpoints.
    - Expects 'Authorization: Bearer <token>' header in the request.
    - Uses global 'uaa_xsuaa_credentials' (must be set in the main app) for verification.
    - Returns 401 if header is missing/invalid, 403 if token is invalid or lacks required scope.
    - On success, calls the wrapped endpoint.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        from flask import current_app
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authorization header missing or invalid'}), 401
        token = auth_header.replace('Bearer ', '', 1)
        # Get credentials from Flask app context
        uaa_credentials = getattr(current_app, 'uaa_xsuaa_credentials', None)
        if not uaa_credentials:
            return jsonify({'error': 'XSUAA credentials not configured'}), 500
        try:
            verify_jwt_token(token, uaa_credentials)
        except HTTPException as e:
            return jsonify({'error': str(e.detail)}), e.status_code
        return f(*args, **kwargs)
    return decorated