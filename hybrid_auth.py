#!/usr/bin/env python3
"""
Hybrid authentication handler for the AI Daily Briefing Agent.
Supports both IMAP (IIT Delhi webmail) and Google OAuth2 (Google Calendar).

This allows the system to:
1. Fetch emails from IIT Delhi webmail using IMAP
2. Access Google Calendar using OAuth2
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from imap_auth import get_imap_credentials, validate_credentials as validate_imap_credentials

# Required scopes for Google Calendar API
GOOGLE_SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.modify'
]

# File paths for Google credentials
GOOGLE_CREDENTIALS_FILE = 'credentials.json'
GOOGLE_TOKEN_FILE = 'token.json'

def get_google_credentials():
    """
    Handle OAuth2 authentication flow for Google APIs.
    
    Returns:
        google.oauth2.credentials.Credentials: Authenticated credentials object
    
    Raises:
        FileNotFoundError: If credentials.json is not found
        Exception: If authentication flow fails
    """
    creds = None
    
    # Check if token.json exists (previously authenticated)
    if os.path.exists(GOOGLE_TOKEN_FILE):
        try:
            creds = Credentials.from_authorized_user_file(GOOGLE_TOKEN_FILE, GOOGLE_SCOPES)
            print("✓ Found existing authentication token")
        except Exception as e:
            print(f"⚠ Error loading existing token: {e}")
            # If loading fails, we'll re-authenticate
            creds = None
    
    # If there are no valid credentials available, let the user log in
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                print("🔄 Refreshing expired token...")
                creds.refresh(Request())
                print("✓ Token refreshed successfully")
            except Exception as e:
                print(f"⚠ Error refreshing token: {e}")
                print("🔐 Starting new authentication flow...")
                creds = None
        
        if not creds:
            # Check if credentials.json exists
            if not os.path.exists(GOOGLE_CREDENTIALS_FILE):
                raise FileNotFoundError(
                    f"""
{GOOGLE_CREDENTIALS_FILE} not found!

To obtain credentials.json:
1. Go to https://console.cloud.google.com/
2. Create a new project or select an existing one
3. Enable the Gmail API and Google Calendar API
4. Go to 'Credentials' and create an OAuth2 client ID for a desktop application
5. Download the credentials file and save it as '{GOOGLE_CREDENTIALS_FILE}' in this directory

Required scopes:
- https://www.googleapis.com/auth/calendar
- https://www.googleapis.com/auth/gmail.modify
                    """
                )
            
            try:
                print("🔐 Starting OAuth2 authentication flow...")
                print("📱 Your web browser will open for authentication")
                
                flow = InstalledAppFlow.from_client_secrets_file(
                    GOOGLE_CREDENTIALS_FILE, 
                    GOOGLE_SCOPES
                )
                
                # Run the OAuth flow
                creds = flow.run_local_server(
                    port=0,
                    prompt='select_account',
                    authorization_prompt_message='Please visit this URL to authorize the application: {url}',
                    success_message='Authentication successful! You can close this window.'
                )
                
                print("✓ Authentication successful!")
                
            except Exception as e:
                raise Exception(f"Authentication failed: {e}")
        
        # Save the credentials for the next run
        try:
            with open(GOOGLE_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
            print(f"✓ Credentials saved to {GOOGLE_TOKEN_FILE}")
        except Exception as e:
            print(f"⚠ Warning: Could not save credentials to {GOOGLE_TOKEN_FILE}: {e}")
    
    # Validate that we have the required scopes
    if creds.scopes:
        missing_scopes = set(GOOGLE_SCOPES) - set(creds.scopes)
        if missing_scopes:
            print(f"⚠ Warning: Missing required scopes: {missing_scopes}")
            print("🔐 Re-authentication required for additional scopes...")
            
            # Force re-authentication with all required scopes
            flow = InstalledAppFlow.from_client_secrets_file(
                GOOGLE_CREDENTIALS_FILE, 
                GOOGLE_SCOPES
            )
            creds = flow.run_local_server(port=0)
            
            with open(GOOGLE_TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
    
    print("✅ Google API credentials ready")
    return creds

def validate_google_credentials(creds):
    """
    Validate that credentials have the required scopes and are valid.
    
    Args:
        creds: Google credentials object
    
    Returns:
        bool: True if credentials are valid and have required scopes
    """
    if not creds or not creds.valid:
        return False
    
    if not creds.scopes:
        return False
    
    required_scopes = set(GOOGLE_SCOPES)
    available_scopes = set(creds.scopes)
    
    return required_scopes.issubset(available_scopes)

def get_hybrid_credentials() -> Tuple[Optional[Dict], Optional[object]]:
    """
    Get both IMAP and Google credentials for hybrid operation.
    
    Returns:
        Tuple[Optional[Dict], Optional[object]]: (imap_credentials, google_credentials)
    """
    print("🔐 Setting up hybrid authentication...")
    print("=" * 50)
    
    # Get IMAP credentials for IIT Delhi webmail
    print("\n📧 IIT Delhi Webmail Authentication")
    print("-" * 30)
    imap_creds = None
    try:
        imap_creds = get_imap_credentials()
        if validate_imap_credentials(imap_creds):
            print("✅ IMAP authentication successful")
        else:
            print("❌ IMAP authentication failed")
            imap_creds = None
    except Exception as e:
        print(f"❌ IMAP authentication error: {e}")
        imap_creds = None
    
    # Get Google credentials for Calendar
    print("\n📅 Google Calendar Authentication")
    print("-" * 30)
    google_creds = None
    try:
        google_creds = get_google_credentials()
        if validate_google_credentials(google_creds):
            print("✅ Google Calendar authentication successful")
        else:
            print("❌ Google Calendar authentication failed")
            google_creds = None
    except Exception as e:
        print(f"❌ Google Calendar authentication error: {e}")
        print("ℹ️  Calendar features will be disabled")
        google_creds = None
    
    return imap_creds, google_creds

def validate_hybrid_credentials(imap_creds: Optional[Dict], google_creds: Optional[object]) -> bool:
    """
    Validate that we have at least IMAP credentials (required for email processing).
    
    Args:
        imap_creds: IMAP credentials dictionary
        google_creds: Google OAuth2 credentials object
    
    Returns:
        bool: True if IMAP credentials are valid (minimum requirement)
    """
    if not imap_creds or not validate_imap_credentials(imap_creds):
        print("❌ IMAP credentials are required for email processing")
        return False
    
    if google_creds and validate_google_credentials(google_creds):
        print("✅ Both email and calendar authentication ready")
    else:
        print("⚠️  Email processing ready, calendar features disabled")
    
    return True

if __name__ == "__main__":
    """Test hybrid authentication."""
    print("Testing hybrid authentication...")
    try:
        imap_creds, google_creds = get_hybrid_credentials()
        if validate_hybrid_credentials(imap_creds, google_creds):
            print("\n✅ Hybrid authentication test successful!")
        else:
            print("\n❌ Hybrid authentication test failed")
    except Exception as e:
        print(f"\n❌ Hybrid authentication test failed: {e}")
