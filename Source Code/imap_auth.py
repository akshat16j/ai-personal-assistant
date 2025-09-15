#!/usr/bin/env python3
"""
IMAP authentication handler for IIT Delhi webmail.
Handles credential management and validation for IMAP connections.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# IIT Delhi webmail IMAP settings
IMAP_CONFIG = {
    'server': 'mailstore.iitd.ac.in',
    'port': 993,
    'security': 'SSL/TLS'
}

# File paths for credentials
CREDENTIALS_FILE = 'imap_credentials.json'

def get_imap_credentials() -> Dict:
    """
    Get IMAP credentials from creds.env file.
    
    Returns:
        Dict: IMAP credentials dictionary
    
    Raises:
        FileNotFoundError: If credentials file is not found
        Exception: If authentication fails
    """
    # Load environment variables from creds.env
    load_dotenv('creds.env')
    
    # Get credentials from environment variables
    username = os.getenv('WEBMAIL_USER')
    password = os.getenv('WEBMAIL_PASSWORD')
    
    if not username or not password:
        raise Exception("Missing credentials in creds.env file. Please ensure WEBMAIL_USER and WEBMAIL_PASSWORD are set.")
    
    # Automatically append @iitd.ac.in to username if not already present
    if not username.endswith('@iitd.ac.in'):
        username = f"{username}@iitd.ac.in"
        print(f"✓ Converted username to: {username}")
    
    credentials = {
        'server': IMAP_CONFIG['server'],
        'port': IMAP_CONFIG['port'],
        'username': username,
        'password': password
    }
    
    print("🔐 IIT Delhi Webmail Authentication")
    print("=" * 40)
    print(f"Server: {IMAP_CONFIG['server']}")
    print(f"Port: {IMAP_CONFIG['port']}")
    print(f"Security: {IMAP_CONFIG['security']}")
    print(f"Username: {username}")
    print()
    
    # Test connection
    print("🔄 Testing connection...")
    if test_imap_connection(credentials):
        print("✅ IMAP authentication successful")
        return credentials
    else:
        raise Exception("Connection test failed")

def test_imap_connection(credentials: Dict) -> bool:
    """
    Test IMAP connection with provided credentials.
    
    Args:
        credentials: IMAP credentials dictionary
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        from tools.imap_tool import IMAPEmailClient
        
        client = IMAPEmailClient(
            server=credentials['server'],
            port=credentials['port'],
            username=credentials['username'],
            password=credentials['password']
        )
        
        success = client.connect()
        if success:
            client.disconnect()
        return success
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False

def validate_credentials(credentials: Dict) -> bool:
    """
    Validate that credentials have the required fields and are valid.
    
    Args:
        credentials: IMAP credentials dictionary
    
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    if not credentials:
        return False
    
    required_fields = ['server', 'port', 'username', 'password']
    if not all(field in credentials for field in required_fields):
        return False
    
    # Test connection
    return test_imap_connection(credentials)

def refresh_credentials_if_needed(credentials: Dict) -> Optional[Dict]:
    """
    Refresh credentials if needed (for IMAP, this is a no-op as we don't have tokens).
    
    Args:
        credentials: IMAP credentials dictionary
    
    Returns:
        Optional[Dict]: Refreshed credentials (same as input for IMAP)
    """
    # IMAP doesn't use tokens, so no refresh needed
    return credentials

if __name__ == "__main__":
    """Test IMAP authentication."""
    print("Testing IIT Delhi webmail authentication...")
    try:
        credentials = get_imap_credentials()
        if validate_credentials(credentials):
            print("✅ Authentication test successful!")
            print(f"Server: {credentials['server']}")
            print(f"Port: {credentials['port']}")
            print(f"Username: {credentials['username']}")
        else:
            print("❌ Authentication test failed - invalid credentials")
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
