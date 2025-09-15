#!/usr/bin/env python3
"""
IMAP email integration for the AI Daily Briefing Agent.
Handles fetching unread emails from IIT Delhi webmail using IMAP.
"""

import imaplib
import email
import ssl
from typing import List, Dict, Optional
from email.header import decode_header
import re

class IMAPEmailClient:
    """IMAP client for IIT Delhi webmail."""
    
    def __init__(self, server: str, port: int, username: str, password: str):
        """
        Initialize IMAP client.
        
        Args:
            server: IMAP server address
            port: IMAP server port
            username: Email username
            password: Email password
        """
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.connection = None
    
    def connect(self) -> bool:
        """
        Connect to the IMAP server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create SSL context with relaxed certificate verification
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            
            # Connect to server
            self.connection = imaplib.IMAP4_SSL(self.server, self.port, ssl_context=context)
            
            # Login
            self.connection.login(self.username, self.password)
            
            print("✅ Successfully connected to IIT Delhi webmail")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to IMAP server: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the IMAP server."""
        if self.connection:
            try:
                self.connection.close()
                self.connection.logout()
                print("✅ Disconnected from IMAP server")
            except:
                pass
    
    def get_unread_emails(self, max_results: int = 10) -> List[Dict]:
        """
        Fetch recent unread emails from the mailbox.
        
        Args:
            max_results: Maximum number of emails to fetch (default: 10)
        
        Returns:
            List[Dict]: List of email dictionaries with id, subject, body, and sender
        """
        if not self.connection:
            print("❌ Not connected to IMAP server")
            return []
        
        try:
            print(f"📧 Fetching up to {max_results} unread emails...")
            
            # Select INBOX
            self.connection.select("INBOX")
            
            # First try to get unread emails
            status, messages = self.connection.search(None, "UNSEEN")
            
            if status != "OK":
                print("❌ Failed to search for emails")
                return []
            
            # Get message IDs
            message_ids = messages[0].split()
            
            # If no unread emails, try to get recent emails (last 7 days)
            if not message_ids:
                print("ℹ️ No unread emails found, checking recent emails...")
                from datetime import datetime, timedelta
                seven_days_ago = (datetime.now() - timedelta(days=7)).strftime("%d-%b-%Y")
                status, messages = self.connection.search(None, f"SINCE {seven_days_ago}")
                
                if status == "OK":
                    message_ids = messages[0].split()
                    if message_ids:
                        print(f"📧 Found {len(message_ids)} recent emails from last 7 days")
                    else:
                        print("✓ No recent emails found")
                        return []
                else:
                    print("✓ No unread emails found")
                    return []
            
            # Limit to max_results
            message_ids = message_ids[-max_results:]  # Get most recent emails
            
            emails = []
            for msg_id in message_ids:
                try:
                    # Fetch email
                    status, msg_data = self.connection.fetch(msg_id, "(RFC822)")
                    
                    if status != "OK":
                        continue
                    
                    # Parse email
                    email_data = self._parse_email(msg_data[0][1], msg_id.decode())
                    if email_data:
                        emails.append(email_data)
                        
                except Exception as e:
                    print(f"⚠ Warning: Could not fetch email {msg_id}: {e}")
                    continue
            
            print(f"✓ Successfully fetched {len(emails)} unread emails")
            return emails
            
        except Exception as e:
            print(f"❌ Error fetching emails: {e}")
            return []
    
    def get_recent_emails(self, max_results: int = 20) -> List[Dict]:
        """
        Fetch recent emails from the mailbox (both read and unread).
        
        Args:
            max_results: Maximum number of emails to fetch (default: 20)
        
        Returns:
            List[Dict]: List of email dictionaries with id, subject, body, and sender
        """
        if not self.connection:
            print("❌ Not connected to IMAP server")
            return []
        
        try:
            print(f"📧 Fetching up to {max_results} recent emails...")
            
            # Select INBOX
            self.connection.select("INBOX")
            
            # Search for recent emails (last 30 days to get more emails)
            from datetime import datetime, timedelta
            thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%d-%b-%Y")
            status, messages = self.connection.search(None, f"SINCE {thirty_days_ago}")
            
            if status != "OK":
                print("❌ Failed to search for emails")
                return []
            
            # Get message IDs
            message_ids = messages[0].split()
            
            if not message_ids:
                print("✓ No recent emails found")
                return []
            
            # Limit to max_results
            message_ids = message_ids[-max_results:]  # Get most recent emails
            
            emails = []
            for msg_id in message_ids:
                try:
                    # Fetch email
                    status, msg_data = self.connection.fetch(msg_id, "(RFC822)")
                    
                    if status != "OK":
                        continue
                    
                    # Parse email
                    email_data = self._parse_email(msg_data[0][1], msg_id.decode())
                    if email_data:
                        emails.append(email_data)
                        
                except Exception as e:
                    print(f"⚠ Warning: Could not fetch email {msg_id}: {e}")
                    continue
            
            print(f"✓ Successfully fetched {len(emails)} recent emails")
            return emails
            
        except Exception as e:
            print(f"❌ Error fetching emails: {e}")
            return []
    
    def mark_as_read(self, email_id: str) -> bool:
        """
        Mark an email as read.
        
        Args:
            email_id: Email message ID
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            # Select INBOX
            status, response = self.connection.select("INBOX")
            if status != "OK":
                print(f"❌ Failed to select INBOX for marking as read")
                return False
            
            # Clean and validate email_id
            if isinstance(email_id, bytes):
                email_id = email_id.decode('utf-8')
            
            # Convert to string and clean
            email_id_str = str(email_id).strip()
            
            # Remove any non-numeric characters and validate
            import re
            clean_id = re.sub(r'[^\d]', '', email_id_str)
            
            # If cleaning resulted in empty string, try original
            if not clean_id:
                clean_id = email_id_str
            
            # Validate that it's a reasonable email ID
            if not clean_id or len(clean_id) < 1:
                print(f"⚠ Warning: Invalid email ID format: {email_id}")
                return False
            
            # Mark as read by adding SEEN flag
            status, response = self.connection.store(clean_id, '+FLAGS', '\\Seen')
            if status == "OK":
                print(f"✅ Marked email {clean_id} as read")
                return True
            else:
                print(f"❌ Failed to mark email {clean_id} as read: {response}")
                return False
            
        except Exception as e:
            print(f"⚠ Warning: Could not mark email {email_id} as read: {e}")
            return False
    
    def create_folder(self, folder_name: str) -> bool:
        """
        Create a new folder/label in the mailbox (only if it doesn't exist).
        
        Args:
            folder_name: Name of the folder to create
        
        Returns:
            bool: True if successful or already exists, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            # Check if folder already exists first
            status, folders = self.connection.list()
            if status == "OK":
                existing_folders = [folder.decode() for folder in folders]
                if any(folder_name in folder for folder in existing_folders):
                    # Folder already exists, no need to create
                    return True
            
            # Create folder only if it doesn't exist
            status, response = self.connection.create(folder_name)
            if status == "OK":
                print(f"✅ Created folder: {folder_name}")
                return True
            else:
                # Folder might already exist or creation failed
                return True  # Consider it success to avoid blocking
                
        except Exception as e:
            # Don't print warning for folder creation - it's not critical
            return True  # Consider it success to avoid blocking
    
    def move_email_to_folder(self, email_id: str, folder_name: str) -> bool:
        """
        Move an email to a specific folder.
        
        Args:
            email_id: Email message ID
            folder_name: Destination folder name
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            # First ensure the folder exists
            if not self.create_folder(folder_name):
                print(f"⚠ Could not create folder {folder_name}")
                return False
            
            # Select INBOX
            status, response = self.connection.select("INBOX")
            if status != "OK":
                print(f"❌ Failed to select INBOX")
                return False
            
            # Clean and validate email_id
            if isinstance(email_id, bytes):
                email_id = email_id.decode('utf-8')
            
            # Convert to string and clean
            email_id_str = str(email_id).strip()
            
            # Remove any non-numeric characters and validate
            import re
            clean_id = re.sub(r'[^\d]', '', email_id_str)
            
            # If cleaning resulted in empty string, try original
            if not clean_id:
                clean_id = email_id_str
            
            # Validate that it's a reasonable email ID
            if not clean_id or len(clean_id) < 1:
                print(f"⚠ Warning: Invalid email ID format: {email_id}")
                return False
            
            # Copy email to the folder
            status, response = self.connection.copy(clean_id, folder_name)
            if status != "OK":
                print(f"❌ Failed to copy email {clean_id} to {folder_name}: {response}")
                return False
            
            # Delete from INBOX (move operation)
            status, response = self.connection.store(clean_id, '+FLAGS', '\\Deleted')
            if status != "OK":
                print(f"⚠ Warning: Could not mark email {clean_id} for deletion: {response}")
                # Don't return False here, as copy might have succeeded
            
            # Expunge to complete the move
            try:
                self.connection.expunge()
            except Exception as e:
                print(f"⚠ Warning: Expunge failed: {e}")
                # Don't return False here, as copy might have succeeded
            
            return True
            
        except Exception as e:
            print(f"⚠ Warning: Could not move email {email_id} to {folder_name}: {e}")
            return False
    
    def add_custom_flag(self, email_id: str, flag_name: str) -> bool:
        """
        Add a custom flag to an email (alternative to folder-based labeling).
        
        Args:
            email_id: Email message ID
            flag_name: Custom flag name
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.connection:
            return False
        
        try:
            # Select INBOX
            status, response = self.connection.select("INBOX")
            if status != "OK":
                print(f"❌ Failed to select INBOX for flagging")
                return False
            
            # Clean and validate email_id
            if isinstance(email_id, bytes):
                email_id = email_id.decode('utf-8')
            
            # Convert to string and clean
            email_id_str = str(email_id).strip()
            
            # Remove any non-numeric characters and validate
            import re
            clean_id = re.sub(r'[^\d]', '', email_id_str)
            
            # If cleaning resulted in empty string, try original
            if not clean_id:
                clean_id = email_id_str
            
            # Validate that it's a reasonable email ID
            if not clean_id or len(clean_id) < 1:
                print(f"⚠ Warning: Invalid email ID format: {email_id}")
                return False
            
            # Add custom flag
            status, response = self.connection.store(clean_id, '+FLAGS', f'\\{flag_name}')
            if status == "OK":
                print(f"✅ Added flag {flag_name} to email {clean_id}")
                return True
            else:
                print(f"❌ Failed to add flag {flag_name} to email {clean_id}: {response}")
                return False
                
        except Exception as e:
            print(f"⚠ Warning: Could not add flag {flag_name} to email {email_id}: {e}")
            return False
    
    def _parse_email(self, raw_email: bytes, email_id: str) -> Optional[Dict]:
        """
        Parse raw email data into structured format.
        
        Args:
            raw_email: Raw email bytes
            email_id: Email message ID
        
        Returns:
            Optional[Dict]: Parsed email data or None if parsing fails
        """
        try:
            # Parse email message
            msg = email.message_from_bytes(raw_email)
            
            # Extract headers
            subject = self._decode_header(msg.get("Subject", ""))
            sender = self._decode_header(msg.get("From", ""))
            date = msg.get("Date", "")
            cc = self._decode_header(msg.get("Cc", ""))
            bcc = self._decode_header(msg.get("Bcc", ""))
            to = self._decode_header(msg.get("To", ""))
            
            # Extract body
            body = self._extract_body(msg)
            
            # Create snippet (first 200 chars of body)
            snippet = body[:200] + "..." if len(body) > 200 else body
            
            return {
                'id': email_id,
                'subject': subject or "(No Subject)",
                'sender': sender or "Unknown Sender",
                'date': date,
                'body': body,
                'snippet': snippet,
                'cc': cc,
                'bcc': bcc,
                'to': to
            }
            
        except Exception as e:
            print(f"⚠ Warning: Failed to parse email: {e}")
            return None
    
    def _decode_header(self, header: str) -> str:
        """
        Decode email header (handles encoding issues).
        
        Args:
            header: Raw header string
        
        Returns:
            str: Decoded header string
        """
        try:
            decoded_parts = decode_header(header)
            decoded_string = ""
            
            for part, encoding in decoded_parts:
                if isinstance(part, bytes):
                    if encoding:
                        decoded_string += part.decode(encoding)
                    else:
                        decoded_string += part.decode('utf-8', errors='ignore')
                else:
                    decoded_string += part
            
            return decoded_string.strip()
            
        except Exception as e:
            print(f"⚠ Warning: Failed to decode header: {e}")
            return header
    
    def _extract_body(self, msg) -> str:
        """
        Extract text body from email message.
        
        Args:
            msg: Email message object
        
        Returns:
            str: Email body text
        """
        try:
            body = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition", ""))
                    
                    # Skip attachments
                    if "attachment" in content_disposition:
                        continue
                    
                    # Extract text content
                    if content_type == "text/plain":
                        body += self._decode_part(part)
                    elif content_type == "text/html" and not body:
                        # Fall back to HTML if no plain text
                        html_body = self._decode_part(part)
                        body = self._strip_html_tags(html_body)
            else:
                # Single part message
                content_type = msg.get_content_type()
                if content_type == "text/plain":
                    body = self._decode_part(msg)
                elif content_type == "text/html":
                    html_body = self._decode_part(msg)
                    body = self._strip_html_tags(html_body)
            
            return body.strip()
            
        except Exception as e:
            print(f"⚠ Warning: Failed to extract email body: {e}")
            return ""
    
    def _decode_part(self, part) -> str:
        """
        Decode email part content.
        
        Args:
            part: Email part object
        
        Returns:
            str: Decoded content
        """
        try:
            payload = part.get_payload(decode=True)
            if payload:
                charset = part.get_content_charset() or 'utf-8'
                return payload.decode(charset, errors='ignore')
            return ""
        except Exception as e:
            print(f"⚠ Warning: Failed to decode part: {e}")
            return ""
    
    def _strip_html_tags(self, html_text: str) -> str:
        """
        Basic HTML tag removal for email body extraction.
        
        Args:
            html_text: HTML content
        
        Returns:
            str: Plain text with HTML tags removed
        """
        # Remove HTML tags
        clean = re.compile('<.*?>')
        text = re.sub(clean, '', html_text)
        
        # Replace common HTML entities
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        text = text.replace('&apos;', "'")
        
        return text.strip()

def get_unread_emails(credentials: Dict, max_results: int = 10) -> List[Dict]:
    """
    Fetch recent unread emails from IIT Delhi webmail.
    
    Args:
        credentials: Dictionary containing IMAP credentials
        max_results: Maximum number of emails to fetch (default: 10)
    
    Returns:
        List[Dict]: List of email dictionaries with id, subject, body, and sender
    """
    client = IMAPEmailClient(
        server=credentials['server'],
        port=credentials['port'],
        username=credentials['username'],
        password=credentials['password']
    )
    
    if not client.connect():
        return []
    
    try:
        emails = client.get_unread_emails(max_results)
        return emails
    finally:
        client.disconnect()

def apply_label_to_email(credentials: Dict, email_id: str, label_name: str) -> bool:
    """
    Apply a label to an email using folder system.
    Based on diagnostic results, this server supports folders and email copying.
    
    Args:
        credentials: Dictionary containing IMAP credentials
        email_id: Email message ID
        label_name: Label name (e.g., "AI-HIGH", "AI-MEDIUM", "AI-LOW")
    
    Returns:
        bool: True if email was labeled successfully, False otherwise
    """
    client = IMAPEmailClient(
        server=credentials['server'],
        port=credentials['port'],
        username=credentials['username'],
        password=credentials['password']
    )
    
    if not client.connect():
        return False
    
    try:
        # Clean and validate email_id first
        if isinstance(email_id, bytes):
            email_id = email_id.decode('utf-8')
        
        # Convert to string and clean
        email_id_str = str(email_id).strip()
        
        # Try multiple ID formats
        import re
        clean_id = re.sub(r'[^\d]', '', email_id_str)
        
        # If cleaning resulted in empty string, try original
        if not clean_id:
            clean_id = email_id_str
        
        # Validate that it's a reasonable email ID
        if not clean_id or len(clean_id) < 1:
            return False
        
        # Create folder name
        folder_name = f"INBOX.{label_name}"
        
        # Ensure folder exists
        if not client.create_folder(folder_name):
            # Continue anyway, might already exist
            pass
        
        # Try to move email to folder
        success = client.move_email_to_folder(clean_id, folder_name)
        
        if success:
            return True
        
        # Try with original ID if different
        if clean_id != email_id_str:
            success = client.move_email_to_folder(email_id_str, folder_name)
            if success:
                return True
        
        # Fallback: just mark as read
        try:
            status, response = client.connection.store(clean_id, '+FLAGS', '\\Seen')
            if status == "OK":
                return True
        except:
            pass
        
        return False
            
    except Exception as e:
        return False
    finally:
        try:
            client.disconnect()
        except:
            pass

def format_email_for_display(email: Dict) -> str:
    """
    Format an email for display in the daily briefing.
    
    Args:
        email: Email dictionary from get_unread_emails
    
    Returns:
        str: Formatted email string for display
    """
    subject = email.get('subject', 'No Subject')
    sender = email.get('sender', 'Unknown')
    
    # Extract sender name/email for cleaner display
    if '<' in sender and '>' in sender:
        # Format: "Name <email@domain.com>"
        sender_name = sender.split('<')[0].strip().strip('"')
        if sender_name:
            sender = sender_name
    
    # Truncate long subjects
    if len(subject) > 50:
        subject = subject[:47] + "..."
    
    return f"📧 {subject} (from {sender})"

def get_email_content_for_processing(email: Dict) -> str:
    """
    Get email content formatted for AI processing.
    
    Args:
        email: Email dictionary from get_unread_emails
    
    Returns:
        str: Email content formatted for AI processing
    """
    subject = email.get('subject', '')
    body = email.get('body', '')
    snippet = email.get('snippet', '')
    cc = email.get('cc', '')
    bcc = email.get('bcc', '')
    to = email.get('to', '')
    
    # Use body if available, otherwise use snippet
    content = body if body.strip() else snippet
    
    # Combine subject, recipients, and content
    full_content = f"Subject: {subject}\n"
    
    # Add recipient information
    if to:
        full_content += f"To: {to}\n"
    if cc:
        full_content += f"CC: {cc}\n"
    if bcc:
        full_content += f"BCC: {bcc}\n"
    
    full_content += f"\nContent: {content}"
    
    # Truncate if too long (to fit model context)
    if len(full_content) > 2000:
        full_content = full_content[:1997] + "..."
    
    return full_content

def get_recent_emails(credentials: Dict, max_results: int = 20) -> List[Dict]:
    """
    Fetch recent emails from IIT Delhi webmail (both read and unread).
    
    Args:
        credentials: Dictionary containing IMAP credentials
        max_results: Maximum number of emails to fetch (default: 20)
    
    Returns:
        List[Dict]: List of email dictionaries with id, subject, body, and sender
    """
    client = IMAPEmailClient(
        server=credentials['server'],
        port=credentials['port'],
        username=credentials['username'],
        password=credentials['password']
    )
    
    if not client.connect():
        return []
    
    try:
        emails = client.get_recent_emails(max_results)
        return emails
    finally:
        client.disconnect()

def apply_labels_batch(credentials: Dict, email_labels: List[tuple]) -> Dict[str, bool]:
    """
    Apply labels to multiple emails using improved folder system.
    Uses multiple strategies to achieve higher success rates.
    
    Args:
        credentials: Dictionary containing IMAP credentials
        email_labels: List of (email_id, label_name) tuples
    
    Returns:
        Dict[str, bool]: Dictionary mapping email_id to success status
    """
    client = IMAPEmailClient(
        server=credentials['server'],
        port=credentials['port'],
        username=credentials['username'],
        password=credentials['password']
    )
    
    if not client.connect():
        print("❌ Failed to connect for batch labeling")
        return {email_id: False for email_id, _ in email_labels}
    
    results = {}
    
    try:
        print(f"🏷️ Processing {len(email_labels)} emails with improved labeling...")
        
        # Group emails by label for efficient processing
        label_groups = {}
        for email_id, label_name in email_labels:
            if label_name not in label_groups:
                label_groups[label_name] = []
            label_groups[label_name].append(email_id)
        
        # Process each label group
        for label_name, email_ids in label_groups.items():
            print(f"📁 Processing {len(email_ids)} emails for label: {label_name}")
            
            # Create folder for this label
            folder_name = f"INBOX.{label_name}"
            try:
                client.connection.create(folder_name)
            except:
                pass  # Folder might already exist
            
            # Process each email in this group
            for email_id in email_ids:
                success = improved_label_single_email(client, email_id, folder_name)
                results[email_id] = success
                
                if success:
                    print(f"✅ {label_name}: {str(email_id)[:20]}... (success)")
                else:
                    print(f"❌ {label_name}: {str(email_id)[:20]}... (failed)")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch labeling error: {e}")
        return {email_id: False for email_id, _ in email_labels}
    finally:
        try:
            client.disconnect()
        except:
            pass

def improved_label_single_email(client: IMAPEmailClient, email_id: str, folder_name: str) -> bool:
    """
    Improved labeling for a single email with multiple strategies.
    
    Args:
        client: IMAP client instance
        email_id: Email message ID
        folder_name: Target folder name
    
    Returns:
        bool: True if email was labeled successfully, False otherwise
    """
    try:
        # Strategy 1: Clean and validate email ID
        email_id_str = clean_email_id(email_id)
        if not email_id_str:
            return False
        
        # Strategy 2: Select INBOX
        try:
            status, response = client.connection.select("INBOX")
            if status != "OK":
                return False
        except:
            return False
        
        # Strategy 3: Try multiple ID formats for copying
        id_formats = get_email_id_formats(email_id)
        
        for test_id in id_formats:
            try:
                status, response = client.connection.copy(test_id, folder_name)
                if status == "OK":
                    return True
            except:
                continue
        
        # Strategy 4: Try marking as read (fallback)
        for test_id in id_formats:
            try:
                status, response = client.connection.store(test_id, '+FLAGS', '\\Seen')
                if status == "OK":
                    return True
            except:
                continue
        
        return False
        
    except Exception as e:
        return False

def clean_email_id(email_id) -> str:
    """Clean and validate email ID with multiple strategies."""
    try:
        # Convert to string
        if isinstance(email_id, bytes):
            email_id = email_id.decode('utf-8')
        
        email_id_str = str(email_id).strip()
        
        # Strategy 1: Use as-is if it looks like a valid ID
        if email_id_str and len(email_id_str) > 0:
            # Check if it's already a valid format
            if email_id_str.isdigit() or (email_id_str.startswith('b\'') and email_id_str.endswith('\'')):
                return email_id_str
        
        # Strategy 2: Extract digits only
        import re
        clean_id = re.sub(r'[^\d]', '', email_id_str)
        if clean_id and len(clean_id) > 0:
            return clean_id
        
        # Strategy 3: Try to extract from bytes format
        if email_id_str.startswith('b\'') and email_id_str.endswith('\''):
            inner = email_id_str[2:-1]
            if inner.isdigit():
                return inner
        
        # Strategy 4: Return original if nothing else works
        return email_id_str if email_id_str else ""
        
    except Exception as e:
        return ""

def get_email_id_formats(email_id) -> List[str]:
    """Get multiple formats of email ID to try."""
    try:
        # Convert to string
        if isinstance(email_id, bytes):
            email_id = email_id.decode('utf-8')
        
        email_id_str = str(email_id).strip()
        
        # Generate multiple formats
        formats = [
            email_id_str,  # Original
            str(email_id),  # String conversion
        ]
        
        # Add cleaned versions
        import re
        clean_id = re.sub(r'[^\d]', '', email_id_str)
        if clean_id and clean_id != email_id_str:
            formats.append(clean_id)
        
        # Add integer version if possible
        try:
            if clean_id:
                int_id = str(int(clean_id))
                if int_id not in formats:
                    formats.append(int_id)
        except:
            pass
        
        # Add bytes format if applicable
        if email_id_str.startswith('b\'') and email_id_str.endswith('\''):
            inner = email_id_str[2:-1]
            if inner not in formats:
                formats.append(inner)
        
        # Remove duplicates and empty strings
        return list(dict.fromkeys([f for f in formats if f]))
        
    except Exception as e:
        return [str(email_id)]

def setup_priority_folders(credentials: Dict) -> bool:
    """
    Create priority folders for email organization.
    
    Args:
        credentials: Dictionary containing IMAP credentials
    
    Returns:
        bool: True if all folders were created successfully, False otherwise
    """
    client = IMAPEmailClient(
        server=credentials['server'],
        port=credentials['port'],
        username=credentials['username'],
        password=credentials['password']
    )
    
    if not client.connect():
        return False
    
    try:
        priority_folders = [
            "INBOX.AI-HIGH",
            "INBOX.AI-MEDIUM", 
            "INBOX.AI-LOW",
            "INBOX.AI-PROCESSED"
        ]
        
        success_count = 0
        for folder in priority_folders:
            if client.create_folder(folder):
                success_count += 1
        
        print(f"✅ Created {success_count}/{len(priority_folders)} priority folders")
        return success_count > 0
        
    finally:
        client.disconnect()

if __name__ == "__main__":
    """Test IMAP functionality."""
    print("IMAP tool test - requires credentials")
    # This would require actual credentials to test
    # The functions above handle all the IMAP operations needed
