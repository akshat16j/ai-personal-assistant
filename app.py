#!/usr/bin/env python3
"""
Flask web application for the AI Daily Briefing Agent V2.
Provides a web interface for the email triage and calendar management system.
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import threading
import time

# Import our tools
from imap_auth import get_imap_credentials, validate_credentials
from hybrid_auth import get_hybrid_credentials, validate_hybrid_credentials
from tools.imap_tool import (
    get_unread_emails, 
    get_recent_emails,
    apply_label_to_email,
    apply_labels_batch,
    format_email_for_display,
    get_email_content_for_processing
)
from tools.calendar_tool import (
    get_daily_calendar_events, 
    create_calendar_event,
    delete_event,
    list_all_events,
    format_event_for_display
)
from tools.finetuned_tools import load_agents

app = Flask(__name__)
app.secret_key = 'ai-briefing-agent-secret-key-2025'

# Global variables for caching
cached_imap_credentials = None
cached_google_credentials = None
cached_agents = None
last_update = None

# Statistics for the briefing
stats = {
    'emails_processed': 0,
    'high_priority_emails': 0,
    'medium_priority_emails': 0,
    'low_priority_emails': 0,
    'high_priority_events_created': 0,
    'existing_events': 0,
    'emails_labeled_successfully': 0,
    'emails_labeling_failed': 0
}

def get_credentials():
    """Get cached IMAP credentials or fetch new ones."""
    global cached_imap_credentials
    if cached_imap_credentials is None:
        try:
            cached_imap_credentials = get_imap_credentials()
        except Exception as e:
            print(f"Error getting IMAP credentials: {e}")
            return None
    return cached_imap_credentials

def get_google_credentials():
    """Get cached Google credentials or fetch new ones."""
    global cached_google_credentials
    if cached_google_credentials is None:
        try:
            from hybrid_auth import get_google_credentials as get_google_creds
            cached_google_credentials = get_google_creds()
        except Exception as e:
            print(f"Error getting Google credentials: {e}")
            return None
    return cached_google_credentials

def get_agents():
    """Get cached AI agents or load new ones."""
    global cached_agents
    if cached_agents is None:
        try:
            cached_agents = load_agents()
        except Exception as e:
            print(f"Error loading agents: {e}")
            return None, None
    return cached_agents

def is_timed_event(event_info: dict, email_content: str) -> bool:
    """
    Determine if an extracted event is a timed event that should be added to calendar.
    
    Args:
        event_info: Extracted event information
        email_content: Original email content
    
    Returns:
        bool: True if this is a timed event that should be calendared
    """
    if not event_info:
        return False
    
    summary = event_info.get('summary', '').upper()
    content_upper = email_content.upper()
    
    # Keywords that indicate timed events
    timed_keywords = [
        'QUIZ', 'ASSIGNMENT', 'DEADLINE', 'DUE', 'SUBMIT', 'SUBMISSION',
        'EXAM', 'FINAL', 'MIDTERM', 'TEST', 'PRESENTATION', 'PRESENT',
        'MEETING', 'LECTURE', 'SEMINAR', 'WORKSHOP', 'TUTORIAL',
        'PROJECT', 'REPORT', 'THESIS', 'VIVA', 'DEFENSE',
        'INTERVIEW', 'CALL', 'CONFERENCE', 'SYMPOSIUM'
    ]
    
    # Check if summary or content contains timed event keywords
    has_timed_keywords = any(keyword in summary or keyword in content_upper for keyword in timed_keywords)
    
    # Check if event has valid time information
    has_time_info = (
        event_info.get('start_time') and 
        event_info.get('end_time') and
        event_info['start_time'] != '2024-01-01T09:00:00Z'  # Not default time
    )
    
    # Check for time-related words in content
    time_words = ['TODAY', 'TOMORROW', 'YESTERDAY', 'MONDAY', 'TUESDAY', 'WEDNESDAY', 
                 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 'AM', 'PM', 'O\'CLOCK',
                 'NEXT', 'THIS', 'WEEK', 'MONTH', 'YEAR']
    has_time_words = any(word in content_upper for word in time_words)
    
    # Check for date patterns
    import re
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)',
        r'\d{1,2}\s*(?:AM|PM|am|pm)',
        r'\d{1,2}:\d{2}',  # Time without AM/PM
        r'\d{1,2}\s+(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)',
        r'\d{1,2}\s+(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)'
    ]
    has_date_patterns = any(re.search(pattern, content_upper) for pattern in date_patterns)
    
    # More lenient: It's a timed event if it has timed keywords OR has time/date information
    return has_timed_keywords or has_time_info or has_time_words or has_date_patterns

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    """API endpoint to check system status."""
    try:
        credentials = get_credentials()
        triage_agent, extraction_agent = get_agents()
        
        status = {
            'authenticated': credentials is not None,
            'models_loaded': triage_agent is not None and extraction_agent is not None,
            'last_update': last_update.isoformat() if last_update else None
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emails')
def api_emails():
    """API endpoint to fetch unread emails."""
    try:
        credentials = get_credentials()
        if not credentials:
            return jsonify({'error': 'Not authenticated'}), 401
        
        emails = get_unread_emails(credentials, max_results=20)
        
        # Process emails for display
        processed_emails = []
        for email in emails:
            processed_emails.append({
                'id': email['id'],
                'subject': email['subject'],
                'sender': email['sender'],
                'date': email['date'],
                'snippet': email['snippet'][:200] + '...' if len(email['snippet']) > 200 else email['snippet']
            })
        
        return jsonify(processed_emails)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emails/<email_id>/classify', methods=['POST'])
def api_classify_email(email_id):
    """API endpoint to classify an email."""
    try:
        credentials = get_credentials()
        triage_agent, _ = get_agents()
        
        if not credentials or not triage_agent:
            return jsonify({'error': 'System not ready'}), 500
        
        # Get email content
        emails = get_unread_emails(credentials, max_results=20)
        target_email = None
        for email in emails:
            if email['id'] == email_id:
                target_email = email
                break
        
        if not target_email:
            return jsonify({'error': 'Email not found'}), 404
        
        # Classify email
        from tools.imap_tool import get_email_content_for_processing
        email_content = get_email_content_for_processing(target_email)
        priority = triage_agent.classify(email_content)
        
        # Apply label and move to priority folder
        label_name = f"AI-{priority}"
        success = apply_label_to_email(credentials, email_id, label_name)
        
        return jsonify({
            'priority': priority,
            'label_applied': success,
            'email': {
                'id': target_email['id'],
                'subject': target_email['subject'],
                'sender': target_email['sender']
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calendar')
def api_calendar():
    """API endpoint to fetch calendar events."""
    try:
        google_creds = get_google_credentials()
        if not google_creds:
            return jsonify({'error': 'Google Calendar not authenticated'}), 401
        
        events = get_daily_calendar_events(google_creds)
        
        # Format events for display
        formatted_events = []
        for event in events:
            formatted_events.append({
                'id': event['id'],
                'summary': event['summary'],
                'start': event['start'],
                'end': event['end'],
                'location': event.get('location', ''),
                'description': event.get('description', ''),
                'all_day': event.get('all_day', False)
            })
        
        return jsonify(formatted_events)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calendar', methods=['POST'])
def api_create_event():
    """API endpoint to create a new calendar event."""
    try:
        google_creds = get_google_credentials()
        if not google_creds:
            return jsonify({'error': 'Google Calendar not authenticated'}), 401
        
        data = request.get_json()
        
        required_fields = ['summary', 'start_time', 'end_time']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        success = create_calendar_event(google_creds, data)
        
        if success:
            return jsonify({'message': 'Event created successfully'})
        else:
            return jsonify({'error': 'Failed to create event'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/calendar/<event_id>', methods=['DELETE'])
def api_delete_event(event_id):
    """API endpoint to delete a calendar event."""
    try:
        google_creds = get_google_credentials()
        if not google_creds:
            return jsonify({'error': 'Google Calendar not authenticated'}), 401
        
        success = delete_event(google_creds, event_id)
        
        if success:
            return jsonify({'message': 'Event deleted successfully'})
        else:
            return jsonify({'error': 'Event not found'}), 404
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process-all', methods=['POST'])
def api_process_all_emails():
    """API endpoint to process all unread emails."""
    try:
        credentials = get_credentials()
        triage_agent, _ = get_agents()
        
        if not credentials or not triage_agent:
            return jsonify({'error': 'System not ready'}), 500
        
        # Process emails in background
        def process_emails():
            global last_update, stats
            try:
                # Reset stats
                stats = {
                    'emails_processed': 0,
                    'high_priority_emails': 0,
                    'medium_priority_emails': 0,
                    'low_priority_emails': 0,
                    'high_priority_events_created': 0,
                    'existing_events': 0,
                    'emails_labeled_successfully': 0,
                    'emails_labeling_failed': 0
                }
                
                # Get recent emails (both read and unread)
                emails = get_recent_emails(credentials, max_results=50)
                print(f"📧 Found {len(emails)} recent emails to process")
                
                # Process each email
                processed_emails = []
                email_labels = []
                
                for email in emails:
                    try:
                        email_content = get_email_content_for_processing(email)
                        priority = triage_agent.classify(email_content)
                        
                        # Update statistics
                        stats['emails_processed'] += 1
                        if priority == 'HIGH':
                            stats['high_priority_emails'] += 1
                        elif priority == 'MEDIUM':
                            stats['medium_priority_emails'] += 1
                        else:
                            stats['low_priority_emails'] += 1
                        
                        processed_emails.append({
                            'email': email,
                            'priority': priority,
                            'content': email_content
                        })
                        
                        email_labels.append((email['id'], f"AI-{priority}"))
                        
                    except Exception as e:
                        print(f"Error processing email {email.get('id', 'unknown')}: {e}")
                        continue
                
                # Apply labels in batch for better efficiency
                if email_labels:
                    print(f"🏷️ Applying labels to {len(email_labels)} emails...")
                    labeling_results = apply_labels_batch(credentials, email_labels)
                    
                    # Count labeling results
                    for email_id, success in labeling_results.items():
                        if success:
                            stats['emails_labeled_successfully'] += 1
                        else:
                            stats['emails_labeling_failed'] += 1
                
                # Extract and create events from HIGH priority emails
                high_priority_emails = [p for p in processed_emails if p['priority'] == 'HIGH']
                if high_priority_emails:
                    print(f"🔍 Analyzing {len(high_priority_emails)} HIGH priority emails for timed events...")
                    
                    triage_agent, extraction_agent = get_agents()
                    if extraction_agent:
                        created_events = []
                        
                        for email_data in high_priority_emails:
                            try:
                                # Extract event using fine-tuned model
                                event_info = extraction_agent.extract_event(email_data['content'])
                                
                                if event_info:
                                    print(f"  🔍 Extracted event: {event_info.get('summary', 'Unknown')}")
                                    # Check if this is a timed event
                                    is_timed = is_timed_event(event_info, email_data['content'])
                                    
                                    if is_timed:
                                        print(f"  ⏰ Timed event detected: {event_info['summary']}")
                                        # Create calendar event
                                        success = create_calendar_event(cached_google_credentials, event_info)
                                        
                                        if success:
                                            stats['high_priority_events_created'] += 1
                                            created_events.append({
                                                'event_info': event_info,
                                                'source_email': email_data['email'],
                                                'priority': email_data['priority']
                                            })
                                            print(f"  ✅ Created: {event_info['summary']} (Priority: {email_data['priority']})")
                                        else:
                                            print(f"  ⚠️ Failed to create: {event_info['summary']}")
                                    else:
                                        print(f"  ℹ️ No timed event found in: {email_data['email']['subject'][:40]}...")
                                else:
                                    print(f"  ℹ️ No event found in: {email_data['email']['subject'][:40]}...")
                                
                            except Exception as e:
                                print(f"  ❌ Error processing email for events: {e}")
                                continue
                        
                        if created_events:
                            print(f"✅ Created {len(created_events)} calendar events for HIGH priority timed events")
                        else:
                            print("ℹ️ No timed events were found in HIGH priority emails")
                
                last_update = datetime.now()
                print(f"📊 Processing complete: {stats['emails_processed']} emails, {stats['high_priority_events_created']} events created")
                
            except Exception as e:
                print(f"Error processing emails: {e}")
        
        # Start background processing
        thread = threading.Thread(target=process_emails)
        thread.daemon = True
        thread.start()
        
        return jsonify({'message': 'Email processing started in background'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint to get processing statistics."""
    try:
        credentials = get_credentials()
        if not credentials:
            return jsonify({'error': 'Not authenticated'}), 401
        
        emails = get_recent_emails(credentials, max_results=50)
        google_creds = get_google_credentials()
        events = get_daily_calendar_events(google_creds) if google_creds else []
        
        # Include the global stats
        global stats
        response_stats = {
            'unread_emails': len(emails),
            'today_events': len(events),
            'last_update': last_update.isoformat() if last_update else None,
            'system_status': 'online',
            'processing_stats': stats
        }
        
        return jsonify(response_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/run-agent', methods=['POST'])
def api_run_agent():
    """API endpoint to run the complete AI briefing agent."""
    try:
        credentials = get_credentials()
        triage_agent, extraction_agent = get_agents()
        
        if not credentials or not triage_agent:
            return jsonify({'error': 'System not ready'}), 500
        
        print("🤖 Starting full AI briefing agent...")
        
        # Reset global stats
        global stats
        stats = {
            'emails_processed': 0,
            'high_priority_emails': 0,
            'medium_priority_emails': 0,
            'low_priority_emails': 0,
            'high_priority_events_created': 0,
            'existing_events': 0,
            'emails_labeled_successfully': 0,
            'emails_labeling_failed': 0
        }
        
        # Step 0: Setup priority folders for email organization
        from tools.imap_tool import setup_priority_folders
        print("📁 Setting up priority folders...")
        setup_priority_folders(credentials)
        
        # Step 1: Process emails (both read and unread)
        emails = get_recent_emails(credentials, max_results=50)
        print(f"📧 Found {len(emails)} recent emails to process")
        
        processed_emails = []
        email_labels = []
        actions_log = []
        
        for email in emails:
            try:
                email_content = get_email_content_for_processing(email)
                priority = triage_agent.classify(email_content)
                
                # Update statistics
                stats['emails_processed'] += 1
                if priority == 'HIGH':
                    stats['high_priority_emails'] += 1
                elif priority == 'MEDIUM':
                    stats['medium_priority_emails'] += 1
                else:
                    stats['low_priority_emails'] += 1
                
                label_name = f"AI-{priority}"
                email_labels.append((email['id'], label_name))
                
                processed_emails.append({
                    'id': email['id'],
                    'subject': email['subject'],
                    'sender': email['sender'],
                    'priority': priority
                })
                
            except Exception as e:
                print(f"⚠ Warning: Failed to process email {email.get('id', 'unknown')}: {e}")
                actions_log.append({
                    'action': f'Failed to process email: {email.get("subject", "Unknown")[:50]}...',
                    'details': str(e),
                    'type': 'error',
                    'status': 'error'
                })
        
        # Apply labels in batch for better efficiency
        if email_labels:
            print(f"🏷️ Applying labels to {len(email_labels)} emails...")
            labeling_results = apply_labels_batch(credentials, email_labels)
            
            # Count labeling results
            for email_id, success in labeling_results.items():
                if success:
                    stats['emails_labeled_successfully'] += 1
                else:
                    stats['emails_labeling_failed'] += 1
            
            # Log the results
            for email in processed_emails:
                email_id = email['id']
                priority = email['priority']
                success = labeling_results.get(email_id, False)
                
                actions_log.append({
                    'action': f'Processed email: {email["subject"][:50]}...',
                    'details': f'Priority: {priority}, Sender: {email["sender"]}, Labeled: {"✅" if success else "❌"}',
                    'type': 'email_label',
                    'status': 'success' if success else 'error'
                })
        
        # Step 2: Get calendar events
        google_creds = get_google_credentials()
        events = get_daily_calendar_events(google_creds) if google_creds else []
        stats['existing_events'] = len(events)
        
        # Step 3: Extract and create events from HIGH priority emails
        high_priority_emails = [p for p in processed_emails if p['priority'] == 'HIGH']
        if high_priority_emails and extraction_agent:
            print(f"🔍 Analyzing {len(high_priority_emails)} HIGH priority emails for timed events...")
            
            created_events = []
            for email in high_priority_emails:
                try:
                    # Get the full email data
                    email_data = next((e for e in emails if e['id'] == email['id']), None)
                    if not email_data:
                        continue
                    
                    email_content = get_email_content_for_processing(email_data)
                    # Extract event using fine-tuned model
                    event_info = extraction_agent.extract_event(email_content)
                    
                    if event_info:
                        print(f"  🔍 Extracted event: {event_info.get('summary', 'Unknown')}")
                        # Check if this is a timed event
                        is_timed = is_timed_event(event_info, email_content)
                        
                        if is_timed:
                            print(f"  ⏰ Timed event detected: {event_info['summary']}")
                            # Create calendar event
                            success = create_calendar_event(google_creds, event_info)
                            
                            if success:
                                stats['high_priority_events_created'] += 1
                                created_events.append({
                                    'event_info': event_info,
                                    'source_email': email,
                                    'priority': email['priority']
                                })
                                print(f"  ✅ Created: {event_info['summary']} (Priority: {email['priority']})")
                                
                                actions_log.append({
                                    'action': f'Created event: {event_info["summary"]}',
                                    'details': f'From email: {email["subject"][:30]}...',
                                    'type': 'event_create',
                                    'status': 'success'
                                })
                            else:
                                print(f"  ⚠️ Failed to create: {event_info['summary']}")
                                actions_log.append({
                                    'action': f'Failed to create event: {event_info["summary"]}',
                                    'details': f'From email: {email["subject"][:30]}...',
                                    'type': 'event_create',
                                    'status': 'error'
                                })
                        else:
                            print(f"  ℹ️ No timed event found in: {email['subject'][:40]}...")
                    else:
                        print(f"  ℹ️ No event found in: {email['subject'][:40]}...")
                    
                except Exception as e:
                    print(f"  ❌ Error processing email for events: {e}")
                    actions_log.append({
                        'action': f'Error processing email for events: {email["subject"][:30]}...',
                        'details': str(e),
                        'type': 'event_create',
                        'status': 'error'
                    })
                    continue
            
            if created_events:
                print(f"✅ Created {len(created_events)} calendar events for HIGH priority timed events")
            else:
                print("ℹ️ No timed events were found in HIGH priority emails")
        
        last_update = datetime.now()
        print(f"📊 Processing complete: {stats['emails_processed']} emails, {stats['high_priority_events_created']} events created")
        
        # Return structured data for the frontend
        return jsonify({
            'message': 'AI briefing agent completed successfully',
            'status': 'completed',
            'statistics': stats,
            'processed_emails': processed_emails,
            'todays_events': events,
            'actions_log': actions_log
        })
        
    except Exception as e:
        print(f"❌ Error in full agent run: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting AI Daily Briefing Agent V2 Web Interface")
    print("📧 IIT Delhi Webmail Integration")
    print("🌐 Web interface will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)