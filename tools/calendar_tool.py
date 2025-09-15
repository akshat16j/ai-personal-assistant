#!/usr/bin/env python3
"""
Google Calendar API integration for the AI Daily Briefing Agent.
Handles fetching today's events and creating new calendar events.
"""

import datetime
from typing import List, Dict, Optional
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def get_daily_calendar_events(credentials) -> List[Dict]:
    """
    Fetch today's events from the user's primary Google Calendar.
    
    Args:
        credentials: Google OAuth2 credentials object
    
    Returns:
        List[Dict]: List of today's calendar events with details
    """
    try:
        # Build the Calendar service
        service = build('calendar', 'v3', credentials=credentials)
        
        # Get today's date range
        now = datetime.datetime.now()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = now.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        # Convert to RFC3339 format
        time_min = start_of_day.isoformat() + 'Z'
        time_max = end_of_day.isoformat() + 'Z'
        
        print(f"📅 Fetching events for {now.strftime('%Y-%m-%d')}")
        
        # Call the Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy='startTime',
            maxResults=50
        ).execute()
        
        events = events_result.get('items', [])
        
        # Process events into a standardized format
        processed_events = []
        for event in events:
            event_data = {
                'id': event.get('id', ''),
                'summary': event.get('summary', 'No Title'),
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'status': event.get('status', 'confirmed'),
                'created': event.get('created', ''),
                'updated': event.get('updated', ''),
                'start': None,
                'end': None,
                'all_day': False
            }
            
            # Handle start time (could be dateTime or date)
            start = event.get('start', {})
            if 'dateTime' in start:
                event_data['start'] = start['dateTime']
                event_data['all_day'] = False
            elif 'date' in start:
                event_data['start'] = start['date']
                event_data['all_day'] = True
            
            # Handle end time (could be dateTime or date)
            end = event.get('end', {})
            if 'dateTime' in end:
                event_data['end'] = end['dateTime']
            elif 'date' in end:
                event_data['end'] = end['date']
            
            processed_events.append(event_data)
        
        print(f"✓ Found {len(processed_events)} events for today")
        return processed_events
        
    except HttpError as error:
        print(f"❌ Calendar API error: {error}")
        return []
    except Exception as error:
        print(f"❌ Unexpected error fetching calendar events: {error}")
        return []

def create_calendar_event(credentials=None, event_data: Dict = None, **kwargs) -> bool:
    """
    Create a new event in the user's primary Google Calendar.
    
    Args:
        credentials: Google OAuth2 credentials object
        event_data: Dictionary containing event information with keys:
            - summary: Event title (required)
            - start_time: Start time in ISO format or date string (required)
            - end_time: End time in ISO format or date string (required)
            - description: Event description (optional)
            - location: Event location (optional)
    
    Returns:
        bool: True if event was created successfully, False otherwise
    """
    try:
        # Handle both calling patterns: (credentials, event_data) and keyword arguments
        if event_data is None:
            event_data = kwargs
        
        # Build the Calendar service
        service = build('calendar', 'v3', credentials=credentials)
        
        # Extract required fields
        summary = event_data.get('summary', 'New Event')
        start_time = event_data.get('start_time')
        end_time = event_data.get('end_time')
        
        if not start_time or not end_time:
            print("❌ Error: start_time and end_time are required for calendar events")
            return False
        
        # Validate and clean the summary (required field)
        if not summary or summary.strip() == '':
            summary = 'Untitled Event'
        else:
            summary = str(summary).strip()
        
        # Validate and format times
        try:
            start_time = str(start_time).strip()
            end_time = str(end_time).strip()
            
            # Validate that times are not empty
            if not start_time or not end_time:
                print(f"❌ Error: Empty time values - start: '{start_time}', end: '{end_time}'")
                return False
                
            # Validate that end time is after start time
            try:
                # Parse times for comparison
                start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                
                if end_dt <= start_dt:
                    print(f"❌ Error: End time must be after start time - start: {start_time}, end: {end_time}")
                    # Fix by adding 1 hour to end time
                    end_dt = start_dt + datetime.timedelta(hours=1)
                    end_time = end_dt.strftime('%Y-%m-%dT%H:%M:%SZ')
                    print(f"🔧 Fixed end time to: {end_time}")
                    
            except Exception as parse_error:
                print(f"⚠ Warning: Could not parse times for validation: {parse_error}")
                # Continue with original times
                
        except Exception as e:
            print(f"❌ Error processing time values: {e}")
            return False
        
        # Determine if this is an all-day event or timed event
        is_all_day = _is_all_day_event(start_time, end_time)
        
        # Prepare event body with proper validation
        event_body = {
            'summary': summary,
        }
        
        # Add optional fields only if they have valid content
        description = event_data.get('description', '')
        if description and str(description).strip():
            event_body['description'] = str(description).strip()
        
        location = event_data.get('location', '')
        if location and str(location).strip():
            event_body['location'] = str(location).strip()
        
        if is_all_day:
            # All-day event - use 'date' field
            event_body['start'] = {'date': _parse_date_string(start_time)}
            event_body['end'] = {'date': _parse_date_string(end_time)}
            print(f"📅 Creating all-day event: {summary}")
        else:
            # Timed event - use 'dateTime' field
            event_body['start'] = {
                'dateTime': _ensure_timezone(start_time),
                'timeZone': 'UTC'
            }
            event_body['end'] = {
                'dateTime': _ensure_timezone(end_time),
                'timeZone': 'UTC'
            }
            print(f"⏰ Creating timed event: {summary}")
        
        # Create the event
        created_event = service.events().insert(
            calendarId='primary',
            body=event_body
        ).execute()
        
        print(f"✅ Event created successfully: {created_event.get('htmlLink', 'N/A')}")
        return True
        
    except HttpError as error:
        print(f"❌ Calendar API error creating event: {error}")
        if hasattr(error, 'content'):
            print(f"❌ Error details: {error.content}")
        print(f"❌ Event data that failed: {event_body}")
        return False
    except Exception as error:
        print(f"❌ Unexpected error creating calendar event: {error}")
        print(f"❌ Event data that failed: {event_body}")
        return False

def _is_all_day_event(start_time: str, end_time: str) -> bool:
    """
    Determine if an event is all-day based on the time format.
    
    Args:
        start_time: Start time string
        end_time: End time string
    
    Returns:
        bool: True if this appears to be an all-day event
    """
    # Check if times are in date format (YYYY-MM-DD) without time component
    try:
        # If we can parse as date only (10 chars), it's likely all-day
        if len(start_time) == 10 and len(end_time) == 10:
            datetime.datetime.strptime(start_time, '%Y-%m-%d')
            datetime.datetime.strptime(end_time, '%Y-%m-%d')
            return True
    except ValueError:
        pass
    
    # Check if times are exactly at midnight (typical for deadlines)
    if 'T00:00:00' in start_time and 'T00:00:00' in end_time:
        return True
    
    return False

def _parse_date_string(date_str: str) -> str:
    """
    Parse date string and return in YYYY-MM-DD format for all-day events.
    
    Args:
        date_str: Date string in various formats
    
    Returns:
        str: Date in YYYY-MM-DD format
    """
    try:
        # Clean the input
        date_str = str(date_str).strip()
        
        # If it's already in the right format
        if len(date_str) == 10 and date_str.count('-') == 2:
            return date_str
        
        # If it contains time information, extract just the date part
        if 'T' in date_str:
            return date_str.split('T')[0]
        
        # Try to parse and reformat
        parsed_date = datetime.datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return parsed_date.strftime('%Y-%m-%d')
        
    except Exception as e:
        print(f"⚠ Warning: Could not parse date '{date_str}': {e}")
        # Return today's date as fallback
        return datetime.datetime.now().strftime('%Y-%m-%d')

def _ensure_timezone(datetime_str: str) -> str:
    """
    Ensure datetime string has timezone information and is properly formatted.
    
    Args:
        datetime_str: Datetime string
    
    Returns:
        str: Datetime string with timezone
    """
    try:
        # Clean the input
        datetime_str = str(datetime_str).strip()
        
        # Check if the datetime string is malformed (contains text that shouldn't be there)
        if len(datetime_str) > 50 or any(char.isalpha() and char not in ['T', 'Z'] for char in datetime_str):
            print(f"⚠ Warning: Malformed datetime string detected: '{datetime_str[:50]}...'")
            # Return a safe default
            return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # If already has timezone info, return as-is
        if datetime_str.endswith('Z') or '+' in datetime_str[-6:] or '-' in datetime_str[-6:]:
            return datetime_str
        
        # Add Z for UTC timezone
        if 'T' in datetime_str:
            return datetime_str + 'Z'
        
        # If no time component, add default time
        return datetime_str + 'T00:00:00Z'
        
    except Exception as e:
        print(f"⚠ Warning: Could not process datetime '{datetime_str}': {e}")
        # Return a safe default
        return datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%SZ')

def format_event_for_display(event: Dict) -> str:
    """
    Format a calendar event for display in the daily briefing.
    
    Args:
        event: Event dictionary from get_daily_calendar_events
    
    Returns:
        str: Formatted event string for display
    """
    summary = event.get('summary', 'Untitled Event')
    
    if event.get('all_day', False):
        return f"📅 {summary} (All Day)"
    else:
        start = event.get('start', '')
        if start:
            try:
                # Parse and format time
                if 'T' in start:
                    dt = datetime.datetime.fromisoformat(start.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M')
                    return f"⏰ {time_str} - {summary}"
                else:
                    return f"📅 {summary}"
            except:
                return f"📅 {summary}"
        else:
            return f"📅 {summary}"

def delete_event(credentials, event_id: str) -> bool:
    """
    Delete an event from the user's primary Google Calendar.
    
    Args:
        credentials: Google OAuth2 credentials object
        event_id: ID of the event to delete
    
    Returns:
        bool: True if event was deleted successfully, False otherwise
    """
    try:
        # Build the Calendar service
        service = build('calendar', 'v3', credentials=credentials)
        
        # Delete the event
        service.events().delete(
            calendarId='primary',
            eventId=event_id
        ).execute()
        
        print(f"✅ Event {event_id} deleted successfully")
        return True
        
    except HttpError as error:
        if error.resp.status == 404:
            print(f"❌ Event {event_id} not found")
        else:
            print(f"❌ Calendar API error deleting event: {error}")
        return False
    except Exception as error:
        print(f"❌ Unexpected error deleting calendar event: {error}")
        return False

def list_all_events(credentials, max_results: int = 100) -> List[Dict]:
    """
    List all events from the user's primary Google Calendar.
    
    Args:
        credentials: Google OAuth2 credentials object
        max_results: Maximum number of events to return (default: 100)
    
    Returns:
        List[Dict]: List of calendar events with details
    """
    try:
        # Build the Calendar service
        service = build('calendar', 'v3', credentials=credentials)
        
        # Get current time for timeMin
        now = datetime.datetime.now()
        time_min = now.isoformat() + 'Z'
        
        print(f"📅 Fetching all events from {now.strftime('%Y-%m-%d')}")
        
        # Call the Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=time_min,
            singleEvents=True,
            orderBy='startTime',
            maxResults=max_results
        ).execute()
        
        events = events_result.get('items', [])
        
        # Process events into a standardized format (same as get_daily_calendar_events)
        processed_events = []
        for event in events:
            event_data = {
                'id': event.get('id', ''),
                'summary': event.get('summary', 'No Title'),
                'description': event.get('description', ''),
                'location': event.get('location', ''),
                'status': event.get('status', 'confirmed'),
                'created': event.get('created', ''),
                'updated': event.get('updated', ''),
                'start': None,
                'end': None,
                'all_day': False
            }
            
            # Handle start time (could be dateTime or date)
            start = event.get('start', {})
            if 'dateTime' in start:
                event_data['start'] = start['dateTime']
                event_data['all_day'] = False
            elif 'date' in start:
                event_data['start'] = start['date']
                event_data['all_day'] = True
            
            # Handle end time (could be dateTime or date)
            end = event.get('end', {})
            if 'dateTime' in end:
                event_data['end'] = end['dateTime']
            elif 'date' in end:
                event_data['end'] = end['date']
            
            processed_events.append(event_data)
        
        print(f"✓ Found {len(processed_events)} events")
        return processed_events
        
    except HttpError as error:
        print(f"❌ Calendar API error: {error}")
        return []
    except Exception as error:
        print(f"❌ Unexpected error fetching calendar events: {error}")
        return []

if __name__ == "__main__":
    """Test calendar functionality."""
    print("Calendar tool test - requires authentication")
    # This would require actual credentials to test
    # The functions above handle all the calendar operations needed