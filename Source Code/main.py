#!/usr/bin/env python3
"""
AI Daily Briefing Agent V2 - Main Orchestrator

This is the main Planner/Executor script that orchestrates the complete workflow:
1. Email Triage & Priority Reassignment: Scan recent emails and reassign their priority
2. Event Extraction & Calendar Update: Extract events from high-priority emails
3. Calendar Check: Fetch today's existing events
4. Daily Briefing: Generate comprehensive CLI summary

The system uses fine-tuned FLAN-T5-Small models with LoRA for reliable email 
classification and event extraction to minimize hallucination risks.

Note: This script processes both read and unread emails to ensure priority 
reassignment happens every time it runs.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Rich library for beautiful CLI output
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich import box

# Import our tools and authentication
from hybrid_auth import get_hybrid_credentials, validate_hybrid_credentials
from tools.imap_tool import (
    get_unread_emails, 
    get_recent_emails,
    apply_label_to_email,
    format_email_for_display,
    get_email_content_for_processing
)
from tools.calendar_tool import (
    get_daily_calendar_events,
    create_calendar_event,
    format_event_for_display
)
from tools.finetuned_tools import load_agents

# Initialize rich console
console = Console()

class DailyBriefingAgent:
    """Main orchestrator for the AI Daily Briefing Agent V2."""
    
    def __init__(self):
        """Initialize the agent with all necessary components."""
        self.imap_credentials = None
        self.google_credentials = None
        self.triage_agent = None
        self.extraction_agent = None
        
        # Statistics for the briefing
        self.stats = {
            'emails_processed': 0,
            'high_priority_emails': 0,
            'medium_priority_emails': 0,
            'low_priority_emails': 0,
            'events_created': 0,
            'existing_events': 0,
            'emails_labeled_successfully': 0,
            'emails_labeling_failed': 0
        }
    
    def initialize(self):
        """Initialize all components: authentication and AI models."""
        console.print("\n[bold blue]🚀 AI Daily Briefing Agent V2[/bold blue]")
        console.print("=" * 50)
        
        # Step 1: Hybrid authentication (IMAP + Google Calendar)
        console.print("\n[yellow]Step 1: Hybrid Authentication Setup[/yellow]")
        try:
            self.imap_credentials, self.google_credentials = get_hybrid_credentials()
            if not validate_hybrid_credentials(self.imap_credentials, self.google_credentials):
                raise Exception("Invalid or insufficient credentials")
            console.print("✅ [green]Authentication successful[/green]")
        except Exception as e:
            console.print(f"❌ [red]Authentication failed: {e}[/red]")
            sys.exit(1)
        
        # Step 2: Load fine-tuned models
        console.print("\n[yellow]Step 2: Loading AI Models[/yellow]")
        try:
            self.triage_agent, self.extraction_agent = load_agents()
            console.print("✅ [green]AI models loaded successfully[/green]")
        except Exception as e:
            console.print(f"❌ [red]Failed to load models: {e}[/red]")
            console.print("\n[dim]Tip: Run 'python models/train.py' to train the models first[/dim]")
            sys.exit(1)
    
    def process_emails(self) -> List[Dict]:
        """Process emails: triage and reassign priorities (both read and unread)."""
        console.print("\n[yellow]Step 3: Email Triage & Priority Reassignment[/yellow]")
        
        # Always fetch recent emails (last 30 days) to reassign priorities
        console.print("ℹ️ [dim]Fetching recent emails for priority reassignment...[/dim]")
        emails = get_recent_emails(self.imap_credentials, max_results=50)
            
        if not emails:
            console.print("ℹ️ [dim]No emails found to process[/dim]")
            return []
        
        self.stats['emails_processed'] = len(emails)
        console.print(f"📧 Processing {len(emails)} recent emails for priority reassignment...")
        
        processed_emails = []
        
        for email in emails:
            try:
                # Get email content for AI processing
                email_content = get_email_content_for_processing(email)
                
                # Classify priority using fine-tuned model
                priority = self.triage_agent.classify(email_content)
                
                # Update statistics
                if priority == "HIGH":
                    self.stats['high_priority_emails'] += 1
                elif priority == "MEDIUM":
                    self.stats['medium_priority_emails'] += 1
                else:
                    self.stats['low_priority_emails'] += 1
                
                # Apply label to email
                label_name = f"AI-{priority}"
                label_applied = apply_label_to_email(
                    self.imap_credentials, 
                    email['id'], 
                    label_name
                )
                
                # Track labeling success
                if label_applied:
                    self.stats['emails_labeled_successfully'] += 1
                else:
                    self.stats['emails_labeling_failed'] += 1
                
                # Store processed email data
                email_data = {
                    'email': email,
                    'priority': priority,
                    'label_applied': label_applied,
                    'content': email_content
                }
                processed_emails.append(email_data)
                
                # Display progress with more detail
                status = "✅" if label_applied else "⚠️"
                status_text = "Labeled" if label_applied else "Label Failed"
                console.print(f"  {status} {priority}: {email['subject'][:40]}... ({status_text})")
                
            except Exception as e:
                console.print(f"  ❌ Error processing email: {e}")
                continue
        
        # Show labeling summary
        total_emails = len(processed_emails)
        successful_labels = self.stats['emails_labeled_successfully']
        failed_labels = self.stats['emails_labeling_failed']
        
        console.print(f"✅ [green]Email processing complete[/green]")
        console.print(f"📊 [dim]Labeling Summary: {successful_labels}/{total_emails} emails labeled successfully[/dim]")
        
        if failed_labels > 0:
            console.print(f"⚠️ [yellow]{failed_labels} emails failed to label - check IMAP connection or email permissions[/yellow]")
        
        return processed_emails
    
    def extract_and_create_events(self, processed_emails: List[Dict]) -> List[Dict]:
        """Extract events from HIGH priority emails and create calendar events for timed events."""
        console.print("\n[yellow]Step 4: Event Extraction & Calendar Creation[/yellow]")
        
        if not self.google_credentials:
            console.print("ℹ️ [dim]Google Calendar not available - skipping event creation[/dim]")
            return []
        
        # Filter HIGH priority emails only
        high_priority_emails = [
            email_data for email_data in processed_emails 
            if email_data['priority'] == 'HIGH'
        ]
        
        if not high_priority_emails:
            console.print("ℹ️ [dim]No HIGH priority emails for event extraction[/dim]")
            return []
        
        console.print(f"🔍 Analyzing {len(high_priority_emails)} HIGH priority emails for timed events...")
        
        created_events = []
        
        for email_data in high_priority_emails:
            try:
                # Extract event using fine-tuned model
                event_info = self.extraction_agent.extract_event(email_data['content'])
                
                if event_info:
                    console.print(f"  🔍 Extracted event: {event_info.get('summary', 'Unknown')}")
                    # Check if this is a timed event (quiz, assignment, deadline, etc.)
                    is_timed_event = self._is_timed_event(event_info, email_data['content'])
                    
                    if is_timed_event:
                        console.print(f"  ⏰ Timed event detected: {event_info['summary']}")
                        # Create calendar event
                        success = create_calendar_event(self.google_credentials, event_info)
                        
                        if success:
                            self.stats['events_created'] += 1
                            created_events.append({
                                'event_info': event_info,
                                'source_email': email_data['email'],
                                'priority': email_data['priority']
                            })
                            console.print(f"  ✅ Created: {event_info['summary']} (Priority: {email_data['priority']})")
                        else:
                            console.print(f"  ⚠️ Failed to create: {event_info['summary']}")
                    else:
                        console.print(f"  ℹ️ No timed event found in: {email_data['email']['subject'][:40]}...")
                else:
                    console.print(f"  ℹ️ No event found in: {email_data['email']['subject'][:40]}...")
                
            except Exception as e:
                console.print(f"  ❌ Error processing email for events: {e}")
                continue
        
        if created_events:
            console.print(f"✅ [green]Created {len(created_events)} calendar events for HIGH priority timed events[/green]")
        else:
            console.print("ℹ️ [dim]No timed events were found in HIGH priority emails[/dim]")
        
        return created_events
    
    def _is_timed_event(self, event_info: Dict, email_content: str) -> bool:
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
    
    def get_daily_calendar(self) -> List[Dict]:
        """Fetch today's calendar events."""
        console.print("\n[yellow]Step 5: Daily Calendar Check[/yellow]")
        
        if not self.google_credentials:
            console.print("ℹ️ [dim]Google Calendar not available - calendar features disabled[/dim]")
            return []
        
        events = get_daily_calendar_events(self.google_credentials)
        self.stats['existing_events'] = len(events)
        
        if events:
            console.print(f"📅 [green]Found {len(events)} events for today[/green]")
        else:
            console.print("ℹ️ [dim]No events scheduled for today[/dim]")
        
        return events
    
    def generate_daily_briefing(self, processed_emails: List[Dict], 
                              created_events: List[Dict], 
                              calendar_events: List[Dict]):
        """Generate and display the comprehensive daily briefing."""
        console.print("\n[yellow]Step 6: Generating Daily Briefing[/yellow]")
        
        # Create main layout
        layout = Layout()
        
        # Header with date and summary
        today = datetime.now().strftime("%A, %B %d, %Y")
        header = Panel(
            f"[bold white]📋 Daily Briefing for {today}[/bold white]",
            style="blue",
            padding=(1, 2)
        )
        
        # Statistics panel
        stats_text = Text()
        stats_text.append("📊 Summary Statistics\n\n", style="bold cyan")
        stats_text.append(f"📧 Emails Processed: {self.stats['emails_processed']}\n")
        stats_text.append(f"✅ Successfully Labeled: {self.stats['emails_labeled_successfully']}\n")
        stats_text.append(f"⚠️ Labeling Failed: {self.stats['emails_labeling_failed']}\n")
        stats_text.append(f"🔴 High Priority: {self.stats['high_priority_emails']}\n")
        stats_text.append(f"🟡 Medium Priority: {self.stats['medium_priority_emails']}\n")
        stats_text.append(f"🟢 Low Priority: {self.stats['low_priority_emails']}\n")
        stats_text.append(f"⏰ HIGH Priority Events Created: {self.stats['events_created']}\n")
        stats_text.append(f"📆 Today's Calendar Events: {self.stats['existing_events']}")
        
        stats_panel = Panel(
            stats_text,
            title="Statistics",
            style="green",
            padding=(1, 2)
        )
        
        # High priority emails panel
        if processed_emails:
            high_priority = [e for e in processed_emails if e['priority'] == 'HIGH']
            if high_priority:
                email_text = Text()
                email_text.append("🚨 High Priority Emails\n\n", style="bold red")
                for email_data in high_priority[:5]:  # Show top 5
                    email = email_data['email']
                    email_text.append(f"• {email['subject']}\n", style="red")
                    email_text.append(f"  From: {email['sender']}\n", style="dim")
                    email_text.append(f"  Label: AI-{email_data['priority']} ✅\n\n", style="dim green")
            else:
                email_text = Text("No high priority emails today! 🎉", style="green")
        else:
            email_text = Text("No emails to process", style="dim")
        
        email_panel = Panel(
            email_text,
            title="Priority Emails",
            style="red",
            padding=(1, 2)
        )
        
        # Today's calendar panel
        if calendar_events:
            cal_text = Text()
            cal_text.append("📅 Today's Schedule\n\n", style="bold blue")
            
            # Sort events by start time
            sorted_events = sorted(calendar_events, key=lambda x: x.get('start', ''))
            
            for event in sorted_events[:10]:  # Show up to 10 events
                formatted_event = format_event_for_display(event)
                cal_text.append(f"{formatted_event}\n")
                
                if event.get('location'):
                    cal_text.append(f"  📍 {event['location']}\n", style="dim")
                cal_text.append("\n")
        else:
            cal_text = Text("No events scheduled for today", style="dim")
        
        calendar_panel = Panel(
            cal_text,
            title="Today's Calendar",
            style="blue",
            padding=(1, 2)
        )
        
        # Newly created events panel
        if created_events:
            new_events_text = Text()
            new_events_text.append("✨ Newly Created HIGH Priority Events\n\n", style="bold green")
            for event_data in created_events:
                event_info = event_data['event_info']
                source_email = event_data['source_email']
                priority = event_data.get('priority', 'UNKNOWN')
                
                # Color code by priority
                priority_color = "red" if priority == "HIGH" else "yellow" if priority == "MEDIUM" else "green"
                
                new_events_text.append(f"• {event_info['summary']}\n", style="green")
                new_events_text.append(f"  Priority: {priority}\n", style=priority_color)
                new_events_text.append(f"  From email: {source_email['subject'][:40]}...\n", style="dim")
                
                # Format time display
                try:
                    from datetime import datetime
                    start_time = event_info['start_time']
                    if 'T' in start_time:
                        dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                        time_str = dt.strftime('%Y-%m-%d %H:%M')
                        new_events_text.append(f"  Time: {time_str}\n", style="dim")
                    else:
                        new_events_text.append(f"  Time: {start_time}\n", style="dim")
                except:
                    new_events_text.append(f"  Time: {event_info['start_time']}\n", style="dim")
                
                if event_info.get('location'):
                    new_events_text.append(f"  Location: {event_info['location']}\n", style="dim")
                
                new_events_text.append("\n")
        else:
            new_events_text = Text("No HIGH priority timed events created today", style="dim")
        
        new_events_panel = Panel(
            new_events_text,
            title="New HIGH Priority Events",
            style="green",
            padding=(1, 2)
        )
        
        # Display the briefing
        console.print("\n")
        console.print(header)
        console.print("\n")
        
        # Create columns for main content
        console.print(Columns([stats_panel, email_panel]))
        console.print("\n")
        console.print(Columns([calendar_panel, new_events_panel]))
        
        # Footer with tips
        footer_text = Text()
        footer_text.append("💡 Tips: ", style="bold yellow")
        footer_text.append("Check your IIT Delhi webmail for priority reassignments • ", style="dim")
        if self.google_credentials:
            footer_text.append("HIGH priority timed events (quizzes, deadlines) are automatically added to Google Calendar • ", style="dim")
        else:
            footer_text.append("Google Calendar integration not available • ", style="dim")
        footer_text.append("Run again anytime to process new emails and create events", style="dim")
        
        footer = Panel(
            footer_text,
            style="yellow",
            padding=(0, 2)
        )
        
        console.print("\n")
        console.print(footer)
    
    def run(self):
        """Execute the complete daily briefing workflow."""
        start_time = datetime.now()
        
        try:
            # Initialize all components
            self.initialize()
            
            # Execute the workflow
            processed_emails = self.process_emails()
            created_events = self.extract_and_create_events(processed_emails)
            calendar_events = self.get_daily_calendar()
            
            # Generate final briefing
            self.generate_daily_briefing(processed_emails, created_events, calendar_events)
            
            # Show completion time
            elapsed = datetime.now() - start_time
            console.print(f"\n[dim]✨ Briefing completed in {elapsed.total_seconds():.1f} seconds[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n\n[yellow]⚠️ Process interrupted by user[/yellow]")
            sys.exit(0)
        except Exception as e:
            console.print(f"\n\n[red]❌ Unexpected error: {e}[/red]")
            sys.exit(1)

def main():
    """Main entry point."""
    # Check if models exist
    models_dir = Path("models/lora_adapters")
    if not models_dir.exists() or not (models_dir / "triage").exists():
        console.print("\n[red]❌ Fine-tuned models not found![/red]")
        console.print("[yellow]Please run the training script first:[/yellow]")
        console.print("[cyan]python models/train.py[/cyan]")
        sys.exit(1)
    
    # Create and run the agent
    agent = DailyBriefingAgent()
    agent.run()

if __name__ == "__main__":
    main()