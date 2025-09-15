#!/usr/bin/env python3
"""
Fine-tuned model inference tools for email triage and event extraction.
Uses LoRA-adapted FLAN-T5-Small models for reliable classification and extraction.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

class TriageAgent:
    """
    Email triage agent using fine-tuned FLAN-T5-Small with LoRA adapters.
    Classifies emails as HIGH, MEDIUM, or LOW priority.
    """
    
    def __init__(self, model_path: str = "models/lora_adapters/triage"):
        """
        Initialize the triage agent.
        
        Args:
            model_path: Path to the LoRA adapter directory
        """
        self.model_path = Path(model_path)
        self.base_model_name = "google/flan-t5-small"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model and LoRA adapters."""
        try:
            print(f"🤖 Loading triage model from {self.model_path}")
            
            # Load tokenizer
            if self.model_path.exists() and (self.model_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            else:
                print("⚠ Using base tokenizer (LoRA adapters not found)")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load LoRA adapters if available
            if self.model_path.exists() and (self.model_path / "adapter_config.json").exists():
                print("✓ Loading LoRA adapters for triage model")
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            else:
                print("⚠ LoRA adapters not found, using base model")
                self.model = base_model
            
            # Move to device
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ Triage model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load triage model: {e}")
            raise
    
    def classify(self, text: str) -> str:
        """
        Classify email priority.
        
        Args:
            text: Email content to classify
        
        Returns:
            str: Priority classification (HIGH, MEDIUM, LOW)
        """
        try:
            # First check for specific rules (course codes, third-party apps, OTP)
            text_upper = text.upper()
            
            # Only use heuristics for third-party apps - let AI model handle everything else
            # Check for third-party apps (LOW priority, except OTP)
            if self._is_third_party_app(text_upper) and not self._is_otp_email(text_upper):
                print("📱 Third-party app detected - LOW priority")
                return "LOW"
            
            # Check for OTP emails (HIGH priority even from third-party apps)
            if self._is_otp_email(text_upper):
                print("🔐 OTP email detected - HIGH priority")
                return "HIGH"
            
            # Check for course codes (just for debugging, no classification)
            if self._has_course_code(text_upper):
                print("🎓 Course code detected - AI will classify")
            
            # Check for infrastructure announcements (just for debugging, no classification)
            if self._is_infrastructure_announcement(text_upper):
                print("🏢 Infrastructure announcement detected - AI will classify")
            
            # Check for general information emails (just for debugging, no classification)
            if self._is_general_info_email(text_upper):
                print("ℹ️ General information email detected - AI will classify")
            
            # Try the AI model first (let it make the decision)
            prompts = [
                f"Classify email priority: {text}",
                f"Priority: {text}",
                f"Rate priority: {text}",
                text  # Sometimes the model works better with just the text
            ]
            
            best_prediction = None
            
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate prediction
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=32,
                        num_beams=3,
                        temperature=0.1,
                        do_sample=False,
                        early_stopping=True
                    )
                
                # Decode output
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = prediction.strip().upper()
                
                # Try to extract priority from the prediction
                priority = self._extract_priority(prediction)
                if priority:
                    best_prediction = priority
                    break
            
            # If AI model gave a valid prediction, use it
            if best_prediction:
                return best_prediction
            
            # Only use heuristics as fallback if AI model completely failed
            best_prediction = self._heuristic_classification(text)
            # Only print warning occasionally to avoid spam
            if hasattr(self, '_warning_count'):
                self._warning_count += 1
            else:
                self._warning_count = 1
            
            if self._warning_count % 10 == 1:  # Print every 10th warning
                print(f"⚠ Using heuristic classification (warning #{self._warning_count})")
            
            return best_prediction
            
        except Exception as e:
            print(f"❌ Error classifying email: {e}")
            return "MEDIUM"  # Safe default
    
    def _extract_priority(self, prediction: str) -> Optional[str]:
        """
        Extract priority from model prediction, handling various formats.
        
        Args:
            prediction: Raw model prediction
        
        Returns:
            Optional[str]: Extracted priority or None if not found
        """
        prediction = prediction.strip().upper()
        
        # Direct matches (most reliable)
        valid_priorities = ["HIGH", "MEDIUM", "LOW"]
        for priority in valid_priorities:
            if priority in prediction:
                return priority
        
        # Try to find priority in common patterns
        # High priority indicators
        if any(word in prediction for word in ["URGENT", "IMMEDIATE", "ASAP", "CRITICAL", "EMERGENCY", "FINAL", "EXAM"]):
            return "HIGH"
        
        # Medium priority indicators (broader set)
        elif any(word in prediction for word in ["IMPORTANT", "MEETING", "ASSIGNMENT", "PROJECT", "QUIZ", "PRESENTATION", "DEADLINE", "DUE", "SUBMIT", "REVIEW", "UPDATE", "CHANGE", "NOTICE"]):
            return "MEDIUM"
        
        # Low priority indicators
        elif any(word in prediction for word in ["REMINDER", "INFO", "GENERAL", "ANNOUNCEMENT", "NEWSLETTER", "UPDATE", "NOTICE"]):
            return "LOW"
        
        # If prediction contains numbers or seems like a confidence score, try to interpret
        if any(char.isdigit() for char in prediction):
            # Look for patterns like "HIGH: 0.8" or "MEDIUM 85%"
            if "HIGH" in prediction:
                return "HIGH"
            elif "MEDIUM" in prediction:
                return "MEDIUM"
            elif "LOW" in prediction:
                return "LOW"
        
        return None
    
    def _heuristic_classification(self, text: str) -> str:
        """
        Fallback heuristic classification - only for third-party apps.
        All other emails should be classified by AI model.
        
        Args:
            text: Email content to classify
        
        Returns:
            str: Priority classification (HIGH, MEDIUM, LOW)
        """
        text_upper = text.upper()
        
        # Only use heuristics for third-party apps - let AI model handle everything else
        # Check for third-party apps (LOW priority, except OTP)
        if self._is_third_party_app(text_upper) and not self._is_otp_email(text_upper):
            return "LOW"
        
        # Check for OTP emails (HIGH priority even from third-party apps)
        if self._is_otp_email(text_upper):
            return "HIGH"
        
        # All other emails should be classified by AI model, not heuristics
        # This method should only be used as fallback when AI model fails
        print("⚠️ Using heuristic fallback - AI model should have handled this")
        return "MEDIUM"  # Safe default
    
    def _has_course_code(self, text: str) -> bool:
        """
        Check if email contains course codes (e.g., ABC123, ABC123D).
        
        Args:
            text: Email content in uppercase
        
        Returns:
            bool: True if course code pattern found
        """
        import re
        
        # Pattern for course codes: Only ABC123 or ABC123D format
        course_patterns = [
            r'\b[A-Z]{3}\d{3}\b',      # ABC123
            r'\b[A-Z]{3}\d{3}[A-Z]\b', # ABC123D
        ]
        
        for pattern in course_patterns:
            if re.search(pattern, text):
                return True
        
        return False
    
    def _is_third_party_app(self, text: str) -> bool:
        """
        Check if email is from a third-party app.
        
        Args:
            text: Email content in uppercase
        
        Returns:
            bool: True if from third-party app
        """
        third_party_keywords = [
            "GRAMMARLY", "ADOBE", "FACEBOOK", "TWITTER",
            "LINKEDIN", "INSTAGRAM", "YOUTUBE", "SPOTIFY", "NETFLIX", "AMAZON",
            "PAYPAL", "STRIPE", "SQUARE", "SHOPIFY", "MAILCHIMP", "CONSTANT CONTACT",
            "HUBSPOT", "SALESFORCE", "ZENDESK", "FRESHWORKS", "INTERCOM",
            "SLACK", "DISCORD", "ZOOM", "WEBEX", "GOTOMEETING", "SKETCH", "INVISION", "EVERNOTE",
            "TRELLO", "ASANA", "MONDAY", "JIRA", "CONFLUENCE", "ATLASSIAN"
        ]
        
        # Check sender domain patterns
        third_party_domains = [
            "@grammarly.com", "@adobe.com", "@facebook.com", "@twitter.com", "@linkedin.com", "@instagram.com",
            "@youtube.com", "@spotify.com", "@netflix.com", "@amazon.com",
            "@paypal.com", "@stripe.com", "@square.com", "@shopify.com",
            "@mailchimp.com", "@constantcontact.com", "@hubspot.com",
            "@salesforce.com", "@zendesk.com", "@freshworks.com", "@intercom.com",
            "@slack.com", "@discord.com", 
            "@webex.com", "@gotomeeting.com", "@canva.com", 
            "@sketch.com", "@invision.com", "@notion.so", "@evernote.com",
            "@trello.com", "@asana.com", "@monday.com", "@atlassian.com"
        ]
        
        # Check for third-party keywords in subject or body
        if any(keyword in text for keyword in third_party_keywords):
            return True
        
        # Check for third-party domains
        if any(domain in text for domain in third_party_domains):
            return True
        
        return False
    
    def _is_otp_email(self, text: str) -> bool:
        """
        Check if email is an OTP/verification code email.
        
        Args:
            text: Email content in uppercase
        
        Returns:
            bool: True if OTP email
        """
        otp_keywords = [
            "OTP", "VERIFICATION CODE", "AUTHENTICATION CODE", "SECURITY CODE",
            "VERIFICATION", "AUTHENTICATION", "SECURITY", "CODE", "PIN",
            "ONE TIME PASSWORD", "TWO FACTOR", "2FA", "MFA"
        ]
        
        return any(keyword in text for keyword in otp_keywords)
    
    def _is_infrastructure_announcement(self, text: str) -> bool:
        """
        Check if email is about internet connectivity or electrical shutdown.
        
        Args:
            text: Email content in uppercase
        
        Returns:
            bool: True if infrastructure announcement
        """
        infrastructure_keywords = [
            "INTERNET", "CONNECTIVITY", "NETWORK", "WIFI", "LAN", "BROADBAND",
            "ELECTRICAL", "POWER", "SHUTDOWN", "MAINTENANCE", "OUTAGE",
            "SERVICE INTERRUPTION", "SYSTEM MAINTENANCE", "INFRASTRUCTURE",
            "ELECTRICITY", "POWER CUT", "LOAD SHEDDING", "GENERATOR",
            "UPS", "BACKUP POWER", "ELECTRICAL MAINTENANCE",
            "NETWORK MAINTENANCE", "INTERNET MAINTENANCE", "SERVICE RESTORATION"
        ]
        
        return any(keyword in text for keyword in infrastructure_keywords)
    
    def _is_direct_email(self, text: str) -> bool:
        """
        Check if email was sent directly to you (no CC or BCC).
        
        Args:
            text: Email content with headers
        
        Returns:
            bool: True if direct email (no CC/BCC)
        """
        # Look for CC and BCC headers in the email content
        lines = text.split('\n')
        
        has_cc = False
        has_bcc = False
        
        for line in lines:
            line_upper = line.upper().strip()
            if line_upper.startswith('CC:'):
                cc_content = line_upper[3:].strip()
                if cc_content and cc_content != '':
                    has_cc = True
            elif line_upper.startswith('BCC:'):
                bcc_content = line_upper[4:].strip()
                if bcc_content and bcc_content != '':
                    has_bcc = True
        
        # Direct email if no CC and no BCC
        return not has_cc and not has_bcc
    
    def _is_general_info_email(self, text: str) -> bool:
        """
        Check if email contains general information (bus details, PhD synopsis, advisory, etc.).
        
        Args:
            text: Email content in uppercase
        
        Returns:
            bool: True if general information email
        """
        general_info_keywords = [
            # Transportation
            "BUS", "BUS ROUTE", "BUS TIMING", "BUS SCHEDULE", "TRANSPORT", "SHUTTLE",
            "BUS STOP", "BUS NUMBER", "ROUTE", "PICKUP", "DROP",
            
            # Academic/Research
            "PHD", "SYNOPSIS", "THESIS", "RESEARCH", "SCHOLAR", "SCHOLARSHIP",
            "DISSERTATION", "VIVA", "DEFENSE", "ACADEMIC", "FACULTY",
            
            # Advisory/Information
            "ADVISORY", "NOTICE", "ANNOUNCEMENT", "INFORMATION", "GUIDELINES",
            "POLICY", "RULES", "REGULATIONS", "PROCEDURES", "INSTRUCTIONS",
            
            # Room/Venue
            "LHC", "LHC ROOM", "LECTURE HALL", "CONFERENCE ROOM", "AUDITORIUM",
            "SEMINAR HALL", "MEETING ROOM", "VENUE", "LOCATION", "ROOM",
            
            # General announcements
            "CIRCULAR", "BULLETIN", "NEWSLETTER", "UPDATE", "REMINDER",
            "GENERAL", "PUBLIC", "COMMUNITY", "CAMPUS", "INSTITUTE",
            
            # Administrative
            "ADMINISTRATIVE", "OFFICE", "REGISTRAR", "ACCOUNTS", "EXAMINATION",
            "RESULT", "GRADE", "MARK", "SCORE", "CERTIFICATE",
            
            # Events/Activities
            "EVENT", "ACTIVITY", "PROGRAM", "WORKSHOP", "SEMINAR",
            "CONFERENCE", "SYMPOSIUM", "FESTIVAL", "CELEBRATION", "FUNCTION"
        ]
        
        return any(keyword in text for keyword in general_info_keywords)

class ExtractionAgent:
    """
    Event extraction agent using fine-tuned FLAN-T5-Small with LoRA adapters.
    Extracts structured event information from high-priority emails.
    """
    
    def __init__(self, model_path: str = "models/lora_adapters/extraction"):
        """
        Initialize the extraction agent.
        
        Args:
            model_path: Path to the LoRA adapter directory
        """
        self.model_path = Path(model_path)
        self.base_model_name = "google/flan-t5-small"
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self._load_model()
    
    def _load_model(self):
        """Load the base model and LoRA adapters."""
        try:
            print(f"🤖 Loading extraction model from {self.model_path}")
            
            # Load tokenizer
            if self.model_path.exists() and (self.model_path / "tokenizer_config.json").exists():
                self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            else:
                print("⚠ Using base tokenizer (LoRA adapters not found)")
                self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            
            # Load base model
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Load LoRA adapters if available
            if self.model_path.exists() and (self.model_path / "adapter_config.json").exists():
                print("✓ Loading LoRA adapters for extraction model")
                self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
            else:
                print("⚠ LoRA adapters not found, using base model")
                self.model = base_model
            
            # Move to device
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print(f"✅ Extraction model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load extraction model: {e}")
            raise
    
    def extract_events(self, text: str) -> list:
        """
        Extract event information from email text.
        
        Args:
            text: Email content to extract events from
        
        Returns:
            list: List of extracted event data (empty if no events found)
        """
        try:
            # Format input for the model - try different prompts
            prompts = [
                f"Extract event information: {text}",
                f"Find events in: {text}",
                f"Extract calendar events: {text}",
                text  # Sometimes the model works better with just the text
            ]
            
            best_prediction = None
            
            for prompt in prompts:
                # Tokenize input
                inputs = self.tokenizer(
                    prompt,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt"
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate prediction
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=3,
                        temperature=0.1,
                        do_sample=False,
                        early_stopping=True
                    )
                
                # Decode output
                prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = prediction.strip()
                
                # Try to parse JSON output
                event_data = self._parse_event_json(prediction)
                if event_data:
                    best_prediction = event_data
                    break
            
            # If no valid prediction found, try heuristic extraction
            if not best_prediction:
                best_prediction = self._heuristic_event_extraction(text)
            
            if best_prediction:
                # Only print success occasionally to avoid spam
                if hasattr(self, '_success_count'):
                    self._success_count += 1
                else:
                    self._success_count = 1
                
                if self._success_count % 5 == 1:  # Print every 5th success
                    print(f"✅ Extracted event: {best_prediction.get('summary', 'Unknown Event')}")
                
                return [best_prediction]  # Return as list for consistency
            else:
                return []
            
        except Exception as e:
            print(f"❌ Error extracting event: {e}")
            return []
    
    def extract_event(self, text: str) -> Optional[Dict]:
        """
        Legacy method for backward compatibility.
        Extract single event information from email text.
        
        Args:
            text: Email content to extract events from
        
        Returns:
            Optional[Dict]: Extracted event data or None if no event found
        """
        events = self.extract_events(text)
        return events[0] if events else None
    
    def _parse_event_json(self, json_string: str) -> Optional[Dict]:
        """
        Parse and validate JSON output from the extraction model.
        
        Args:
            json_string: JSON string from model output
        
        Returns:
            Optional[Dict]: Parsed and validated event data
        """
        try:
            # Clean the input string
            json_string = json_string.strip()
            
            # Handle empty or invalid input
            if not json_string or json_string in ['', 'null', 'None', 'N/A', 'n/a']:
                return None
            
            # Try to find JSON-like content in the string
            import re
            
            # Look for JSON object patterns
            json_patterns = [
                r'\{[^{}]*\}',  # Simple JSON object
                r'\{.*\}',      # Any JSON object
            ]
            
            json_match = None
            for pattern in json_patterns:
                match = re.search(pattern, json_string)
                if match:
                    json_match = match.group(0)
                    break
            
            if not json_match:
                # Try to extract key-value pairs from text
                return self._extract_key_value_pairs(json_string)
            
            # Try to parse the JSON
            try:
                event_data = json.loads(json_match)
            except json.JSONDecodeError:
                # Try to fix common JSON issues
                fixed_json = self._fix_json_string(json_match)
                if fixed_json:
                    event_data = json.loads(fixed_json)
                else:
                    return self._extract_key_value_pairs(json_string)
            
            # Validate required fields
            if not isinstance(event_data, dict):
                return None
            
            required_fields = ['summary', 'start_time', 'end_time']
            for field in required_fields:
                if field not in event_data or not event_data[field]:
                    return None
            
            # Validate and clean data
            cleaned_data = {
                'summary': str(event_data['summary']).strip(),
                'start_time': str(event_data['start_time']).strip(),
                'end_time': str(event_data['end_time']).strip(),
                'description': str(event_data.get('description', '')).strip(),
                'location': str(event_data.get('location', '')).strip()
            }
            
            # Remove empty optional fields
            if not cleaned_data['description']:
                del cleaned_data['description']
            if not cleaned_data['location']:
                del cleaned_data['location']
            
            return cleaned_data
            
        except Exception as e:
            return None
    
    def _fix_json_string(self, json_string: str) -> Optional[str]:
        """
        Try to fix common JSON formatting issues.
        
        Args:
            json_string: Potentially malformed JSON string
        
        Returns:
            Optional[str]: Fixed JSON string or None if unfixable
        """
        try:
            # Remove extra characters and fix quotes
            json_string = json_string.strip()
            
            # Ensure it starts and ends with braces
            if not json_string.startswith('{'):
                json_string = '{' + json_string
            if not json_string.endswith('}'):
                json_string = json_string + '}'
            
            # Try to fix common quote issues
            json_string = json_string.replace("'", '"')
            
            # Try to parse to validate
            json.loads(json_string)
            return json_string
            
        except:
            return None
    
    def _extract_key_value_pairs(self, text: str) -> Optional[Dict]:
        """
        Extract key-value pairs from text when JSON parsing fails.
        
        Args:
            text: Text to extract information from
        
        Returns:
            Optional[Dict]: Extracted event data or None
        """
        try:
            import re
            
            # Look for common patterns
            patterns = {
                'summary': [r'title[:\s]+([^\n]+)', r'subject[:\s]+([^\n]+)', r'event[:\s]+([^\n]+)'],
                'start_time': [r'time[:\s]+([^\n]+)', r'start[:\s]+([^\n]+)', r'when[:\s]+([^\n]+)'],
                'end_time': [r'end[:\s]+([^\n]+)', r'until[:\s]+([^\n]+)'],
                'location': [r'location[:\s]+([^\n]+)', r'where[:\s]+([^\n]+)', r'room[:\s]+([^\n]+)'],
                'description': [r'description[:\s]+([^\n]+)', r'details[:\s]+([^\n]+)']
            }
            
            event_data = {}
            
            for field, field_patterns in patterns.items():
                for pattern in field_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        event_data[field] = match.group(1).strip()
                        break
            
            # If we found some data, create a basic event
            if event_data:
                # Ensure required fields have defaults
                if 'summary' not in event_data:
                    event_data['summary'] = 'Extracted Event'
                if 'start_time' not in event_data:
                    event_data['start_time'] = '2024-01-01T09:00:00Z'
                if 'end_time' not in event_data:
                    event_data['end_time'] = '2024-01-01T10:00:00Z'
                
                return event_data
            
            return None
            
        except Exception as e:
            return None
    
    def _heuristic_event_extraction(self, text: str) -> Optional[Dict]:
        """
        Fallback heuristic event extraction based on keywords and patterns.
        Enhanced to detect timed events like quizzes, assignment deadlines, etc.
        
        Args:
            text: Email content to extract events from
        
        Returns:
            Optional[Dict]: Extracted event data or None if no event found
        """
        text_upper = text.upper()
        
        # Look for time patterns
        import re
        from datetime import datetime, timedelta
        
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
            r'(\d{1,2}\s*(?:AM|PM|am|pm))',
            r'(today|tomorrow|yesterday)',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december))',
            r'(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec))'
        ]
        
        # Look for location patterns
        location_patterns = [
            r'(room\s+\d+)',
            r'(conference\s+room)',
            r'(library)',
            r'(lab\s+\d+)',
            r'(auditorium)',
            r'(hall\s+\d+)',
            r'(lhc)',
            r'(lecture\s+hall)',
            r'(seminar\s+hall)'
        ]
        
        # Enhanced event keywords for timed events
        timed_event_keywords = [
            'QUIZ', 'ASSIGNMENT', 'DEADLINE', 'DUE', 'SUBMIT', 'SUBMISSION',
            'EXAM', 'FINAL', 'MIDTERM', 'TEST', 'PRESENTATION', 'PRESENT',
            'MEETING', 'LECTURE', 'SEMINAR', 'WORKSHOP', 'TUTORIAL',
            'PROJECT', 'REPORT', 'THESIS', 'VIVA', 'DEFENSE',
            'INTERVIEW', 'CALL', 'CONFERENCE', 'SYMPOSIUM'
        ]
        
        # Check if text contains timed event-related content
        has_timed_event_keywords = any(keyword in text_upper for keyword in timed_event_keywords)
        has_time_info = any(re.search(pattern, text_upper) for pattern in time_patterns)
        
        if not (has_timed_event_keywords or has_time_info):
            return None
        
        # Extract basic event information
        event_data = {
            'summary': 'Timed Event',
            'start_time': '2024-01-01T09:00:00Z',
            'end_time': '2024-01-01T10:00:00Z',
            'description': text[:200] + '...' if len(text) > 200 else text
        }
        
        # Try to extract a better summary based on event type
        if 'QUIZ' in text_upper:
            event_data['summary'] = 'Quiz'
        elif 'ASSIGNMENT' in text_upper or 'DEADLINE' in text_upper or 'DUE' in text_upper:
            event_data['summary'] = 'Assignment Deadline'
        elif 'EXAM' in text_upper or 'FINAL' in text_upper or 'MIDTERM' in text_upper:
            event_data['summary'] = 'Exam'
        elif 'PRESENTATION' in text_upper or 'PRESENT' in text_upper:
            event_data['summary'] = 'Presentation'
        elif 'MEETING' in text_upper:
            event_data['summary'] = 'Meeting'
        elif 'LECTURE' in text_upper:
            event_data['summary'] = 'Lecture'
        elif 'SEMINAR' in text_upper:
            event_data['summary'] = 'Seminar'
        elif 'WORKSHOP' in text_upper:
            event_data['summary'] = 'Workshop'
        elif 'PROJECT' in text_upper:
            event_data['summary'] = 'Project Deadline'
        elif 'VIVA' in text_upper or 'DEFENSE' in text_upper:
            event_data['summary'] = 'Viva/Defense'
        else:
            # Use first few words as summary
            words = text.split()[:5]
            event_data['summary'] = ' '.join(words)
        
        # Try to extract and parse time information
        parsed_time = self._extract_time_from_text(text)
        if parsed_time:
            event_data['start_time'] = parsed_time['start_time']
            event_data['end_time'] = parsed_time['end_time']
        
        # Try to extract location
        for pattern in location_patterns:
            match = re.search(pattern, text_upper)
            if match:
                event_data['location'] = match.group(1)
                break
        
        return event_data
    
    def _extract_time_from_text(self, text: str) -> Optional[Dict]:
        """
        Extract time information from text and convert to proper datetime format.
        
        Args:
            text: Email content to extract time from
        
        Returns:
            Optional[Dict]: Dictionary with start_time and end_time or None
        """
        import re
        from datetime import datetime, timedelta
        
        text_upper = text.upper()
        now = datetime.now()
        
        # Look for specific time patterns
        time_patterns = [
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
            r'(\d{1,2}\s*(?:AM|PM|am|pm))',
            r'(today|tomorrow|yesterday)',
            r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)',
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})'
        ]
        
        # Find time references
        time_matches = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text_upper)
            time_matches.extend(matches)
        
        if not time_matches:
            return None
        
        # Try to parse the first time match
        time_str = time_matches[0]
        
        try:
            # Handle relative dates
            if 'TODAY' in time_str:
                event_date = now
            elif 'TOMORROW' in time_str:
                event_date = now + timedelta(days=1)
            elif 'YESTERDAY' in time_str:
                event_date = now - timedelta(days=1)
            else:
                # Try to parse as date with multiple formats
                try:
                    # Clean the time string
                    time_str = time_str.strip()
                    
                    # Try different date formats
                    date_formats = [
                        '%m/%d/%Y',      # MM/DD/YYYY
                        '%m-%d-%Y',      # MM-DD-YYYY
                        '%d/%m/%Y',      # DD/MM/YYYY
                        '%d-%m-%Y',      # DD-MM-YYYY
                        '%Y-%m-%d',      # YYYY-MM-DD
                        '%m/%d/%y',      # MM/DD/YY
                        '%m-%d-%y',      # MM-DD-YY
                        '%d/%m/%y',      # DD/MM/YY
                        '%d-%m-%y',      # DD-MM-YY
                        '%B %d, %Y',     # Month DD, YYYY
                        '%b %d, %Y',     # Mon DD, YYYY
                        '%d %B %Y',      # DD Month YYYY
                        '%d %b %Y',      # DD Mon YYYY
                    ]
                    
                    event_date = None
                    for fmt in date_formats:
                        try:
                            event_date = datetime.strptime(time_str, fmt)
                            break
                        except ValueError:
                            continue
                    
                    if event_date is None:
                        # If no format matches, use current date
                        event_date = now
                        
                except Exception as e:
                    print(f"⚠ Warning: Could not parse date '{time_str}': {e}")
                    event_date = now
            
            # Extract time from text if available - try multiple patterns
            time_patterns = [
                r'(\d{1,2}):?(\d{2})?\s*(AM|PM|am|pm)',  # 12:30 PM, 1:45am, etc.
                r'(\d{1,2}):(\d{2})',                    # 14:30, 09:15, etc. (24-hour)
                r'(\d{1,2})\s*(AM|PM|am|pm)',            # 12 PM, 1am, etc.
            ]
            
            time_found = False
            for pattern in time_patterns:
                time_match = re.search(pattern, text_upper)
                if time_match:
                    try:
                        if ':' in pattern:
                            # Pattern with minutes
                            hour = int(time_match.group(1))
                            minute = int(time_match.group(2)) if time_match.group(2) else 0
                            period = time_match.group(3) if len(time_match.groups()) > 2 else None
                        else:
                            # Pattern without minutes
                            hour = int(time_match.group(1))
                            minute = 0
                            period = time_match.group(2) if len(time_match.groups()) > 1 else None
                        
                        # Convert to 24-hour format if period is specified
                        if period:
                            period = period.upper()
                            if period == 'PM' and hour != 12:
                                hour += 12
                            elif period == 'AM' and hour == 12:
                                hour = 0
                        
                        # Validate hour and minute
                        if 0 <= hour <= 23 and 0 <= minute <= 59:
                            event_date = event_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                            time_found = True
                            print(f"🕐 Extracted time: {hour:02d}:{minute:02d}")
                            break
                    except (ValueError, IndexError) as e:
                        print(f"⚠ Warning: Error parsing time pattern '{pattern}': {e}")
                        continue
            
            if not time_found:
                # Default to 11:59 PM if no specific time found (as requested)
                event_date = event_date.replace(hour=23, minute=59, second=0, microsecond=0)
                print(f"🕐 No time found, using default: 23:59")
            
            # Create start and end times with proper ISO format
            start_time = event_date.strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
            
            # Calculate end time - if it's 11:59 PM, make it 11:59 PM next day
            if event_date.hour == 23 and event_date.minute == 59:
                end_time = (event_date + timedelta(minutes=1)).strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
            else:
                end_time = (event_date + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S') + 'Z'
            
            print(f"📅 Final event times - Start: {start_time}, End: {end_time}")
            
            # Validate the datetime strings before returning
            if len(start_time) > 30 or len(end_time) > 30:
                print(f"⚠ Warning: Generated datetime strings are too long: start={len(start_time)}, end={len(end_time)}")
                return None
            
            # Check for any non-standard characters
            if any(char.isalpha() and char not in ['T', 'Z'] for char in start_time + end_time):
                print(f"⚠ Warning: Generated datetime strings contain invalid characters")
                return None
            
            return {
                'start_time': start_time,
                'end_time': end_time
            }
            
        except Exception as e:
            print(f"⚠ Warning: Could not parse time '{time_str}': {e}")
            return None

def load_agents() -> tuple[TriageAgent, ExtractionAgent]:
    """
    Load both triage and extraction agents.
    
    Returns:
        tuple: (TriageAgent, ExtractionAgent)
    """
    print("🔄 Loading fine-tuned models...")
    
    triage_agent = TriageAgent()
    extraction_agent = ExtractionAgent()
    
    print("✅ All models loaded successfully")
    return triage_agent, extraction_agent

if __name__ == "__main__":
    """Test the fine-tuned models."""
    print("Testing fine-tuned models...")
    
    # Test triage
    try:
        triage_agent = TriageAgent()
        test_email = "URGENT: Final exam tomorrow at 9 AM. Please review all materials."
        priority = triage_agent.classify(test_email)
        print(f"Triage test - Priority: {priority}")
    except Exception as e:
        print(f"Triage test failed: {e}")
    
    # Test extraction  
    try:
        extraction_agent = ExtractionAgent()
        test_email = "Meeting scheduled for today at 3 PM in conference room."
        event = extraction_agent.extract_event(test_email)
        print(f"Extraction test - Event: {event}")
    except Exception as e:
        print(f"Extraction test failed: {e}")
    
    print("Model testing complete")