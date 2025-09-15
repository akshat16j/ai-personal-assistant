# AI Daily Briefing Agent V2

An intelligent email triage and calendar management system that processes IIT Delhi webmail and integrates with Google Calendar using AI-powered classification.

## 🚀 Features

- **📧 Smart Email Processing**: Fetches and processes emails from IIT Delhi webmail using IMAP
- **🤖 AI-Powered Classification**: Uses fine-tuned FLAN-T5-Small models for intelligent email priority classification
- **⏰ Automatic Timed Event Detection**: Automatically detects and creates calendar events for HIGH priority emails containing quizzes, assignments, deadlines, exams, meetings, and other timed events
- **📅 Google Calendar Integration**: Creates events from timed events and shows your real calendar
- **🔄 Dynamic Priority Reassignment**: Reassigns email priorities every time you run the system
- **🌐 Modern Web Interface**: Beautiful Flask-based dashboard with real-time processing
- **📊 Comprehensive Analytics**: Detailed statistics and processing reports

## 🧠 AI Model Features

- **Intelligent Classification**: AI models handle all email classification (except third-party apps)
- **Fine-tuned Models**: Custom trained on your specific email patterns and dataset
- **Balanced Priority Distribution**: Proper HIGH/MEDIUM/LOW classification based on content
- **Context-Aware**: Considers email content, sender, and context for accurate classification
- **Minimal Heuristics**: Only uses heuristics for third-party app filtering

## 📁 Project Structure

```
ai_briefing_agent/
├── main.py                          # Main CLI application
├── app.py                           # Flask web interface
├── hybrid_auth.py                   # Hybrid authentication (IMAP + Google)
├── imap_auth.py                     # IIT Delhi webmail authentication
├── requirements.txt                 # Python dependencies
├── creds.env                        # IIT Delhi webmail credentials
├── token.json                       # Google authentication token
├── data/                           # Training datasets
│   ├── triage_dataset_json.json    # Email triage training data
│   └── extraction_dataset.json     # Event extraction training data
├── models/                         # AI models and training
│   ├── train.py                    # Model training script
│   └── lora_adapters/              # Fine-tuned model adapters
│       ├── triage/                 # Email triage model
│       └── extraction/             # Event extraction model
├── tools/                          # Core functionality modules
│   ├── imap_tool.py                # IMAP email operations
│   ├── calendar_tool.py            # Google Calendar operations
│   └── finetuned_tools.py          # AI model interfaces
└── templates/                      # Web interface templates
    └── index.html                  # Main dashboard template
```

## 🛠️ Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure IIT Delhi Webmail
Edit `creds.env`:
```
WEBMAIL_USER=your_username
WEBMAIL_PASSWORD=your_password
```

### 3. Setup Google Calendar Integration
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google Calendar API
4. Create OAuth2 credentials
5. Download `credentials.json` and place in project root
6. Run the application and complete OAuth flow
7. The `token.json` file will be created automatically after OAuth

**Note**: `credentials.json` and `token.json` are not included in the repository for security reasons. You need to set them up locally.

### 4. Train AI Models (Optional)
```bash
python models/train.py
```
*Note: Pre-trained models are included. Retraining is only needed if you want to update with new data.*

## 🚀 Usage

### Web Interface (Recommended)
```bash
python app.py
```
- Open http://localhost:5000 in your browser
- Interactive dashboard with real-time processing
- Comprehensive statistics and email management
- One-click processing and calendar integration

### CLI Interface
```bash
python main.py
```
- Processes recent emails and reassigns priorities
- Shows comprehensive daily briefing
- Integrates with your Google Calendar

## 🔧 Key Components

### AI-Powered Classification
- **Triage Model**: Classifies emails as HIGH/MEDIUM/LOW priority using fine-tuned FLAN-T5-Small
- **Extraction Model**: Extracts structured event information from high-priority emails
- **Smart Filtering**: Automatically filters third-party apps while letting AI handle academic emails
- **Context-Aware**: Considers email content, course codes, and academic context

### Email Processing
- Fetches recent emails (last 30 days, up to 50 emails)
- AI-driven priority classification
- Automatic labeling and organization
- Handles both read and unread emails
- Robust error handling and recovery

### Calendar Integration
- Shows your real Google Calendar events
- **Automatically creates events for HIGH priority timed events**
- Intelligent time parsing from email content
- Supports various date/time formats
- Default time handling (11:59 PM for date-only events)

## 📊 Classification Logic

### HIGH Priority
- Urgent keywords: "URGENT", "IMMEDIATE", "ASAP", "CRITICAL", "EMERGENCY"
- Time-sensitive: "DUE TODAY", "DUE TOMORROW", "FAILURE", "PENALTY"
- Security codes: "OTP", "VERIFICATION CODE", "AUTHENTICATION CODE"
- Quiz/Exam terms: "QUIZ", "MINOR", "MAJOR", "EXAM", "FINAL", "MIDTERM", "VIVA", "DEFENSE"
- Direct emails: No CC/BCC (personal communication)

### MEDIUM Priority
- Academic terms: "SYLLABUS", "REMINDER", "ANNOUNCEMENT", "CIRCULAR", "BULLETIN"
- Course content: "COURSE", "CLASS", "TUTORIAL", "LAB", "LECTURE", "SEMINAR"
- Assignments: "ASSIGNMENT", "PROJECT", "DEADLINE", "SUBMISSION"
- Academic admin: "REGISTRATION", "ENROLLMENT", "ADMISSION", "APPLICATION"

### LOW Priority
- Third-party apps: Marketing, social media, notifications
- General information: "INFO", "GENERAL", "NEWSLETTER", "LIBRARY", "PARKING"
- Social content: "SOCIAL", "ENTERTAINMENT", "SPORTS", "RECREATION"

## ⏰ Timed Event Detection

The system automatically detects and creates calendar events for various timed activities mentioned in HIGH priority emails:

### Supported Event Types
- **Academic**: Quizzes, exams, assignments, project deadlines, presentations, lectures, seminars
- **Professional**: Meetings, interviews, conferences, workshops, calls
- **Research**: Thesis defense, viva, research presentations
- **General**: Any event with specific date/time information

### Time Parsing Capabilities
- **Relative dates**: "today", "tomorrow", "next Monday"
- **Specific dates**: "12/15/2024", "15-Dec-2024", "07.08.2025"
- **Time formats**: "10:00 AM", "2:30 PM", "3 PM", "23:59"
- **Combined**: "Quiz tomorrow at 10 AM", "Meeting on Friday 2 PM"
- **Default time**: 11:59 PM for date-only events

### Example Email Processing
```
Subject: Quiz Tomorrow at 10 AM (HIGH Priority)
Content: There will be a quiz tomorrow at 10:00 AM in Room 101.

→ Automatically creates: "Quiz" event for tomorrow at 10:00 AM
```

## 🔒 Security

- Credentials stored in environment files
- OAuth2 token-based authentication
- SSL/TLS encrypted connections
- No hardcoded passwords
- Secure credential management

## 📝 Notes

- The system processes both read and unread emails
- Priorities are reassigned every time you run it
- Google Calendar integration requires OAuth2 setup
- All models are fine-tuned for your specific use case
- AI models handle classification with minimal heuristic interference
- Only HIGH priority emails with timed events create calendar events

## 🎯 Use Cases

- Daily email triage and organization
- **Automatic timed event detection and calendar creation**
- Priority-based email management
- Comprehensive daily briefings
- Web-based email dashboard
- **Never miss important deadlines or events**
- Academic email management for students and faculty

## 🚀 Getting Started

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Configure credentials**: Edit `creds.env` with your IIT Delhi webmail credentials
4. **Setup Google Calendar**: Follow the OAuth2 setup process
5. **Run the application**: `python app.py`
6. **Open your browser**: Navigate to http://localhost:5000
7. **Start processing**: Click "Run AI Agent" to begin email processing

## 🤝 Contributing

This project is designed for IIT Delhi students and faculty. Feel free to customize the AI models and classification logic for your specific needs.

## 📄 License

This project is for educational and personal use. Please respect IIT Delhi's email policies and Google's API terms of service.