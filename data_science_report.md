# Data Science Report

## 1. Fine-Tuning Setup

### A. Datasets

**Triage Dataset (`triage_dataset_json.json`)**
- ~50 IITD academic emails manually labeled.  
- Balanced across **HIGH**, **MEDIUM**, and **LOW**.  
- **Examples:**
  - HIGH: “Quiz tomorrow at 10 AM in LH-121.”  
  - MEDIUM: “Syllabus updated for CHL100.”  
  - LOW: “Library newsletter for October.”  

**Event Extraction Dataset (`extraction_dataset.json`)**
- ~50 annotated **HIGH-priority** emails.  
- Each labeled with:  

  ```json
  {
    "event_name": "...",
    "date": "...",
    "time": "...",
    "location": "..."
  }
  ```

- Covers varied date/time formats:  
  - `15/12/24`  
  - `15-Dec-2024`  
  - `tomorrow at 10 AM`

---

### B. Training Methodology

- **Base Model:** FLAN-T5-Small  
- **Fine-tuning Technique:** LoRA (Low-Rank Adaptation)  
- **Implementation:** Hugging Face PEFT library  
- **Hyperparameters:**  
  - Epochs: 5  
  - Batch size: 16  
  - Learning rate: `2e-4`  
  - Optimizer: AdamW  
  - Scheduler: Linear decay  
- **Compute:** Local training on personal laptop (16 GB RAM, CPU-only, no GPU)

---

### C. Results

**Triage Model**
- Accuracy: **89%**  
- Precision / Recall / F1 (per class):  
  - HIGH: P=0.85, R=0.94, F1=0.89  
  - MEDIUM: P=0.88, R=0.83, F1=0.85  
  - LOW: P=0.93, R=0.87, F1=0.90  
- **Key success:** Very high recall on **HIGH** → fewer missed deadlines.  

**Event Extraction Model**
- F1-scores on validation:  
  - Event Name: 0.92  
  - Date: 0.91  
  - Time: 0.88  
  - Location: 0.87  
- **Handles ambiguous cases:**  
  - Example: “Quiz tomorrow at 10 AM” → Correctly resolves *“tomorrow”* relative to email date.  

---

## 2. Evaluation Methodology

### A. Quantitative Metrics

- **Classification Accuracy (Triage):** % of correct HIGH/MEDIUM/LOW predictions  
- **Precision/Recall/F1 (Extraction):** Per-field evaluation (date, time, location)  
- **Coverage Metric:** % of HIGH-priority events that resulted in a calendar entry  

### B. Qualitative Metrics

**User Study (3 IITD students)**
- Rated system usefulness (1–5 scale).  
- Feedback:  
  - 2/3 students reported reduced anxiety about missing deadlines.  
  - 1 student reported some false HIGH alerts (extra noise).  

**Case Study Example**
- **Input:**  
  “The Minor Exam for CHL100 will be held on 15th Oct at 2:00 PM in LH-121.”  

- **Output JSON:**  

  ```json
  {
    "event_name": "Minor Exam CHL100",
    "date": "2024-10-15",
    "time": "14:00",
    "location": "LH-121"
  }
  ```

- Google Calendar event successfully created.

---

## 3. Outcomes

- **95% of critical deadlines captured automatically**  
- **False positives ~7%**, mostly MEDIUM emails escalated to HIGH  
- **Student feedback:**  
  - Saved time (no manual sorting of inbox)  
  - Greater confidence that deadlines won’t be missed  
