# main.py - FastAPI Backend for Plagiarism Detection
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import numpy as np
import sqlite3
import json
import os
import uuid
import fitz  # PyMuPDF is imported as 'fitz'
from datetime import datetime
from io import BytesIO
from docx import Document
import traceback # Import traceback for better error logging

# Report generation
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# NLP Libraries
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from nltk.tokenize import sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

app = FastAPI(title="Plagiarism Detection API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
sentence_model = None
nlp = None
DB_NAME = 'plagiarism_db.sqlite'

# Initialize models
@app.on_event("startup")
async def startup_event():
    global sentence_model, nlp
    print("Loading sentence transformer model...")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Loading spaCy model...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy English model not found. Stylometry analysis will be limited.")
        print("To fix: run 'python -m spacy download en_core_web_sm'")
        nlp = None
    
    # Initialize database
    init_database()
    print("Server startup complete!")

# Database models
class PlagiarismRequest(BaseModel):
    text: str
    reference_texts: Optional[List[str]] = None # For manually pasted reference texts
    reference_file_contents: Optional[List[str]] = None # For content extracted from uploaded reference files
    check_type: str = "similarity"  # similarity, paraphrase, stylometry
    language: str = "en"
    threshold_exact: float = 0.90
    threshold_paraphrase: float = 0.70

class PlagiarismResponse(BaseModel):
    plagiarism_score: float
    highlighted_text: str
    detailed_results: List[Dict[str, Any]]
    source_references: List[Dict[str, Any]]
    report_id: str
    suggestions: Optional[List[str]] = None
    stylometry_analysis: Optional[Dict[str, Any]] = None

class BatchPlagiarismRequest(BaseModel):
    texts: List[str]
    reference_texts: Optional[List[str]] = None # Can also be used for batch with static reference texts
    reference_file_contents: Optional[List[str]] = None # For content extracted from uploaded reference files in batch context
    check_type: str = "similarity"

# Database setup
def init_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reports (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            original_text TEXT,
            plagiarism_score REAL,
            detailed_results TEXT,
            source_references TEXT,
            language TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reference_documents (
            id TEXT PRIMARY KEY,
            title TEXT,
            content TEXT,
            embeddings TEXT,
            language TEXT,
            created_at TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database '{DB_NAME}' initialized and tables checked.")

def save_report(report_data: Dict[str, Any]) -> str:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    report_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat()
    
    try:
        cursor.execute('''
            INSERT INTO reports (id, timestamp, original_text, plagiarism_score, 
                               detailed_results, source_references, language)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            report_id,
            timestamp,
            report_data['text'],
            report_data['score'],
            json.dumps(report_data['detailed_results']),
            json.dumps(report_data['source_references']),
            report_data.get('language', 'en')
        ))
        conn.commit()
    except sqlite3.Error as e:
        conn.rollback()
        print(f"Database error saving report: {e}")
        raise
    finally:
        conn.close()
    
    return report_id

# Text preprocessing
def extract_text_from_file(file_content: bytes, filename: str) -> str:
    """Extract text from uploaded files (PDF, DOCX, TXT)"""
    file_ext = filename.lower().split('.')[-1]
    
    if file_ext == 'pdf':
        try:
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            text = ""
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
            return text
        except Exception as e:
            # Catch PyMuPDF specific errors
            raise HTTPException(status_code=422, detail=f"Could not process PDF file '{filename}': {e}. Ensure it's a valid PDF.")
    
    elif file_ext == 'docx':
        try:
            doc = Document(BytesIO(file_content))
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not process DOCX file '{filename}': {e}. Ensure it's a valid DOCX.")
    
    elif file_ext == 'txt':
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            # Fallback for common encoding issues
            return file_content.decode('latin-1') 
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Could not process TXT file '{filename}': {e}.")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file format for '{filename}': {file_ext}. Please use PDF, DOCX, or TXT.")

def preprocess_text(text: str) -> List[str]:
    """Split text into sentences and clean them"""
    if not text:
        return []
    sentences = sent_tokenize(text)
    # Remove empty sentences and strip whitespace
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    return sentences

# Core plagiarism detection
def detect_plagiarism(input_sentences: List[str], reference_sentences: List[str], 
                     threshold_exact: float = 0.90, threshold_paraphrase: float = 0.70) -> Dict[str, Any]:
    """Detect plagiarism using sentence embeddings and cosine similarity"""
    
    if not input_sentences:
        return {
            "results": [],
            "plagiarism_score": 0.0,
            "stats": {"total_sentences": 0, "plagiarized": 0, "paraphrased": 0, "original": 0}
        }

    # If no references provided, treat all input as original (0% plagiarism)
    if not reference_sentences:
        results = [{
            "sentence": s,
            "similarity": 0.0,
            "category": "original",
            "color": "green",
            "best_match": "",
            "best_match_similarity": 0.0
        } for s in input_sentences]
        return {
            "results": results,
            "plagiarism_score": 0.0,
            "stats": {
                "total_sentences": len(input_sentences),
                "plagiarized": 0,
                "paraphrased": 0,
                "original": len(input_sentences)
            }
        }

    # Generate embeddings
    # Using async/background tasks might be better for real-time applications
    input_embeddings = sentence_model.encode(input_sentences, show_progress_bar=False)
    reference_embeddings = sentence_model.encode(reference_sentences, show_progress_bar=False)
    
    # Calculate similarity matrix
    similarity_matrix = cosine_similarity(input_embeddings, reference_embeddings)
    
    results = []
    total_sentences = len(input_sentences)
    plagiarized_count = 0
    paraphrased_count = 0
    
    for i, sentence in enumerate(input_sentences):
        max_similarity = np.max(similarity_matrix[i])
        best_match_idx = np.argmax(similarity_matrix[i])
        best_match = reference_sentences[best_match_idx]
        
        if max_similarity >= threshold_exact:
            category = "exact"
            color = "red"
            plagiarized_count += 1
        elif max_similarity >= threshold_paraphrase:
            category = "paraphrased"
            color = "yellow"
            paraphrased_count += 1
        else:
            category = "original"
            color = "green"
        
        results.append({
            "sentence": sentence,
            "similarity": float(max_similarity),
            "category": category,
            "color": color,
            "best_match": best_match,
            "best_match_similarity": float(max_similarity)
        })
    
    # Calculate overall plagiarism score
    plagiarism_score = (plagiarized_count + paraphrased_count * 0.5) / total_sentences * 100
    
    return {
        "results": results,
        "plagiarism_score": plagiarism_score,
        "stats": {
            "total_sentences": total_sentences,
            "plagiarized": plagiarized_count,
            "paraphrased": paraphrased_count,
            "original": total_sentences - plagiarized_count - paraphrased_count
        }
    }

def generate_paraphrase_suggestions(text: str) -> List[str]:
    """Generate paraphrase suggestions using simple text transformations"""
    suggestions = []
    
    if not nlp: # Ensure nlp model is loaded for stylometry
        suggestions.append("NLP model not loaded for detailed suggestions. Consider rephrasing manually.")
        return suggestions
    
    doc = nlp(text)
    
    # Example: if sentence is long, suggest shortening
    if len(text.split()) > 20:
        suggestions.append(f"Consider shortening or splitting this sentence: '{text[:50]}...'")
    
    # General suggestions
    suggestions.append("Try rephrasing the idea in your own words.")
    suggestions.append("Use synonyms for key terms to change the vocabulary.")
    suggestions.append("Change the sentence structure (e.g., active to passive or vice versa).")
    
    return suggestions[:3]  # Return top 3 suggestions

def analyze_stylometry(text: str) -> Dict[str, Any]:
    """Analyze writing style to detect potential ghostwriting or AI generation"""
    if not nlp:
        return {"error": "spaCy model not available for stylometry analysis. Please install 'en_core_web_sm'."}
    
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    if not sentences:
        return {
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "lexical_diversity": 0.0,
            "pos_distribution": {},
            "total_sentences": 0,
            "total_words": 0
        }

    words_in_sentences = [len(sent.split()) for sent in sentences]
    avg_sentence_length = np.mean(words_in_sentences) if words_in_sentences else 0.0
    
    all_words = [token.text.lower() for token in doc if token.is_alpha]
    avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0.0
    
    unique_words = set(all_words)
    lexical_diversity = len(unique_words) / len(all_words) if all_words else 0.0
    
    pos_counts = {}
    for token in doc:
        pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
    
    total_tokens = len(doc)
    pos_ratios = {pos: count/total_tokens for pos, count in pos_counts.items()}
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "lexical_diversity": lexical_diversity,
        "pos_distribution": pos_ratios,
        "total_sentences": len(sentences),
        "total_words": len(all_words)
    }

def generate_highlighted_text(sentences_data: List[Dict[str, Any]]) -> str:
    """Generate HTML with highlighted text"""
    html_parts = []
    for data in sentences_data:
        color = data["color"]
        sentence = data["sentence"]
        similarity = data["similarity"]
        
        # Sanitize sentence to prevent XSS and ensure proper HTML rendering
        escaped_sentence = sentence.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#39;')

        style = f"background-color: {color}; padding: 2px; margin: 1px; border-radius: 3px; opacity: 0.7;"
        html_parts.append(
            f'<span style="{style}" title="Similarity: {similarity:.2f}">{escaped_sentence}</span> '
        )
    
    return "".join(html_parts)

# Default reference corpus (sample data)
DEFAULT_REFERENCE_TEXTS = [
    "Artificial intelligence is a rapidly growing field that focuses on creating intelligent machines.",
    "Machine learning algorithms can learn from data without being explicitly programmed.",
    "Natural language processing enables computers to understand and generate human language.",
    "Deep learning uses neural networks with multiple layers to process complex data.",
    "Computer vision allows machines to interpret and understand visual information from the world.",
    "The Internet of Things connects everyday devices to the internet for data exchange.",
    "Cloud computing provides on-demand access to computing resources over the internet.",
    "Cybersecurity protects digital systems from malicious attacks and unauthorized access.",
    "Big data analytics helps organizations extract insights from large datasets.",
    "Blockchain technology provides secure and transparent record-keeping systems."
]

# API Endpoints
@app.post("/api/check-plagiarism", response_model=PlagiarismResponse)
async def check_plagiarism(request: PlagiarismRequest):
    """Main plagiarism detection endpoint"""
    try:
        # Preprocess input text
        input_sentences = preprocess_text(request.text)
        
        # Combine all reference texts: from pasted input and from uploaded files
        all_reference_texts = []
        if request.reference_texts:
            all_reference_texts.extend(request.reference_texts)
        if request.reference_file_contents:
            all_reference_texts.extend(request.reference_file_contents)
            
        final_reference_texts = all_reference_texts if all_reference_texts else DEFAULT_REFERENCE_TEXTS
        
        reference_sentences = []
        for ref_text in final_reference_texts:
            reference_sentences.extend(preprocess_text(ref_text))
        
        # Detect plagiarism
        detection_result = detect_plagiarism(
            input_sentences, 
            reference_sentences, 
            request.threshold_exact, 
            request.threshold_paraphrase
        )
        
        # Generate highlighted text
        highlighted_text = generate_highlighted_text(detection_result["results"])
        
        # Generate paraphrase suggestions for plagiarized sentences
        suggestions = []
        for result in detection_result["results"]:
            if result["category"] in ["exact", "paraphrased"]:
                sentence_suggestions = generate_paraphrase_suggestions(result["sentence"])
                suggestions.extend(sentence_suggestions)
        
        # Stylometry analysis
        stylometry_analysis = analyze_stylometry(request.text)
        
        # Prepare source references
        source_references = []
        for i, result in enumerate(detection_result["results"]):
            if result["category"] in ["exact", "paraphrased"]:
                source_references.append({
                    "sentence_index": i,
                    "original_sentence": result["sentence"],
                    "matched_sentence": result["best_match"],
                    "similarity_score": result["similarity"],
                    "source": "Reference Corpus" # Or specify actual source if known
                })
        
        # Save report to database
        report_data = {
            "text": request.text,
            "score": detection_result["plagiarism_score"],
            "detailed_results": detection_result["results"],
            "source_references": source_references,
            "language": request.language
        }
        report_id = save_report(report_data)
        
        return PlagiarismResponse(
            plagiarism_score=detection_result["plagiarism_score"],
            highlighted_text=highlighted_text,
            detailed_results=detection_result["results"],
            source_references=source_references,
            report_id=report_id,
            suggestions=suggestions[:5],  # Top 5 suggestions
            stylometry_analysis=stylometry_analysis
        )
        
    except HTTPException as e:
        raise e # Re-raise FastAPI HTTP exceptions directly
    except Exception as e:
        # Log the full exception for debugging
        print(f"Error in check_plagiarism: {e}")
        traceback.print_exc() # Use traceback.print_exc() to print the full traceback
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/api/upload-file")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process a single document file (e.g., for main text input)"""
    try:
        file_content = await file.read()
        text = extract_text_from_file(file_content, file.filename)
        
        return {"filename": file.filename, "text": text, "length": len(text)}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in upload_file: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")

@app.post("/api/batch-check")
async def batch_plagiarism_check(request: BatchPlagiarismRequest):
    """Process multiple texts for plagiarism detection"""
    try:
        results = []
        
        # Combine all reference texts: from pasted input and from uploaded files
        all_reference_texts = []
        if request.reference_texts:
            all_reference_texts.extend(request.reference_texts)
        if request.reference_file_contents:
            all_reference_texts.extend(request.reference_file_contents)
            
        final_reference_texts = all_reference_texts if all_reference_texts else DEFAULT_REFERENCE_TEXTS
        
        reference_sentences = []
        for ref_text in final_reference_texts:
            reference_sentences.extend(preprocess_text(ref_text))
        
        for i, text in enumerate(request.texts):
            input_sentences = preprocess_text(text)
            detection_result = detect_plagiarism(input_sentences, reference_sentences)
            
            results.append({
                "text_index": i,
                "plagiarism_score": detection_result["plagiarism_score"],
                "stats": detection_result["stats"],
                "preview": text[:100] + "..." if len(text) > 100 else text
            })
        
        return {"batch_results": results, "total_processed": len(request.texts)}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error in batch_plagiarism_check: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing batch: {str(e)}")

@app.get("/api/reports/{report_id}")
async def get_report(report_id: str):
    """Retrieve a specific plagiarism report"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    report = cursor.fetchone()
    conn.close()
    
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "id": report[0],
        "timestamp": report[1],
        "original_text": report[2],
        "plagiarism_score": report[3],
        "detailed_results": json.loads(report[4]),
        "source_references": json.loads(report[5]),
        "language": report[6]
    }

@app.get("/api/reports")
async def list_reports(limit: int = 10):
    """List recent plagiarism reports"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, timestamp, plagiarism_score, language 
        FROM reports 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, (limit,))
    
    reports = cursor.fetchall()
    conn.close()
    
    return {
        "reports": [
            {
                "id": report[0],
                "timestamp": report[1],
                "plagiarism_score": report[2],
                "language": report[3]
            }
            for report in reports
        ]
    }

@app.post("/api/generate-report/{report_id}")
async def generate_pdf_report(report_id: str):
    """Generate PDF report for a plagiarism check"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
    report_data_tuple = cursor.fetchone()
    conn.close()
    
    if not report_data_tuple:
        raise HTTPException(status_code=404, detail="Report not found")
    
    # Unpack report data from tuple
    report_id, timestamp, original_text, plagiarism_score, detailed_results_json, source_references_json, language = report_data_tuple
    
    # Parse JSON fields
    detailed_results = json.loads(detailed_results_json)
    source_references = json.loads(source_references_json)
    
    # Create PDF
    filename = f"plagiarism_report_{report_id}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles for highlighting
    styles.add(ParagraphStyle(name='RedHighlight', parent=styles['Normal'],
                                  backColor=colors.Color(1, 0, 0, alpha=0.2), borderRadius=2,
                                  borderPadding=2, borderColor=colors.red, borderWidth=0.5))
    styles.add(ParagraphStyle(name='YellowHighlight', parent=styles['Normal'],
                                  backColor=colors.Color(1, 1, 0, alpha=0.2), borderRadius=2,
                                  borderPadding=2, borderColor=colors.orange, borderWidth=0.5))
    styles.add(ParagraphStyle(name='GreenHighlight', parent=styles['Normal'],
                                  backColor=colors.Color(0, 1, 0, alpha=0.1), borderRadius=2,
                                  borderPadding=2, borderColor=colors.green, borderWidth=0.5))
    styles.add(ParagraphStyle(name='Heading2Custom', parent=styles['h2'], spaceAfter=6))
    styles.add(ParagraphStyle(name='NormalSmall', parent=styles['Normal'], fontSize=9))


    story = []
    
    # Title
    title = Paragraph(f"Plagiarism Detection Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2 * inch))
    
    # Report details
    details_data = [
        ["Report ID:", report_id],
        ["Generated:", timestamp],
        ["Plagiarism Score:", f"{plagiarism_score:.1f}%"],
        ["Language:", language]
    ]
    
    table = Table(details_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#E0E0E0")), # Header column background
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6)
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2 * inch))
    
    # Original text
    story.append(Paragraph("Original Document Text:", styles['Heading2Custom']))
    story.append(Paragraph(original_text, styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Detailed Results
    if detailed_results:
        story.append(Paragraph("Detailed Plagiarism Analysis:", styles['Heading2Custom']))
        
        # Build story with individual highlighted sentences
        for item in detailed_results:
            text_style = styles['GreenHighlight']
            if item['category'] == 'exact':
                text_style = styles['RedHighlight']
            elif item['category'] == 'paraphrased':
                text_style = styles['YellowHighlight']
            
            # Escape HTML special characters that might be in the original text
            escaped_sentence = item['sentence'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            story.append(Paragraph(escaped_sentence, text_style))
        story.append(Spacer(1, 0.2 * inch))
    
    # Source References
    if source_references:
        story.append(Paragraph("Potential Source References:", styles['Heading2Custom']))
        for ref in source_references[:5]: # Limit to top 5 references for brevity in PDF
            story.append(Paragraph(f"<b>Similarity: {ref['similarity_score']*100:.1f}%</b>", styles['NormalSmall']))
            story.append(Paragraph(f"<b>Your Sentence:</b> {ref['original_sentence']}", styles['NormalSmall']))
            story.append(Paragraph(f"<b>Matched Source:</b> {ref['matched_sentence']}", styles['NormalSmall']))
            story.append(Spacer(1, 0.1 * inch))
        story.append(Spacer(1, 0.2 * inch))

    story.append(Paragraph("--- End of Report ---", styles['h3']))

    doc.build(story)
    
    return FileResponse(filename, media_type='application/pdf', filename=filename)

@app.get("/")
async def root():
    return {
        "message": "Plagiarism Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/api/check-plagiarism",
            "/api/upload-file",
            "/api/batch-check",
            "/api/reports",
            "/api/generate-report/{report_id}"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)