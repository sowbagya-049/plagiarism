# AI-Powered Plagiarism Detection System

A comprehensive, state-of-the-art plagiarism detection system designed for modern academic and professional needs. This project leverages advanced Natural Language Processing (NLP) techniques to identify exact matches, paraphrasing, and stylistic inconsistencies across various document formats.

## 🚀 Key Features

- **Advanced Similarity Detection**: Uses `Sentence-Transformers` (SBERT) for deep semantic understanding, going beyond simple keyword matching.
- **Multi-Type Analysis**:
    - **Exact Match**: Detects word-for-word copying.
    - **Paraphrase Detection**: Identifies rephrased content using transformer-based embeddings.
    - **Stylometry Analysis**: Analyzes writing style (sentence length, word complexity, POS distribution) to identify potential ghostwriting or AI-generated text.
- **Batch Processing**: Check multiple documents simultaneously against a reference corpus.
- **Multi-Format Support**: Seamlessly process `.pdf`, `.docx`, and `.txt` files.
- **Automated PDF Reports**: Generate professional, detailed plagiarism reports with color-coded highlighting of suspicious segments.
- **Training Suite**: Includes a complete pipeline for fine-tuning models on datasets like PAN, Quora Question Pairs, and MRPC.
- **Interactive UI**: A clean, modern web interface for easy document submission and result visualization.

## 🛠️ Technology Stack

- **Backend**: [FastAPI](https://fastapi.tiangolo.com/) (Python)
- **Deep Learning**: [Sentence-Transformers](https://www.sbert.net/), [PyTorch](https://pytorch.org/)
- **NLP Utilities**: [NLTK](https://www.nltk.org/), [SpaCy](https://spacy.io/)
- **Database**: [SQLite](https://www.sqlite.org/) (via `sqlite3`)
- **Report Generation**: [ReportLab](https://www.reportlab.com/), [PyMuPDF](https://pymupdf.readthedocs.io/) (fitz)
- **Frontend**: HTML5, Vanilla CSS, JavaScript (Fetch API)

## 📋 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd plagiarism-detection-project
   ```

2. **Set up a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install fastapi uvicorn sentence-transformers spacy nltk reportlab pymupdf python-docx numpy pandas scikit-learn
   ```

4. **Download NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## 🖥️ Usage

### 1. Start the API Server
Run the FastAPI backend using Uvicorn:
```bash
python -m uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`. You can explore the interactive docs at `http://localhost:8000/docs`.

### 2. Run the Training & Evaluation Suite
To train the model or run performance benchmarks:
```bash
python app.py
```
This script handles dataset preparation, model fine-tuning, and system evaluation.

### 3. Access the Frontend
Simply open `index.html` in your web browser to start using the plagiarism checker.

## 📂 Project Structure

- `main.py`: The heart of the system—FastAPI backend containing detection logic, report generation, and API endpoints.
- `app.py`: Comprehensive suite for dataset management, model training, and performance/accuracy testing.
- `index.html`: Modern web interface for user interaction.
- `plagiarism_db.sqlite`: Local database for storing reports and reference document metadata.
- `datasets/`: Directory containing specialized datasets (PAN, Quora, MRPC) for training.
- `fine_tuned_model/`: Directory where the optimized model is saved after training.

---
