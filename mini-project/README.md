# Resume-Job Matching and Recommendation System

A comprehensive AI-powered system for matching resumes with job postings and providing personalized job recommendations based on skills, experience, and qualifications. This system uses natural language processing, semantic embeddings, and machine learning techniques to analyze resumes and job postings, identify matching skills, and generate tailored recommendations.

## Features

-
**Resume Parsing**: Extract structured information from resumes in PDF, DOCX, and TXT formats including contact details, skills, education, work experience, and projects
- **Job Posting Analysis**: Parse job descriptions to identify requirements, responsibilities, qualifications, location, and salary information
- **AI-Powered Matching**: Match resumes with job postings using advanced NLP and embedding techniques with weighted scoring based on semantic similarity, skill matching, and location
- **Skill Gap Analysis**: Identify missing skills and provide recommendations for improvement to increase job match potential
- **Personalized Recommendations**: Generate tailored job recommendations with detailed explanations of why each job is a good match
- **Visualization**: Create visual representations of match quality, skill comparisons, and skill gaps using interactive charts and graphs
- **Web Interface**: User-friendly web application for uploading resumes, viewing recommendations, and exploring job matches

## Project Structure

```
├── data/                  # Data directory
│   ├── dummy_data/        # Sample data for testing
│   └── resumes/           # Directory for uploaded resumes
├── src/                   # Source code
│   ├── data_ingestion.py  # Data loading and preprocessing
│   ├── resume_parser.py   # Resume parsing module
│   ├── job_parser.py      # Job posting parsing module
│   ├── embedding.py       # Text embedding generation
│   ├── matcher.py         # Resume-job matching logic
│   ├── recommendation.py  # Recommendation engine
│   ├── visualization.py   # Visualization utilities
│   └── frontend/          # Web application
│       ├── app.py         # Flask application
│       ├── templates/     # HTML templates
│       └── assets/        # Static assets
├── tests/                 # Test directory
│   └── dummy_data/        # Test data
├── main.py                # Main entry point
└── requirements.txt       # Project dependencies
```

## Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Basic Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/resume-job-matching.git
cd resume-job-matching
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download spaCy language model (required for optimal NLP functionality):

```bash
python -m spacy download en_core_web_md
```

### Advanced Installation Options

#### Minimal Installation (without visualization support)

If you only need the core functionality without visualization capabilities:

```bash
pip install -r requirements-minimal.txt
```

#### Full Installation (with all optional components)

For complete functionality including all visualization options and advanced NLP:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg  # Larger, more accurate language model
pip install torch torchvision  # For advanced embedding models
```

### Troubleshooting

- If you encounter issues with PDF extraction, ensure you have the necessary system dependencies for PyPDF2 or try installing pymupdf as an alternative.
- For visualization issues, check that matplotlib and its dependencies are correctly installed for your operating system.
- If sentence-transformers fails to install, you may need to install PyTorch separately following the instructions at https://pytorch.org/get-started/locally/

## Usage

The system can be used in two ways: through a command-line interface for batch processing or via a web interface for interactive use.

### Command Line Interface

The command-line interface is ideal for processing individual resumes against job databases or for batch processing in automated workflows.

Run the system from the command line:

```bash
python main.py --resume 10985403.pdf --jobs job_market_data.csv --output ./output
```

Options:
- `--resume`: Path to the resume file (PDF, DOCX, or TXT)
- `--jobs`: Path to the job postings CSV file (optional, uses sample data if not provided)
- `--output`: Directory to save output files (optional, defaults to './output')
- `--top_n`: Number of top job matches to return (optional, defaults to 5)
- `--verbose`: Enable verbose output for detailed logging

Example usage:

```bash
# Process a resume with sample job data
python main.py --resume data/resumes/my_resume.pdf --output ./my_results

# Process a resume with custom job data and return top 10 matches
python main.py --resume data/resumes/my_resume.pdf --jobs data/jobs/company_jobs.csv --top_n 10 --verbose
```

The system will:
1. Parse the resume to extract structured information
2. Load and parse job postings
3. Generate embeddings for the resume and jobs
4. Match the resume with jobs using multiple criteria
5. Generate recommendations with explanations
6. Create visualizations and an HTML report
7. Print a summary of the top recommendations to the console

### Web Interface

The web interface provides an interactive way to upload resumes, view job matches, and explore recommendations with visualizations.

Start the web application:

```bash
python src/frontend/app.py
```

Then open your browser and navigate to `http://localhost:5000`.

The web interface offers:

1. **Resume Upload**: Upload your resume in PDF, DOCX, or TXT format
2. **Automatic Processing**: The system automatically processes your resume and matches it with available jobs
3. **Interactive Results**: View your top job matches with detailed explanations
4. **Skill Gap Analysis**: See which skills you're missing for specific jobs
5. **Visualizations**: Explore interactive charts showing match scores and skill comparisons
6. **Downloadable Reports**: Save or print your job match report for future reference

The web application uses sample job data by default, but can be configured to use custom job databases.

## System Architecture

The system consists of several key components that work together to provide resume-job matching functionality:

### Component Overview

1. **Data Ingestion**: Handles the input of resume and job posting files in various formats (PDF, DOCX, etc.).

2. **Resume Parser**: Extracts structured information from resumes, including contact details, skills, experience, education, and projects.

3. **Job Parser**: Extracts structured information from job postings, including job title, company, location, required skills, and responsibilities.

4. **Embedding Generator**: Creates vector representations of resumes and job postings using advanced NLP techniques.

5. **Matcher**: Compares resume and job embeddings to calculate match scores based on content similarity, skill overlap, and location compatibility.

6. **Recommendation Engine**: Generates personalized job recommendations for each resume based on match scores and additional criteria.

7. **Visualization Engine**: Creates visual representations of match results, skill gaps, and other insights.

8. **Web Interface**: Provides a user-friendly interface for uploading resumes, viewing recommendations, and exploring insights.

### Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Resume Files   │     │   Job Postings  │
│  (PDF, DOCX)    │     │  (JSON, CSV)    │
│                 │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Data Ingestion │     │  Data Ingestion │
│                 │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │
│  Resume Parser  │     │   Job Parser    │
│                 │     │                 │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌───────────────────────────────────────────┐
│                                           │
│           Embedding Generator             │
│                                           │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│                                           │
│                 Matcher                   │
│                                           │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│                                           │
│           Recommendation Engine           │
│                                           │
└────────┬─────────────────────────┬────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│                 │       │                 │
│  Visualization  │       │   Web/API       │
│     Engine      │──────▶│   Interface     │
│                 │       │                 │
└─────────────────┘       └─────────────────┘
```

### Data Flow

1. **Input**: Resume and job posting files are uploaded through the web interface or provided via the API.

2. **Processing**: The files are processed by the respective parsers to extract structured information.

3. **Embedding**: The extracted information is converted into vector representations by the Embedding Generator.

4. **Matching**: The Matcher compares the embeddings to calculate match scores and identify skill gaps.

5. **Recommendation**: The Recommendation Engine generates personalized job recommendations based on the match results.

6. **Visualization**: The Visualization Engine creates charts and graphs to represent the match results and insights.

7. **Output**: The results are presented to the user through the web interface or returned via the API.

### Data Ingestion

The `DataIngestion` module handles loading and preprocessing of data:
- **ResumeReader**: Extracts text from PDF, DOCX, and TXT resume files using PyPDF2 and python-docx libraries
- **JobReader**: Loads job posting data from CSV files using pandas
- Provides data cleaning and normalization functions
- Includes functionality to generate sample data for testing

### Resume Parser

The `ResumeParser` class extracts structured information from resume text using NLP techniques:
- Uses spaCy for named entity recognition and pattern matching (with fallback to regex)
- Extracts personal information (name, email, phone, location)
- Identifies skills using a comprehensive skill database and pattern matching
- Parses education history (degree, institution, dates)
- Extracts work experience (job titles, companies, dates, descriptions)
- Identifies projects and achievements
- Handles different resume formats and section variations

### Job Parser

The `JobParser` class analyzes job postings to extract structured information:
- Extracts job title, company name, and posting date
- Identifies required skills and qualifications using pattern matching
- Parses responsibilities and duties into structured lists
- Normalizes location information for better matching
- Extracts salary ranges and benefits information
- Categorizes job requirements by importance (required vs. preferred)

### Embedding Generator

The `EmbeddingGenerator` class creates vector representations of resumes and job postings:
- Uses Sentence-BERT models to generate semantic embeddings
- Extracts relevant text from parsed resume and job data for embedding
- Handles resume text by focusing on skills, experience, and education
- Processes job descriptions with emphasis on requirements and responsibilities
- Provides fallback to dummy embeddings when models are unavailable
- Enables semantic matching beyond simple keyword comparison

### Matcher

The `Matcher` class compares resumes with job postings using a multi-factor approach:
- Calculates embedding similarity using cosine similarity (50% weight)
- Computes skill match scores using Jaccard similarity (40% weight)
- Determines location match based on exact or partial matches (10% weight)
- Identifies matching skills between resume and job requirements
- Lists skills required by the job but missing from the resume
- Produces a comprehensive match report with component scores

### Recommendation Engine

The `RecommendationEngine` class generates personalized job recommendations:
- Ranks jobs based on match scores from the Matcher
- Creates human-readable explanations for why each job is recommended
- Highlights matching skills that make the candidate a good fit
- Identifies skill gaps and suggests improvements
- Provides personalized career development recommendations
- Generates comprehensive skill gap analysis across all jobs

### Visualization Engine

The `VisualizationEngine` class creates visual representations of matching results:
- Generates interactive match score charts comparing different jobs
- Creates skill gap visualizations to identify improvement areas
- Produces skill match heatmaps showing strengths and weaknesses
- Builds comprehensive HTML reports with interactive visualizations
- Supports different visualization types (bar charts, radar charts, etc.)
- Provides fallback to text-based reports when visualization libraries are unavailable

## Dependencies

### Core Dependencies

- **Python**: 3.7 or higher
- **Data Processing**: NumPy, Pandas for data manipulation and analysis
- **Utilities**: python-dateutil, tqdm for progress tracking, requests for API calls

### NLP and Machine Learning

- **spaCy**: Industrial-strength NLP library for named entity recognition and text processing
- **NLTK**: Natural Language Toolkit for text analysis and processing
- **sentence-transformers**: For generating semantic embeddings using transformer models
- **transformers** (optional): For advanced NLP capabilities and state-of-the-art language models

### Document Processing

- **PyPDF2**: For extracting text from PDF files
- **python-docx**: For parsing DOCX documents
- **pymupdf** (optional): Alternative PDF processing library with better performance for complex PDFs

### Visualization

- **Matplotlib**: For creating static visualizations and charts
- **Seaborn**: For enhanced statistical visualizations
- **Plotly** (optional): For interactive visualizations in the web interface

### Web Framework

- **Flask**: Lightweight web framework for the application interface
- **Werkzeug**: WSGI utility library for Flask
- **Flask-WTF**: Form handling for Flask
- **python-multipart**: For handling file uploads in the web interface

### Development Tools

- **pytest** (optional): For running unit tests
- **black** (optional): For code formatting
- **flake8** (optional): For code linting

## API Documentation

The system provides a RESTful API that allows programmatic access to the resume-job matching functionality.

### Endpoints

#### POST /api/process

Process a resume and return matching jobs, recommendations, and skill gap analysis.

**Request:**

- Content-Type: `multipart/form-data`
- Body:
  - `resume`: File upload (PDF or DOCX)

**Response:**

```json
{
  "parsed_resume": {
    "contact_info": { ... },
    "skills": [ ... ],
    "experience": [ ... ],
    "education": [ ... ],
    "projects": [ ... ]
  },
  "recommendations": [
    {
      "job_id": "job123",
      "job_title": "Software Engineer",
      "company": "Tech Corp",
      "location": "San Francisco, CA",
      "match_score": 0.85,
      "skill_match_score": 0.9,
      "content_match_score": 0.8,
      "location_match_score": 0.85,
      "matching_skills": [ ... ],
      "missing_skills": [ ... ],
      "explanation": "..."
    },
    ...
  ],
  "skill_gap_analysis": {
    "missing_skills": {
      "python": 5,
      "react": 3,
      ...
    },
    "total_jobs_analyzed": 10
  }
}
```

### Using the API

#### Python Example

```python
import requests

url = "http://localhost:5000/api/process"

with open("resume.pdf", "rb") as f:
    files = {"resume": ("resume.pdf", f, "application/pdf")}
    response = requests.post(url, files=files)

if response.status_code == 200:
    data = response.json()
    recommendations = data["recommendations"]
    skill_gaps = data["skill_gap_analysis"]
    
    # Process the recommendations and skill gaps
    for rec in recommendations:
        print(f"Job: {rec['job_title']} at {rec['company']}")
        print(f"Match Score: {rec['match_score']}")
        print(f"Missing Skills: {', '.join(rec['missing_skills'])}")
        print("---")
```

#### cURL Example

```bash
curl -X POST \
  -F "resume=@/path/to/resume.pdf" \
  http://localhost:5000/api/process
```

## Contributing

Contributions are welcome! Here's how you can contribute to the project:

### Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/resume-job-matching.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install development dependencies: `pip install -r requirements-dev.txt`

### Code Style

This project follows PEP 8 style guidelines. We recommend using the following tools to ensure code quality:

- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting

You can run these tools before committing:

```bash
black src/
flake8 src/
isort src/
```

### Testing

Write tests for new features and ensure existing tests pass:

```bash
pytest tests/
```

### Pull Request Process

1. Update the README.md with details of changes if applicable
2. Update the requirements.txt file if you've added new dependencies
3. Make sure all tests pass
4. Submit a pull request with a clear description of the changes

### Feature Requests and Bug Reports

Please use the GitHub Issues section to report bugs or request features.

## Future Development

The following features and improvements are planned for future releases:

### Short-term Goals

- **Enhanced Resume Parsing**: Improve extraction accuracy for complex resume formats
- **Additional File Formats**: Support for more document types (HTML, RTF, etc.)
- **Multilingual Support**: Add capability to process resumes in languages other than English
- **Improved Skill Matching**: Enhanced algorithms for matching skills with different phrasings

### Medium-term Goals

- **User Accounts**: Allow users to create accounts and save their resumes and job matches
- **Job Scraping Integration**: Automatically fetch job postings from popular job boards
- **Custom Recommendation Filters**: Allow users to filter recommendations based on various criteria
- **Resume Improvement Suggestions**: Provide specific suggestions to improve resume content

### Long-term Goals

- **Advanced ML Models**: Implement more sophisticated machine learning models for matching
- **Interview Preparation**: Generate potential interview questions based on job requirements
- **Career Path Analysis**: Suggest potential career paths based on current skills and interests
- **Salary Insights**: Provide salary range information for recommended positions

### How to Contribute to Future Development

If you're interested in contributing to any of these features, please check the GitHub Issues section for related tasks or create a new issue to discuss your ideas.

## License

This project is licensed under the MIT License - see the LICENSE file for details.