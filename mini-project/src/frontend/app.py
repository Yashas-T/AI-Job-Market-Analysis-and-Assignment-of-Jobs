#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web Application for Resume-Job Matching System.

This module provides a web interface for the resume-job matching system
using Flask.
"""

import os
import sys
import logging
import json
import uuid
from datetime import datetime
from pathlib import Path
from werkzeug.utils import secure_filename

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    logging.error("Flask is not installed. Please install it using 'pip install flask'.")
    sys.exit(1)

# Import project modules
try:
    from data_ingestion import ResumeReader, JobReader, create_sample_data
    from resume_parser import ResumeParser
    from job_parser import JobParser
    from embedding import EmbeddingGenerator
    from matcher import Matcher
    from recommendation import RecommendationEngine
    from visualization import VisualizationEngine, VISUALIZATION_AVAILABLE
except ImportError as e:
    logging.error(f"Error importing project modules: {str(e)}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev_key_for_development_only')

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload size

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Set app configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Add datetime to all templates
@app.context_processor
def inject_now():
    return {'now': datetime.now()}

# Sample data path
SAMPLE_JOBS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               'data', 'dummy_data', 'sample_jobs.csv')

# Initialize components
resume_reader = ResumeReader(UPLOAD_FOLDER)
job_reader = JobReader(SAMPLE_JOBS_PATH)
resume_parser = ResumeParser()
job_parser = JobParser()
embedding_generator = EmbeddingGenerator()
matcher = Matcher()
recommendation_engine = RecommendationEngine()
visualization_engine = VisualizationEngine(OUTPUT_FOLDER)

# Helper functions
def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_resume(file_path):
    """Process a single resume file and return structured data."""
    try:
        # Read resume
        resume_data = resume_reader.read_resume(file_path)
        
        # Parse resume
        parsed_resume = resume_parser.parse(resume_data["content"])
        parsed_resume["file_path"] = file_path
        parsed_resume["file_name"] = resume_data["file_name"]
        
        return parsed_resume
    except Exception as e:
        logger.error(f"Error processing resume {file_path}: {str(e)}")
        return None

def process_jobs(file_path=None):
    """Process job postings and return structured data."""
    try:
        # If no file path is provided, use sample data
        if file_path is None:
            # Check if sample data exists, if not create it
            if not os.path.exists(SAMPLE_JOBS_PATH):
                create_sample_data(SAMPLE_JOBS_PATH)
            file_path = SAMPLE_JOBS_PATH
        
        # Read jobs
        parsed_jobs = job_reader.read_jobs(file_path)
        
        # Parse jobs if needed
        if parsed_jobs and not all("parsed" in job for job in parsed_jobs):
            parsed_jobs = job_parser.parse_multiple_jobs(parsed_jobs)
        
        logger.info(f"Successfully processed {len(parsed_jobs)} jobs")
        return parsed_jobs
    except Exception as e:
        logger.error(f"Error processing jobs {file_path}: {str(e)}")
        return []

def generate_embeddings(parsed_resume, parsed_jobs):
    """Generate embeddings for resume and jobs."""
    try:
        # Generate resume embedding
        resume_id = parsed_resume['file_path']
        resume_embeddings = {}
        resume_embedding = embedding_generator.generate_resume_embedding(parsed_resume)
        resume_embeddings[resume_id] = resume_embedding
        
        # Generate job embeddings
        job_embeddings = {}
        for job in parsed_jobs:
            job_id = str(id(job))
            job_embedding = embedding_generator.generate_job_embedding(job)
            job_embeddings[job_id] = job_embedding
        
        return resume_embeddings, job_embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        return {}, {}

def match_and_recommend(parsed_resume, parsed_jobs, resume_embeddings, job_embeddings):
    """Match resume with jobs and generate recommendations."""
    try:
        # Match resume with jobs
        matcher_results = matcher.match_resumes_with_jobs(
            [parsed_resume], parsed_jobs, resume_embeddings, job_embeddings
        )
        
        # Generate recommendations
        resume_id = parsed_resume['file_path']
        recommendation_engine.set_matcher_results(matcher_results)
        recommendations = recommendation_engine.generate_recommendations(resume_id)
        
        # Generate skill gap analysis
        skill_gap = recommendation_engine.get_skill_gap_analysis(resume_id)
        
        return recommendations, skill_gap
    except Exception as e:
        logger.error(f"Error matching and recommending: {str(e)}")
        return [], {}

def generate_visualizations(recommendations, skill_gap, resume_id):
    """Generate visualizations and HTML report."""
    try:
        # Generate HTML report
        report_path = visualization_engine.generate_html_report(
            recommendations, skill_gap, resume_id
        )
        
        return report_path
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")
        return None

# Routes
@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    # Check if a file was uploaded
    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['resume']
    
    # Check if the file is empty
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the resume
        parsed_resume = process_resume(file_path)
        
        # Process jobs (using sample data for now)
        parsed_jobs = process_jobs()
        
        # Generate embeddings
        resume_embeddings, job_embeddings = generate_embeddings(parsed_resume, parsed_jobs)
        
        # Match and recommend
        recommendations, skill_gap = match_and_recommend(
            parsed_resume, parsed_jobs, resume_embeddings, job_embeddings
        )
        
        # Generate visualizations and report
        report_path = generate_visualizations(
            recommendations, skill_gap, os.path.basename(file_path)
        )
        
        # Get the relative path for the report
        if report_path:
            report_filename = os.path.basename(report_path)
            return redirect(url_for('show_report', filename=report_filename))
        else:
            flash('Error generating report')
            return redirect(url_for('index'))
    
    flash('File type not allowed')
    return redirect(url_for('index'))

@app.route('/report/<filename>')
def show_report(filename):
    """Display the generated report."""
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/process', methods=['POST'])
def api_process():
    """API endpoint for processing resumes."""
    # Check if a file was uploaded
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['resume']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file is allowed
    if file and allowed_file(file.filename):
        # Generate a unique filename
        filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"{timestamp}_{unique_id}_{filename}"
        
        # Save the file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        
        # Process the resume
        parsed_resume = process_resume(file_path)
        
        # Process jobs (using sample data for now)
        parsed_jobs = process_jobs()
        
        # Generate embeddings
        resume_embeddings, job_embeddings = generate_embeddings(parsed_resume, parsed_jobs)
        
        # Match and recommend
        recommendations, skill_gap = match_and_recommend(
            parsed_resume, parsed_jobs, resume_embeddings, job_embeddings
        )
        
        # Return JSON response
        return jsonify({
            'resume': parsed_resume,
            'recommendations': recommendations,
            'skill_gap': skill_gap
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render the contact page."""
    return render_template('contact.html')

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template('500.html'), 500

# Main entry point
if __name__ == '__main__':
    # Check if Flask is available
    if not FLASK_AVAILABLE:
        logger.error("Flask is not installed. Please install it using 'pip install flask'.")
        sys.exit(1)
    
    # Check if visualization libraries are available
    if not VISUALIZATION_AVAILABLE:
        logger.warning("Visualization libraries not available. Some features will be limited.")
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)