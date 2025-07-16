#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Resume-Job Matching and Recommendation System.

This script orchestrates the entire workflow from data ingestion to visualization
and provides a command-line interface for processing resumes and job postings.
"""

import os
import logging
import argparse
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import project modules
from src.data_ingestion import DataIngestion
from src.resume_parser import ResumeParser
from src.job_parser import JobParser
from src.embedding import EmbeddingGenerator
from src.matcher import Matcher
from src.recommendation import RecommendationEngine
from src.visualization import VisualizationEngine


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Resume-Job Matching and Recommendation System'
    )
    parser.add_argument(
        '--resume', 
        type=str,
        help='Path to the resume file (PDF, DOCX, or TXT)'
    )
    parser.add_argument(
        '--jobs', 
        type=str, 
        default=None,
        help='Path to the job postings CSV file (optional, uses sample data if not provided)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='./output',
        help='Directory to save output files (optional, defaults to ./output)'
    )
    parser.add_argument(
        '--top_n', 
        type=int, 
        default=5,
        help='Number of top job matches to return (optional, defaults to 5)'
    )
    parser.add_argument(
        '--max_jobs',
        type=int,
        default=1000,
        help='Maximum number of jobs to process (optional, defaults to 1000)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=100,
        help='Number of jobs to process in each batch (optional, defaults to 100)'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--run_web',
        action='store_true',
        help='Run the Flask web application'
    )
    return parser.parse_args()


def setup_directories(output_dir):
    """Create necessary directories if they don't exist."""
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create data directories if they don't exist
    data_dir = Path('./data')
    resumes_dir = data_dir / 'resumes'
    dummy_data_dir = data_dir / 'dummy_data'
    
    for directory in [data_dir, resumes_dir, dummy_data_dir]:
        directory.mkdir(exist_ok=True)
    
    return output_path

def load_dummy_jobs():
    """Load dummy job data if no job file is provided."""
    dummy_jobs_path = Path('./data/dummy_data/dummy_jobs.csv')
    
    # If dummy jobs file doesn't exist, create it
    if not dummy_jobs_path.exists():
        logger.info("Creating dummy job data...")
        dummy_jobs = pd.DataFrame({
            'job_id': range(1, 11),
            'title': [
                'Software Engineer', 'Data Scientist', 'Product Manager',
                'UX Designer', 'DevOps Engineer', 'Frontend Developer',
                'Backend Developer', 'Full Stack Developer', 'Machine Learning Engineer',
                'Data Analyst'
            ],
            'company': [
                'TechCorp', 'DataInsights', 'ProductInnovations',
                'DesignMasters', 'CloudOps', 'FrontendTech',
                'BackendSolutions', 'FullStackInc', 'AILabs',
                'AnalyticsFirst'
            ],
            'location': [
                'San Francisco, CA', 'New York, NY', 'Seattle, WA',
                'Austin, TX', 'Boston, MA', 'Chicago, IL',
                'Los Angeles, CA', 'Denver, CO', 'Atlanta, GA',
                'Portland, OR'
            ],
            'description': [
                'Develop and maintain software applications using Python and JavaScript.',
                'Analyze large datasets and build predictive models using machine learning techniques.',
                'Lead product development from conception to launch, working with cross-functional teams.',
                'Create user-centered designs and prototypes for web and mobile applications.',
                'Build and maintain CI/CD pipelines and cloud infrastructure using AWS and Kubernetes.',
                'Develop responsive web interfaces using React, HTML, CSS, and JavaScript.',
                'Build scalable backend services using Node.js, Python, and SQL/NoSQL databases.',
                'Develop full-stack applications using modern frameworks and technologies.',
                'Design and implement machine learning models for production environments.',
                'Analyze data and create visualizations and reports for business stakeholders.'
            ],
            'requirements': [
                'Python, JavaScript, Git, SQL, REST APIs, Agile',
                'Python, R, SQL, Machine Learning, Statistics, Data Visualization',
                'Product Management, Agile, User Stories, Roadmapping, Analytics',
                'UI/UX Design, Figma, Adobe XD, User Research, Prototyping',
                'AWS, Docker, Kubernetes, CI/CD, Linux, Terraform',
                'React, HTML, CSS, JavaScript, TypeScript, Responsive Design',
                'Node.js, Python, SQL, NoSQL, REST APIs, Microservices',
                'JavaScript, Python, React, Node.js, SQL, Git',
                'Python, TensorFlow, PyTorch, Machine Learning, Deep Learning',
                'SQL, Python, Excel, Tableau, Data Visualization'
            ],
            'salary_range': [
                '$100,000 - $150,000', '$120,000 - $160,000', '$130,000 - $180,000',
                '$90,000 - $130,000', '$110,000 - $160,000', '$90,000 - $140,000',
                '$100,000 - $150,000', '$110,000 - $160,000', '$130,000 - $180,000',
                '$80,000 - $120,000'
            ],
            'experience_level': [
                'Mid-Senior', 'Senior', 'Senior', 'Mid-Level', 'Senior',
                'Mid-Level', 'Mid-Senior', 'Mid-Senior', 'Senior', 'Entry-Mid'
            ]
        })
        
        # Create directory if it doesn't exist
        dummy_jobs_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save dummy jobs to CSV
        dummy_jobs.to_csv(dummy_jobs_path, index=False)
        logger.info(f"Dummy job data created and saved to {dummy_jobs_path}")
    
    # Load and return dummy jobs
    return pd.read_csv(dummy_jobs_path)

def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level based on verbose flag
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run web application if requested
    if args.run_web:
        logger.info("Starting Flask web application...")
        import subprocess
        flask_app_path = Path(__file__).parent / "src" / "frontend" / "app.py"
        subprocess.run(["python", str(flask_app_path)])
        return
    
    # Setup directories
    output_dir = setup_directories(args.output)
    logger.info(f"Output directory: {output_dir}")
    
    # Initialize data ingestion
    data_ingestion = DataIngestion()
    
    # Process resume
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            logger.error(f"Resume file not found: {resume_path}")
            return
        
        logger.info(f"Processing resume: {resume_path}")
        resume_text = data_ingestion.load_resume(str(resume_path))
        resume_parser = ResumeParser()
        resume_dict = {"content": resume_text}
        resume_data = resume_parser.parse_resume(resume_dict)
        logger.info(f"Resume parsed successfully: {resume_data['name']}")
    else:
        logger.error("No resume file provided. Use --resume to specify a resume file.")
        return
    
    # Load and process job postings
    if args.jobs:
        logger.info(f"Loading job postings from: {args.jobs}")
        job_data = pd.read_csv(args.jobs)
        # Limit the number of jobs to process
        if len(job_data) > args.max_jobs:
            logger.info(f"Limiting to {args.max_jobs} jobs out of {len(job_data)} total jobs")
            job_data = job_data.sample(n=args.max_jobs, random_state=42)  # Use random sampling
        logger.info(f"Loaded {len(job_data)} job postings")
    else:
        logger.info("No job file provided, using dummy data")
        job_data = load_dummy_jobs()
    
    # Parse job postings in batches
    job_parser = JobParser()
    parsed_jobs = []
    total_jobs = len(job_data)
    
    for start_idx in range(0, total_jobs, args.batch_size):
        end_idx = min(start_idx + args.batch_size, total_jobs)
        batch = job_data.iloc[start_idx:end_idx]
        logger.info(f"Processing job batch {start_idx//args.batch_size + 1}/{(total_jobs + args.batch_size - 1)//args.batch_size} ({start_idx+1}-{end_idx} of {total_jobs})")
        
        for _, job in batch.iterrows():
            job_dict = job.to_dict()
            parsed_job = job_parser.parse_job(job_dict)
            parsed_jobs.append(parsed_job)
    
    logger.info("Job postings parsed successfully")
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embedding_generator = EmbeddingGenerator()
    # Wrap single resume in a list for embedding generation
    resume_embeddings = embedding_generator.generate_resume_embeddings([resume_data])
    # Get the embedding for our resume (there's only one)
    resume_embedding = next(iter(resume_embeddings.values()))
    
    # Generate job embeddings in batches
    job_embeddings = {}
    for start_idx in range(0, len(parsed_jobs), args.batch_size):
        end_idx = min(start_idx + args.batch_size, len(parsed_jobs))
        batch = parsed_jobs[start_idx:end_idx]
        logger.info(f"Generating embeddings for job batch {start_idx//args.batch_size + 1}/{(len(parsed_jobs) + args.batch_size - 1)//args.batch_size} ({start_idx+1}-{end_idx} of {len(parsed_jobs)})")
        batch_embeddings = embedding_generator.generate_job_embeddings(batch)
        job_embeddings.update(batch_embeddings)
    
    logger.info("Embeddings generated successfully")
    
    # Match resume with jobs
    logger.info("Matching resume with job postings...")
    matcher = Matcher()
    # Wrap single resume in a list for matching
    matches = matcher.match_resumes_with_jobs([resume_data], parsed_jobs, resume_embeddings, job_embeddings)
    # Get matches for our resume (there's only one)
    resume_id = next(iter(matches.keys()))
    match_results = matches[resume_id]
    logger.info(f"Found {len(match_results)} matches")
    
    # Generate recommendations
    logger.info("Generating recommendations...")
    recommendation_engine = RecommendationEngine()
    # Set the matcher results in the recommendation engine
    recommendation_engine.set_matcher_results({resume_id: match_results})
    # Generate recommendations for our resume
    recommendations = recommendation_engine.generate_recommendations(resume_id, top_n=args.top_n)
    logger.info(f"Generated {len(recommendations)} recommendations")

    # Generate visualizations and report
    logger.info("Generating visualizations and report...")
    visualization_engine = VisualizationEngine(output_dir=output_dir)
    
    # Generate skill gap analysis
    skill_gap = recommendation_engine.get_skill_gap_analysis(resume_id)
    
    # Generate visualizations
    match_score_chart = visualization_engine.generate_match_score_chart(recommendations, resume_data['name'])
    skill_gap_chart = visualization_engine.generate_skill_gap_chart(skill_gap, resume_data['name'])
    skill_match_heatmap = visualization_engine.generate_skill_match_heatmap(recommendations, resume_data['name'])
    
    # Generate HTML report
    report_path = visualization_engine.generate_html_report(recommendations, skill_gap, resume_data['name'])
    logger.info(f"Report generated successfully: {report_path}")

    # Print recommendations to console
    print("\n" + "="*80)
    print(f"Resume: {resume_data['name']}")
    print("Top {} Job Recommendations:".format(args.top_n))
    print("="*80)
    
    for i, rec in enumerate(recommendations, 1):
        job = rec['job']  # The job details are in the 'job' key
        print(f"\n{i}. {job.get('title', 'Unknown Position')} at {job.get('company', 'Unknown Company')}")
        print(f"   Location: {job.get('location', 'Unknown')}")
        print(f"   Match Score: {rec['score']:.2%}")
        print(f"   Explanation: {rec['explanation']}")
        if rec['matching_skills']:
            print(f"   Matching Skills: {', '.join(rec['matching_skills'])}")
        if rec['missing_skills']:
            print(f"   Missing Skills: {', '.join(rec['missing_skills'])}")
        print("-"*80)


if __name__ == "__main__":
    main()