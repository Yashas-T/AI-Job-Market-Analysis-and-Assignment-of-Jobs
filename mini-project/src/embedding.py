#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Embedding Module for Resume-Job Matching System.

This module provides functionality to generate embeddings for resumes and job postings
using Sentence-BERT models.
"""

import logging
from typing import Dict, List, Any, Union
import numpy as np

# For generating embeddings
try:
    from sentence_transformers import SentenceTransformer
    model_available = True
except ImportError:
    logging.warning("sentence-transformers not found. Using dummy embeddings.")
    model_available = False

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Class for generating embeddings for resumes and job postings."""
    
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        """Initialize the embedding generator with a Sentence-BERT model.
        
        Args:
            model_name: Name of the Sentence-BERT model to use
        """
        self.model_name = model_name
        self.model = None
        
        if model_available:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {str(e)}")
                logger.warning("Using dummy embeddings instead.")
        else:
            logger.warning("Using dummy embeddings.")
    
    def generate_resume_embeddings(self, resumes: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for resumes.
        
        Args:
            resumes: List of parsed resume dictionaries
            
        Returns:
            Dictionary mapping resume IDs to embeddings
        """
        resume_embeddings = {}
        
        for resume in resumes:
            try:
                # Generate a unique ID for the resume
                resume_id = resume.get("file_path", str(id(resume)))
                
                # Extract relevant text for embedding
                embedding_text = self._extract_resume_text_for_embedding(resume)
                
                # Generate embedding
                embedding = self._generate_embedding(embedding_text)
                
                resume_embeddings[resume_id] = embedding
                logger.debug(f"Generated embedding for resume: {resume.get('file_name', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Error generating embedding for resume {resume.get('file_name', 'Unknown')}: {str(e)}")
        
        logger.info(f"Generated embeddings for {len(resume_embeddings)} resumes")
        return resume_embeddings
    
    def generate_job_embeddings(self, jobs: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate embeddings for job postings.
        
        Args:
            jobs: List of parsed job dictionaries
            
        Returns:
            Dictionary mapping job IDs to embeddings
        """
        job_embeddings = {}
        
        for job in jobs:
            try:
                # Generate a unique ID for the job
                job_id = str(id(job))
                
                # Extract relevant text for embedding
                embedding_text = self._extract_job_text_for_embedding(job)
                
                # Generate embedding
                embedding = self._generate_embedding(embedding_text)
                
                job_embeddings[job_id] = embedding
                logger.debug(f"Generated embedding for job: {job.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Error generating embedding for job {job.get('title', 'Unknown')}: {str(e)}")
        
        logger.info(f"Generated embeddings for {len(job_embeddings)} jobs")
        return job_embeddings
    
    def _extract_resume_text_for_embedding(self, resume: Dict[str, Any]) -> str:
        """Extract relevant text from a resume for embedding.
        
        Args:
            resume: Parsed resume dictionary
            
        Returns:
            Text to use for generating the embedding
        """
        text_parts = []
        
        # Add skills
        if "skills" in resume and resume["skills"]:
            text_parts.append("Skills: " + ", ".join(resume["skills"]))
        
        # Add experience descriptions
        if "experience" in resume and resume["experience"]:
            for exp in resume["experience"]:
                if "title" in exp:
                    text_parts.append(f"Experience: {exp['title']}")
                if "description" in exp:
                    text_parts.append(exp["description"])
        
        # Add project descriptions
        if "projects" in resume and resume["projects"]:
            for proj in resume["projects"]:
                if "name" in proj:
                    text_parts.append(f"Project: {proj['name']}")
                if "description" in proj:
                    text_parts.append(proj["description"])
        
        # Add education
        if "education" in resume and resume["education"]:
            for edu in resume["education"]:
                if "degree" in edu and "institution" in edu:
                    text_parts.append(f"Education: {edu['degree']} at {edu['institution']}")
        
        # If no structured data, use the raw content
        if not text_parts and "content" in resume:
            return resume["content"][:1000]  # Limit to first 1000 chars for efficiency
        
        return "\n".join(text_parts)
    
    def _extract_job_text_for_embedding(self, job: Dict[str, Any]) -> str:
        """Extract relevant text from a job posting for embedding.
        
        Args:
            job: Parsed job dictionary
            
        Returns:
            Text to use for generating the embedding
        """
        text_parts = []
        
        # Add title
        if "title" in job:
            text_parts.append(f"Job Title: {job['title']}")
        
        # Add required skills
        if "required_skills" in job and job["required_skills"]:
            if isinstance(job["required_skills"], list):
                text_parts.append("Required Skills: " + ", ".join(job["required_skills"]))
            elif isinstance(job["required_skills"], str):
                text_parts.append(f"Required Skills: {job['required_skills']}")
        
        # Add extracted skills
        if "extracted_skills" in job and job["extracted_skills"]:
            text_parts.append("Skills: " + ", ".join(job["extracted_skills"]))
        
        # Add requirements
        if "requirements" in job and job["requirements"]:
            text_parts.append("Requirements: " + "\n".join(job["requirements"]))
        
        # Add responsibilities
        if "responsibilities" in job and job["responsibilities"]:
            text_parts.append("Responsibilities: " + "\n".join(job["responsibilities"]))
        
        # If no structured data, use the raw description
        if not text_parts and "job_description" in job:
            return job["job_description"]
        
        return "\n".join(text_parts)
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate an embedding for the given text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is not None:
            try:
                # Generate embedding using Sentence-BERT
                embedding = self.model.encode(text, convert_to_numpy=True)
                return embedding
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
        
        # Fallback to dummy embedding if model not available or error occurs
        return self._generate_dummy_embedding()
    
    def _generate_dummy_embedding(self, dim: int = 384) -> np.ndarray:
        """Generate a dummy embedding for testing purposes.
        
        Args:
            dim: Dimension of the embedding vector
            
        Returns:
            Random embedding vector as numpy array
        """
        # Generate a random vector and normalize it
        embedding = np.random.randn(dim)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with sample data
    sample_resume = {
        "file_name": "sample_resume.txt",
        "skills": ["Python", "Machine Learning", "SQL", "Data Analysis"],
        "experience": [
            {
                "title": "Data Scientist",
                "company": "Tech Solutions Inc.",
                "description": "Developed machine learning models for customer segmentation."
            }
        ],
        "projects": [
            {
                "name": "Sentiment Analysis Tool",
                "description": "Developed a sentiment analysis tool using BERT."
            }
        ]
    }
    
    sample_job = {
        "title": "Data Scientist",
        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
        "job_description": "We are looking for a Data Scientist to join our team.",
        "requirements": ["3+ years of experience in data science", "Strong programming skills in Python"]
    }
    
    # Initialize embedding generator
    embedding_generator = EmbeddingGenerator()
    
    # Generate embeddings
    resume_embedding = embedding_generator.generate_resume_embeddings([sample_resume])
    job_embedding = embedding_generator.generate_job_embeddings([sample_job])
    
    # Print embedding shapes
    for resume_id, embedding in resume_embedding.items():
        print(f"Resume embedding shape: {embedding.shape}")
    
    for job_id, embedding in job_embedding.items():
        print(f"Job embedding shape: {embedding.shape}")