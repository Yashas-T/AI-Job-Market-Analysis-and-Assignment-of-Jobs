#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matcher Module for Resume-Job Matching System.

This module provides functionality to match resumes with job postings
using embedding similarity and other matching techniques.
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class Matcher:
    """Class for matching resumes with job postings."""
    
    def __init__(self):
        """Initialize the matcher."""
        logger.info("Initialized Matcher")
    
    def match_resumes_with_jobs(
        self,
        resumes: List[Dict[str, Any]],
        jobs: List[Dict[str, Any]],
        resume_embeddings: Dict[str, np.ndarray],
        job_embeddings: Dict[str, np.ndarray]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Match resumes with job postings.
        
        Args:
            resumes: List of parsed resume dictionaries
            jobs: List of parsed job dictionaries
            resume_embeddings: Dictionary mapping resume IDs to embeddings
            job_embeddings: Dictionary mapping job IDs to embeddings
            
        Returns:
            Dictionary mapping resume IDs to lists of job matches with scores
        """
        matches = {}
        
        for resume in resumes:
            try:
                # Generate a unique ID for the resume
                resume_id = resume.get("file_path", str(id(resume)))
                
                # Get the resume embedding
                resume_embedding = resume_embeddings.get(resume_id)
                if resume_embedding is None:
                    logger.warning(f"No embedding found for resume: {resume.get('file_name', 'Unknown')}")
                    continue
                
                # Match the resume with all jobs
                resume_matches = self._match_resume_with_jobs(resume, jobs, resume_embedding, job_embeddings)
                
                # Sort matches by score in descending order
                resume_matches.sort(key=lambda x: x["total_score"], reverse=True)
                
                matches[resume_id] = resume_matches
                logger.debug(f"Matched resume {resume.get('file_name', 'Unknown')} with {len(resume_matches)} jobs")
                
            except Exception as e:
                logger.error(f"Error matching resume {resume.get('file_name', 'Unknown')}: {str(e)}")
        
        logger.info(f"Matched {len(matches)} resumes with jobs")
        return matches
    
    def _match_resume_with_jobs(
        self,
        resume: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        resume_embedding: np.ndarray,
        job_embeddings: Dict[str, np.ndarray]
    ) -> List[Dict[str, Any]]:
        """Match a single resume with all job postings.
        
        Args:
            resume: Parsed resume dictionary
            jobs: List of parsed job dictionaries
            resume_embedding: Embedding vector for the resume
            job_embeddings: Dictionary mapping job IDs to embeddings
            
        Returns:
            List of job matches with scores
        """
        matches = []
        
        for job in jobs:
            try:
                # Generate a unique ID for the job
                job_id = str(id(job))
                
                # Get the job embedding
                job_embedding = job_embeddings.get(job_id)
                if job_embedding is None:
                    logger.warning(f"No embedding found for job: {job.get('title', 'Unknown')}")
                    continue
                
                # Calculate embedding similarity score
                embedding_score = self._calculate_embedding_similarity(resume_embedding, job_embedding)
                
                # Calculate skill match score
                skill_score = self._calculate_skill_match_score(resume, job)
                
                # Calculate location match score
                location_score = self._calculate_location_match_score(resume, job)
                
                # Calculate total score (weighted average)
                total_score = (
                    0.5 * embedding_score +  # 50% weight for embedding similarity
                    0.4 * skill_score +      # 40% weight for skill match
                    0.1 * location_score     # 10% weight for location match
                )
                
                # Create match object
                match = {
                    "job": job,
                    "embedding_score": float(embedding_score),  # Convert to Python float for JSON serialization
                    "skill_score": float(skill_score),
                    "location_score": float(location_score),
                    "total_score": float(total_score),
                    "matching_skills": self._get_matching_skills(resume, job),
                    "missing_skills": self._get_missing_skills(resume, job)
                }
                
                matches.append(match)
                
            except Exception as e:
                logger.error(f"Error matching resume with job {job.get('title', 'Unknown')}: {str(e)}")
        
        return matches
    
    def _calculate_embedding_similarity(self, resume_embedding: np.ndarray, job_embedding: np.ndarray) -> float:
        """Calculate cosine similarity between resume and job embeddings.
        
        Args:
            resume_embedding: Embedding vector for the resume
            job_embedding: Embedding vector for the job
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Calculate cosine similarity
        dot_product = np.dot(resume_embedding, job_embedding)
        norm_resume = np.linalg.norm(resume_embedding)
        norm_job = np.linalg.norm(job_embedding)
        
        if norm_resume == 0 or norm_job == 0:
            return 0.0
        
        similarity = dot_product / (norm_resume * norm_job)
        
        # Ensure the similarity is between 0 and 1
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def _calculate_skill_match_score(self, resume: Dict[str, Any], job: Dict[str, Any]) -> float:
        """Calculate skill match score between resume and job.
        
        Args:
            resume: Parsed resume dictionary
            job: Parsed job dictionary
            
        Returns:
            Skill match score (0-1)
        """
        # Get resume skills
        resume_skills = set()
        if "skills" in resume and resume["skills"]:
            resume_skills = {skill.lower() for skill in resume["skills"]}
        
        # Get job skills
        job_skills = set()
        
        # First try required_skills
        if "required_skills" in job and job["required_skills"]:
            if isinstance(job["required_skills"], list):
                job_skills.update({skill.lower() for skill in job["required_skills"]})
            elif isinstance(job["required_skills"], str):
                job_skills.add(job["required_skills"].lower())
        
        # Then try extracted_skills
        if "extracted_skills" in job and job["extracted_skills"]:
            job_skills.update({skill.lower() for skill in job["extracted_skills"]})
        
        # If no skills found, return 0
        if not resume_skills or not job_skills:
            return 0.0
        
        # Calculate Jaccard similarity
        matching_skills = resume_skills.intersection(job_skills)
        all_skills = resume_skills.union(job_skills)
        
        if not all_skills:
            return 0.0
        
        return len(matching_skills) / len(all_skills)
    
    def _calculate_location_match_score(self, resume: Dict[str, Any], job: Dict[str, Any]) -> float:
        """Calculate location match score between resume and job.
        
        Args:
            resume: Parsed resume dictionary
            job: Parsed job dictionary
            
        Returns:
            Location match score (0-1)
        """
        # Get resume location
        resume_location = resume.get("location", "").lower()
        
        # Get job location
        job_location = ""
        if "normalized_location" in job and job["normalized_location"]:
            job_location = job["normalized_location"].lower()
        elif "location" in job and job["location"]:
            job_location = job["location"].lower()
        
        # If either location is missing, return 0.5 (neutral score)
        if not resume_location or not job_location:
            return 0.5
        
        # Check for exact match
        if resume_location == job_location:
            return 1.0
        
        # Check for partial match (e.g., same city or state)
        resume_parts = set(part.strip() for part in resume_location.replace(",", " ").split())
        job_parts = set(part.strip() for part in job_location.replace(",", " ").split())
        
        common_parts = resume_parts.intersection(job_parts)
        
        if common_parts:
            return 0.8  # High score for partial match
        
        # No match
        return 0.0
    
    def _get_matching_skills(self, resume: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
        """Get list of skills that match between resume and job.
        
        Args:
            resume: Parsed resume dictionary
            job: Parsed job dictionary
            
        Returns:
            List of matching skills
        """
        # Get resume skills
        resume_skills = set()
        if "skills" in resume and resume["skills"]:
            resume_skills = {skill.lower(): skill for skill in resume["skills"]}
        
        # Get job skills
        job_skills = set()
        
        # First try required_skills
        if "required_skills" in job and job["required_skills"]:
            if isinstance(job["required_skills"], list):
                for skill in job["required_skills"]:
                    job_skills.add(skill.lower())
            elif isinstance(job["required_skills"], str):
                job_skills.add(job["required_skills"].lower())
        
        # Then try extracted_skills
        if "extracted_skills" in job and job["extracted_skills"]:
            for skill in job["extracted_skills"]:
                job_skills.add(skill.lower())
        
        # Find matching skills
        matching_skills = []
        for skill in resume_skills.keys():
            if skill in job_skills:
                matching_skills.append(resume_skills[skill])  # Use original case from resume
        
        return matching_skills
    
    def _get_missing_skills(self, resume: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
        """Get list of skills required by the job but missing from the resume.
        
        Args:
            resume: Parsed resume dictionary
            job: Parsed job dictionary
            
        Returns:
            List of missing skills
        """
        # Get resume skills
        resume_skills = set()
        if "skills" in resume and resume["skills"]:
            resume_skills = {skill.lower() for skill in resume["skills"]}
        
        # Get job skills with original case
        job_skills = {}
        
        # First try required_skills
        if "required_skills" in job and job["required_skills"]:
            if isinstance(job["required_skills"], list):
                for skill in job["required_skills"]:
                    job_skills[skill.lower()] = skill
            elif isinstance(job["required_skills"], str):
                job_skills[job["required_skills"].lower()] = job["required_skills"]
        
        # Then try extracted_skills
        if "extracted_skills" in job and job["extracted_skills"]:
            for skill in job["extracted_skills"]:
                job_skills[skill.lower()] = skill
        
        # Find missing skills
        missing_skills = []
        for skill_lower, skill_original in job_skills.items():
            if skill_lower not in resume_skills:
                missing_skills.append(skill_original)  # Use original case from job
        
        return missing_skills


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with sample data
    sample_resume = {
        "file_path": "sample_resume.txt",
        "file_name": "sample_resume.txt",
        "skills": ["Python", "Machine Learning", "SQL", "Data Analysis"],
        "location": "New York, NY"
    }
    
    sample_job = {
        "title": "Data Scientist",
        "location": "New York, NY",
        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
        "job_description": "We are looking for a Data Scientist to join our team."
    }
    
    # Create dummy embeddings
    resume_embedding = np.random.randn(384)
    resume_embedding = resume_embedding / np.linalg.norm(resume_embedding)
    
    job_embedding = np.random.randn(384)
    job_embedding = job_embedding / np.linalg.norm(job_embedding)
    
    # Create embeddings dictionaries
    resume_embeddings = {sample_resume["file_path"]: resume_embedding}
    job_embeddings = {str(id(sample_job)): job_embedding}
    
    # Initialize matcher
    matcher = Matcher()
    
    # Match resume with job
    matches = matcher.match_resumes_with_jobs([sample_resume], [sample_job], resume_embeddings, job_embeddings)
    
    # Print match results
    for resume_id, job_matches in matches.items():
        print(f"\nMatches for resume: {resume_id}")
        for i, match in enumerate(job_matches):
            print(f"\nMatch {i+1}:")
            print(f"Job: {match['job']['title']}")
            print(f"Total Score: {match['total_score']:.2f}")
            print(f"Embedding Score: {match['embedding_score']:.2f}")
            print(f"Skill Score: {match['skill_score']:.2f}")
            print(f"Location Score: {match['location_score']:.2f}")
            print(f"Matching Skills: {', '.join(match['matching_skills'])}")
            print(f"Missing Skills: {', '.join(match['missing_skills'])}")