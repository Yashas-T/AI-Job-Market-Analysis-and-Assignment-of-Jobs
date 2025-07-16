#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Recommendation Module for Resume-Job Matching System.

This module provides functionality to generate personalized job recommendations
based on resume-job matching results.
"""

import logging
from typing import Dict, List, Any, Tuple
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """Class for generating personalized job recommendations."""
    
    def __init__(self, matcher_results=None):
        """Initialize the recommendation engine.
        
        Args:
            matcher_results: Optional dictionary of matcher results
        """
        self.matcher_results = matcher_results or {}
        logger.info("Initialized RecommendationEngine")
    
    def set_matcher_results(self, matcher_results):
        """Set the matcher results.
        
        Args:
            matcher_results: Dictionary of matcher results
        """
        self.matcher_results = matcher_results
        logger.info("Updated matcher results in RecommendationEngine")
    
    def generate_recommendations(self, resume_id, top_n=5):
        """Generate job recommendations for a specific resume.
        
        Args:
            resume_id: ID of the resume to generate recommendations for
            top_n: Number of top recommendations to return
            
        Returns:
            List of recommended jobs with scores and explanations
        """
        if resume_id not in self.matcher_results:
            logger.warning(f"No matcher results found for resume ID: {resume_id}")
            return []
        
        # Get the matcher results for this resume
        matches = self.matcher_results[resume_id]
        
        # Sort matches by total score in descending order
        sorted_matches = sorted(matches, key=lambda x: x["total_score"], reverse=True)
        
        # Take the top N matches
        top_matches = sorted_matches[:top_n]
        
        # Generate recommendations with explanations
        recommendations = []
        for match in top_matches:
            recommendation = self._create_recommendation(match)
            recommendations.append(recommendation)
        
        logger.info(f"Generated {len(recommendations)} recommendations for resume ID: {resume_id}")
        return recommendations
    
    def generate_all_recommendations(self, top_n=5):
        """Generate job recommendations for all resumes.
        
        Args:
            top_n: Number of top recommendations to return per resume
            
        Returns:
            Dictionary mapping resume IDs to lists of recommendations
        """
        all_recommendations = {}
        
        for resume_id in self.matcher_results:
            recommendations = self.generate_recommendations(resume_id, top_n)
            all_recommendations[resume_id] = recommendations
        
        logger.info(f"Generated recommendations for {len(all_recommendations)} resumes")
        return all_recommendations
    
    def _create_recommendation(self, match):
        """Create a recommendation object from a match.
        
        Args:
            match: Match dictionary from matcher
            
        Returns:
            Recommendation dictionary with job details and explanation
        """
        job = match["job"]
        
        # Create explanation based on match scores and skills
        explanation = self._generate_explanation(match)
        
        # Create recommendation object
        recommendation = {
            "job": job,
            "score": match["total_score"],
            "explanation": explanation,
            "matching_skills": match["matching_skills"],
            "missing_skills": match["missing_skills"],
            "skill_score": match["skill_score"],
            "location_score": match["location_score"],
            "embedding_score": match["embedding_score"]
        }
        
        return recommendation
    
    def _generate_explanation(self, match):
        """Generate a human-readable explanation for a recommendation.
        
        Args:
            match: Match dictionary from matcher
            
        Returns:
            String explanation of why this job is recommended
        """
        job = match["job"]
        job_title = job.get("title", "This job")
        
        # Start with a base explanation
        explanation = f"{job_title} is a good match for your profile. "
        
        # Add skill match explanation
        matching_skills = match["matching_skills"]
        if matching_skills:
            if len(matching_skills) > 3:
                skill_text = f"{', '.join(matching_skills[:3])} and {len(matching_skills) - 3} more"
            else:
                skill_text = f"{', '.join(matching_skills)}"
            
            explanation += f"You have {len(matching_skills)} matching skills including {skill_text}. "
        
        # Add missing skills explanation if relevant
        missing_skills = match["missing_skills"]
        if missing_skills and len(missing_skills) <= 3:
            explanation += f"Consider developing skills in {', '.join(missing_skills)} to improve your match. "
        elif missing_skills:
            explanation += f"Consider developing additional skills to improve your match. "
        
        # Add location explanation if it's a good match
        if match["location_score"] > 0.7:
            explanation += "The job location matches your preferred location. "
        
        # Add overall match quality
        score = match["total_score"]
        if score > 0.8:
            explanation += "Overall, this is an excellent match for your profile."
        elif score > 0.6:
            explanation += "Overall, this is a good match for your profile."
        else:
            explanation += "This job partially matches your profile."
        
        return explanation
    
    def get_skill_gap_analysis(self, resume_id):
        """Generate a skill gap analysis for a resume.
        
        Args:
            resume_id: ID of the resume to analyze
            
        Returns:
            Dictionary with skill gap analysis
        """
        if resume_id not in self.matcher_results:
            logger.warning(f"No matcher results found for resume ID: {resume_id}")
            return {}
        
        # Get the matcher results for this resume
        matches = self.matcher_results[resume_id]
        
        # Collect all missing skills across all jobs
        all_missing_skills = defaultdict(int)
        for match in matches:
            for skill in match["missing_skills"]:
                all_missing_skills[skill] += 1
        
        # Sort missing skills by frequency
        sorted_missing_skills = sorted(all_missing_skills.items(), key=lambda x: x[1], reverse=True)
        
        # Create skill gap analysis
        skill_gap = {
            "missing_skills": [skill for skill, count in sorted_missing_skills],
            "skill_frequency": {skill: count for skill, count in sorted_missing_skills},
            "total_jobs_analyzed": len(matches)
        }
        
        logger.info(f"Generated skill gap analysis for resume ID: {resume_id}")
        return skill_gap


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create sample matcher results
    sample_job1 = {
        "title": "Data Scientist",
        "company": "ABC Corp",
        "location": "New York, NY",
        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
        "job_description": "We are looking for a Data Scientist to join our team."
    }
    
    sample_job2 = {
        "title": "Machine Learning Engineer",
        "company": "XYZ Inc",
        "location": "San Francisco, CA",
        "required_skills": ["Python", "TensorFlow", "Deep Learning", "Computer Vision"],
        "job_description": "Join our ML team to build cutting-edge models."
    }
    
    sample_match1 = {
        "job": sample_job1,
        "total_score": 0.85,
        "embedding_score": 0.9,
        "skill_score": 0.8,
        "location_score": 0.7,
        "matching_skills": ["Python", "SQL", "Machine Learning"],
        "missing_skills": ["Statistics"]
    }
    
    sample_match2 = {
        "job": sample_job2,
        "total_score": 0.75,
        "embedding_score": 0.8,
        "skill_score": 0.6,
        "location_score": 0.3,
        "matching_skills": ["Python"],
        "missing_skills": ["TensorFlow", "Deep Learning", "Computer Vision"]
    }
    
    sample_matcher_results = {
        "sample_resume.txt": [sample_match1, sample_match2]
    }
    
    # Initialize recommendation engine
    recommendation_engine = RecommendationEngine(sample_matcher_results)
    
    # Generate recommendations
    recommendations = recommendation_engine.generate_recommendations("sample_resume.txt", top_n=2)
    
    # Print recommendations
    print("\nRecommendations:")
    for i, rec in enumerate(recommendations):
        print(f"\nRecommendation {i+1}:")
        print(f"Job: {rec['job']['title']} at {rec['job']['company']}")
        print(f"Score: {rec['score']:.2f}")
        print(f"Explanation: {rec['explanation']}")
        print(f"Matching Skills: {', '.join(rec['matching_skills'])}")
        print(f"Missing Skills: {', '.join(rec['missing_skills'])}")
    
    # Generate skill gap analysis
    skill_gap = recommendation_engine.get_skill_gap_analysis("sample_resume.txt")
    
    print("\nSkill Gap Analysis:")
    print(f"Missing Skills (in order of importance): {', '.join(skill_gap['missing_skills'])}")
    print(f"Total Jobs Analyzed: {skill_gap['total_jobs_analyzed']}")