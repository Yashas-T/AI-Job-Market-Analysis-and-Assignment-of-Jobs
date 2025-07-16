#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Job Parser Module for Resume-Job Matching System.

This module provides functionality to parse job postings and extract structured information
such as required skills, job description, and location.
"""

import re
import logging
from typing import Dict, List, Any, Optional

# For NLP processing
try:
    import spacy
    from spacy.matcher import Matcher
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
except ImportError:
    logging.warning("spaCy or en_core_web_md model not found. Using basic parsing.")
    nlp = None

logger = logging.getLogger(__name__)


class JobParser:
    """Class for parsing job postings and extracting structured information."""
    
    def __init__(self):
        """Initialize the job parser."""
        self.matcher = None
        if nlp:
            # Initialize spaCy matcher
            self.matcher = Matcher(nlp.vocab)
            
            # Add patterns for skills, requirements, etc.
            self._add_patterns()
        
        logger.info("Initialized JobParser")
    
    def _add_patterns(self):
        """Add patterns to the spaCy matcher for identifying job sections."""
        # Pattern for requirements section headers
        requirements_pattern = [{"LOWER": {"IN": ["requirements", "qualifications", "skills", "required"]}}]
        self.matcher.add("REQUIREMENTS", [requirements_pattern])
        
        # Pattern for responsibilities section headers
        responsibilities_pattern = [{"LOWER": {"IN": ["responsibilities", "duties", "what you'll do", "what you will do", "role"]}}]
        self.matcher.add("RESPONSIBILITIES", [responsibilities_pattern])
        
        # Pattern for benefits section headers
        benefits_pattern = [{"LOWER": {"IN": ["benefits", "perks", "what we offer", "compensation"]}}]
        self.matcher.add("BENEFITS", [benefits_pattern])
    
    def parse_jobs(self, jobs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse a list of job postings and extract structured information.
        
        Args:
            jobs: List of job dictionaries
            
        Returns:
            List of parsed job dictionaries with additional structured information
        """
        parsed_jobs = []
        
        for job in jobs:
            try:
                parsed_job = self.parse_job(job)
                parsed_jobs.append(parsed_job)
                logger.debug(f"Successfully parsed job: {job.get('title', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error parsing job {job.get('title', 'Unknown')}: {str(e)}")
                # Add the original job to maintain the list structure
                parsed_jobs.append(job)
        
        logger.info(f"Successfully parsed {len(parsed_jobs)} jobs")
        return parsed_jobs
    
    def parse_job(self, job: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single job posting and extract structured information.
        
        Args:
            job: Job dictionary
            
        Returns:
            Parsed job dictionary with additional structured information
        """
        # Create a copy of the job to avoid modifying the original
        parsed_job = job.copy()
        
        # Get the job description
        description = job.get("job_description", "")
        
        # Extract information
        parsed_job["extracted_skills"] = self._extract_skills(description, job.get("required_skills", []))
        parsed_job["requirements"] = self._extract_requirements(description)
        parsed_job["responsibilities"] = self._extract_responsibilities(description)
        parsed_job["benefits"] = self._extract_benefits(description)
        parsed_job["normalized_location"] = self._normalize_location(job.get("location", ""))
        
        # Extract salary range if not already present
        if "salary_min" not in job or "salary_max" not in job:
            salary_min, salary_max = self._extract_salary_range(job.get("salary", ""))
            parsed_job["salary_min"] = salary_min
            parsed_job["salary_max"] = salary_max
        
        return parsed_job
    
    def _extract_skills(self, description: str, required_skills: List[str] = None) -> List[str]:
        """Extract skills from the job description.
        
        Args:
            description: Job description text
            required_skills: List of required skills if already provided
            
        Returns:
            List of extracted skills
        """
        skills = []
        
        # If required_skills is already a list, use it
        if required_skills and isinstance(required_skills, list):
            skills.extend(required_skills)
        
        # Common technical skills to look for
        common_skills = [
            "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift", "kotlin",
            "sql", "mysql", "postgresql", "mongodb", "oracle", "nosql", "firebase",
            "html", "css", "react", "angular", "vue", "node", "express", "django", "flask",
            "spring", "hibernate", "asp.net", "laravel", "ruby on rails",
            "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy",
            "matplotlib", "seaborn", "tableau", "power bi", "excel", "spss", "sas", "r",
            "machine learning", "deep learning", "nlp", "computer vision", "data science",
            "data analysis", "data visualization", "statistics", "big data", "hadoop", "spark",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "git", "github", "gitlab",
            "ci/cd", "agile", "scrum", "kanban", "jira", "confluence", "slack", "trello"
        ]
        
        # Extract requirements section
        requirements_section = self._extract_section(description, "REQUIREMENTS")
        
        # If no requirements section found, use the entire description
        text_to_search = requirements_section if requirements_section else description
        
        # Use spaCy for better extraction if available
        if nlp:
            doc = nlp(text_to_search)
            for token in doc:
                if token.text.lower() in common_skills and token.text.lower() not in [s.lower() for s in skills]:
                    skills.append(token.text)
        
        # Fallback to regex pattern matching
        for skill in common_skills:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_to_search, re.IGNORECASE):
                if skill.lower() not in [s.lower() for s in skills]:
                    skills.append(skill)
        
        return skills
    
    def _extract_requirements(self, description: str) -> List[str]:
        """Extract requirements from the job description.
        
        Args:
            description: Job description text
            
        Returns:
            List of requirements
        """
        requirements = []
        
        # Extract requirements section
        requirements_section = self._extract_section(description, "REQUIREMENTS")
        
        if not requirements_section:
            return requirements
        
        # Split into lines and look for bullet points
        lines = requirements_section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or re.match(r'^\d+\.\s', line):
                requirements.append(line)
        
        # If no bullet points found, try to split by sentences
        if not requirements and nlp:
            doc = nlp(requirements_section)
            for sent in doc.sents:
                if len(sent.text.strip()) > 10:  # Ignore very short sentences
                    requirements.append(sent.text.strip())
        
        return requirements
    
    def _extract_responsibilities(self, description: str) -> List[str]:
        """Extract responsibilities from the job description.
        
        Args:
            description: Job description text
            
        Returns:
            List of responsibilities
        """
        responsibilities = []
        
        # Extract responsibilities section
        responsibilities_section = self._extract_section(description, "RESPONSIBILITIES")
        
        if not responsibilities_section:
            return responsibilities
        
        # Split into lines and look for bullet points
        lines = responsibilities_section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or re.match(r'^\d+\.\s', line):
                responsibilities.append(line)
        
        # If no bullet points found, try to split by sentences
        if not responsibilities and nlp:
            doc = nlp(responsibilities_section)
            for sent in doc.sents:
                if len(sent.text.strip()) > 10:  # Ignore very short sentences
                    responsibilities.append(sent.text.strip())
        
        return responsibilities
    
    def _extract_benefits(self, description: str) -> List[str]:
        """Extract benefits from the job description.
        
        Args:
            description: Job description text
            
        Returns:
            List of benefits
        """
        benefits = []
        
        # Extract benefits section
        benefits_section = self._extract_section(description, "BENEFITS")
        
        if not benefits_section:
            return benefits
        
        # Split into lines and look for bullet points
        lines = benefits_section.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or re.match(r'^\d+\.\s', line):
                benefits.append(line)
        
        # If no bullet points found, try to split by sentences
        if not benefits and nlp:
            doc = nlp(benefits_section)
            for sent in doc.sents:
                if len(sent.text.strip()) > 10:  # Ignore very short sentences
                    benefits.append(sent.text.strip())
        
        return benefits
    
    def _normalize_location(self, location: str) -> str:
        """Normalize location string for better matching.
        
        Args:
            location: Location string (e.g., "New York, NY")
            
        Returns:
            Normalized location string
        """
        if not location:
            return ""
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', location).strip()
        
        # Extract city and state for US locations
        us_pattern = r'([A-Za-z\s]+),\s*([A-Z]{2})'
        match = re.match(us_pattern, normalized)
        if match:
            city = match.group(1).strip()
            state = match.group(2)
            return f"{city}, {state}"
        
        return normalized
    
    def _extract_salary_range(self, salary: str) -> tuple:
        """Extract minimum and maximum salary from salary string.
        
        Args:
            salary: Salary string (e.g., "$120,000 - $150,000")
            
        Returns:
            Tuple of (min_salary, max_salary) as integers
        """
        if not salary:
            return (None, None)
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[\$,]', '', salary)
        
        # Look for range pattern
        range_pattern = r'(\d+)\s*-\s*(\d+)'
        match = re.search(range_pattern, cleaned)
        
        if match:
            min_salary = int(match.group(1))
            max_salary = int(match.group(2))
            return (min_salary, max_salary)
        
        # Look for single number pattern
        single_pattern = r'(\d+)'
        match = re.search(single_pattern, cleaned)
        
        if match:
            salary_value = int(match.group(1))
            return (salary_value, salary_value)
        
        return (None, None)
    
    def _extract_section(self, text: str, section_type: str) -> str:
        """Extract a specific section from the job description.
        
        Args:
            text: Job description text
            section_type: Type of section to extract (REQUIREMENTS, RESPONSIBILITIES, BENEFITS)
            
        Returns:
            Extracted section text or empty string if not found
        """
        # If spaCy is available, use the matcher
        if nlp and self.matcher:
            doc = nlp(text)
            matches = self.matcher(doc)
            
            for match_id, start, end in matches:
                if nlp.vocab.strings[match_id] == section_type:
                    # Found the section header, now extract the content until the next section
                    section_start = doc[start:end].start_char
                    
                    # Find the next section header
                    next_section_start = len(text)
                    for next_match_id, next_start, next_end in matches:
                        if doc[next_start:next_end].start_char > section_start:
                            next_section_start = doc[next_start:next_end].start_char
                            break
                    
                    # Extract the section content
                    section_header_end = text.find('\n', section_start)
                    if section_header_end == -1:
                        section_header_end = section_start + len(doc[start:end].text)
                    
                    return text[section_header_end:next_section_start].strip()
        
        # Fallback to regex pattern matching
        section_patterns = {
            "REQUIREMENTS": r'(?i)\bREQUIREMENTS\b|\bQUALIFICATIONS\b|\bSKILLS\s+REQUIRED\b',
            "RESPONSIBILITIES": r'(?i)\bRESPONSIBILITIES\b|\bDUTIES\b|\bWHAT\s+YOU\'LL\s+DO\b',
            "BENEFITS": r'(?i)\bBENEFITS\b|\bPERKS\b|\bWHAT\s+WE\s+OFFER\b'
        }
        
        pattern = section_patterns.get(section_type)
        if not pattern:
            return ""
        
        match = re.search(pattern, text)
        if not match:
            return ""
        
        section_start = match.start()
        
        # Find the next section
        next_section_start = len(text)
        for next_pattern in section_patterns.values():
            next_match = re.search(next_pattern, text[section_start + 1:])
            if next_match:
                next_start = section_start + 1 + next_match.start()
                if next_start < next_section_start:
                    next_section_start = next_start
        
        # Extract the section content
        section_header_end = text.find('\n', section_start)
        if section_header_end == -1:
            section_header_end = section_start + len(match.group(0))
        
        return text[section_header_end:next_section_start].strip()


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with a sample job
    sample_job = {
        "title": "Data Scientist",
        "location": "New York, NY",
        "required_skills": ["Python", "SQL", "Machine Learning", "Statistics"],
        "job_description": """We are looking for a Data Scientist to join our team.
        
        Requirements:
        - 3+ years of experience in data science or related field
        - Strong programming skills in Python and SQL
        - Experience with machine learning algorithms and statistical analysis
        - Excellent communication skills
        
        Responsibilities:
        - Develop and implement machine learning models
        - Analyze large datasets to extract insights
        - Collaborate with cross-functional teams
        - Present findings to stakeholders
        
        Benefits:
        - Competitive salary
        - Health insurance
        - 401(k) matching
        - Flexible work hours
        """,
        "salary": "$120,000 - $150,000"
    }
    
    parser = JobParser()
    parsed_job = parser.parse_job(sample_job)
    
    print("\nExtracted Information:")
    print(f"Title: {parsed_job['title']}")
    print(f"Location: {parsed_job['location']} (Normalized: {parsed_job['normalized_location']})")
    print(f"Skills: {', '.join(parsed_job['extracted_skills'])}")
    print(f"Salary Range: ${parsed_job['salary_min']} - ${parsed_job['salary_max']}")
    
    print("\nRequirements:")
    for req in parsed_job['requirements']:
        print(f"- {req}")
    
    print("\nResponsibilities:")
    for resp in parsed_job['responsibilities']:
        print(f"- {resp}")
    
    print("\nBenefits:")
    for benefit in parsed_job['benefits']:
        print(f"- {benefit}")