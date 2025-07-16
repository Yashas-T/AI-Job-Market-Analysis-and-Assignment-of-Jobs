#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Resume Parser Module for Resume-Job Matching System.

This module provides functionality to parse resume text and extract structured information
such as skills, education, experience, projects, and location.
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


class ResumeParser:
    """Class for parsing resume text and extracting structured information."""
    
    def __init__(self):
        """Initialize the resume parser."""
        self.matcher = None
        if nlp:
            # Initialize spaCy matcher
            self.matcher = Matcher(nlp.vocab)
            
            # Add patterns for skills, education, experience, etc.
            self._add_patterns()
        
        logger.info("Initialized ResumeParser")
    
    def _add_patterns(self):
        """Add patterns to the spaCy matcher for identifying resume sections."""
        # Pattern for education section headers
        education_pattern = [{"LOWER": {"IN": ["education", "academic", "academics", "degree", "degrees", "qualification", "qualifications"]}}]
        self.matcher.add("EDUCATION", [education_pattern])
        
        # Pattern for experience section headers
        experience_pattern = [{"LOWER": {"IN": ["experience", "employment", "work", "history", "professional"]}}]
        self.matcher.add("EXPERIENCE", [experience_pattern])
        
        # Pattern for skills section headers
        skills_pattern = [{"LOWER": {"IN": ["skills", "abilities", "competencies", "expertise"]}}]
        self.matcher.add("SKILLS", [skills_pattern])
        
        # Pattern for projects section headers
        projects_pattern = [{"LOWER": {"IN": ["projects", "project", "portfolio", "works"]}}]
        self.matcher.add("PROJECTS", [projects_pattern])
    
    def parse_resumes(self, resumes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse a list of resumes and extract structured information.
        
        Args:
            resumes: List of resume dictionaries with 'content' key containing raw text
            
        Returns:
            List of parsed resume dictionaries with additional structured information
        """
        parsed_resumes = []
        
        for resume in resumes:
            try:
                parsed_resume = self.parse_resume(resume)
                parsed_resumes.append(parsed_resume)
                logger.debug(f"Successfully parsed resume: {resume.get('file_name', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error parsing resume {resume.get('file_name', 'Unknown')}: {str(e)}")
                # Add the original resume to maintain the list structure
                parsed_resumes.append(resume)
        
        logger.info(f"Successfully parsed {len(parsed_resumes)} resumes")
        return parsed_resumes
    
    def parse_resume(self, resume: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single resume and extract structured information.
        
        Args:
            resume: Resume dictionary with 'content' key containing raw text
            
        Returns:
            Parsed resume dictionary with additional structured information
        """
        # Create a copy of the resume to avoid modifying the original
        parsed_resume = resume.copy()
        
        # Get the content
        content = resume.get("content", "")
        
        # Extract information
        parsed_resume["name"] = self._extract_name(content)
        parsed_resume["email"] = self._extract_email(content)
        parsed_resume["phone"] = self._extract_phone(content)
        parsed_resume["location"] = self._extract_location(content)
        parsed_resume["skills"] = self._extract_skills(content)
        parsed_resume["education"] = self._extract_education(content)
        parsed_resume["experience"] = self._extract_experience(content)
        parsed_resume["projects"] = self._extract_projects(content)
        
        return parsed_resume
    
    def _extract_name(self, text: str) -> str:
        """Extract the name from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted name or empty string if not found
        """
        # Simple heuristic: First line is often the name
        lines = text.strip().split('\n')
        if lines:
            # Return first non-empty line with less than 5 words (likely a name)
            for line in lines[:3]:  # Check first 3 lines
                line = line.strip()
                if line and len(line.split()) < 5:
                    return line
        
        return ""
    
    def _extract_email(self, text: str) -> str:
        """Extract email address from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted email or empty string if not found
        """
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted phone number or empty string if not found
        """
        # Pattern for phone numbers with various formats
        phone_pattern = r'(\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else ""
    
    def _extract_location(self, text: str) -> str:
        """Extract location from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            Extracted location or empty string if not found
        """
        # Common US city and state pattern
        location_pattern = r'([A-Z][a-zA-Z\s]+,\s*[A-Z]{2})'
        match = re.search(location_pattern, text)
        
        if match:
            return match.group(0)
        
        # If spaCy is available, try to extract locations using NER
        if nlp:
            doc = nlp(text[:1000])  # Process first 1000 chars for efficiency
            for ent in doc.ents:
                if ent.label_ == "GPE":  # Geopolitical Entity
                    return ent.text
        
        return ""
    
    def _extract_skills(self, text: str) -> List[str]:
        """Extract skills from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of extracted skills
        """
        skills = []
        
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
        
        # Look for skills section
        skills_section = self._extract_section(text, "SKILLS")
        
        if skills_section:
            # Process skills section
            if nlp:
                doc = nlp(skills_section)
                for token in doc:
                    if token.text.lower() in common_skills and token.text.lower() not in [s.lower() for s in skills]:
                        skills.append(token.text)
            
            # Fallback to regex pattern matching
            for skill in common_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', skills_section, re.IGNORECASE):
                    if skill.lower() not in [s.lower() for s in skills]:
                        skills.append(skill)
        
        # If no skills found in skills section, search the entire text
        if not skills:
            for skill in common_skills:
                if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                    if skill.lower() not in [s.lower() for s in skills]:
                        skills.append(skill)
        
        return skills
    
    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education information from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of dictionaries containing education information
        """
        education = []
        
        # Extract education section
        education_section = self._extract_section(text, "EDUCATION")
        
        if not education_section:
            return education
        
        # Split into paragraphs (likely different education entries)
        paragraphs = re.split(r'\n\s*\n', education_section)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            edu_entry = {}
            
            # Try to extract degree
            degree_pattern = r'(Bachelor|Master|PhD|Doctorate|BSc|MSc|BA|MA|MBA|BBA|B\.S\.|M\.S\.|B\.A\.|M\.A\.|Ph\.D\.)[^\n]*'
            degree_match = re.search(degree_pattern, paragraph, re.IGNORECASE)
            if degree_match:
                edu_entry["degree"] = degree_match.group(0).strip()
            
            # Try to extract university/institution
            lines = paragraph.split('\n')
            for line in lines:
                if "university" in line.lower() or "college" in line.lower() or "institute" in line.lower() or "school" in line.lower():
                    edu_entry["institution"] = line.strip()
                    break
            
            # Try to extract graduation date
            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|\d{4}'
            date_match = re.search(date_pattern, paragraph)
            if date_match:
                edu_entry["date"] = date_match.group(0).strip()
            
            # Add to education list if we found something
            if edu_entry:
                education.append(edu_entry)
        
        return education
    
    def _extract_experience(self, text: str) -> List[Dict[str, str]]:
        """Extract work experience information from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of dictionaries containing experience information
        """
        experience = []
        
        # Extract experience section
        experience_section = self._extract_section(text, "EXPERIENCE")
        
        if not experience_section:
            return experience
        
        # Split into paragraphs (likely different experience entries)
        paragraphs = re.split(r'\n\s*\n', experience_section)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            exp_entry = {}
            
            # Try to extract job title
            lines = paragraph.split('\n')
            if lines:
                exp_entry["title"] = lines[0].strip()
            
            # Try to extract company
            company_pattern = r'([A-Za-z0-9\s\.&]+),\s*([A-Za-z\s]+,\s*[A-Za-z]{2})'
            company_match = re.search(company_pattern, paragraph)
            if company_match:
                exp_entry["company"] = company_match.group(1).strip()
                exp_entry["location"] = company_match.group(2).strip()
            
            # Try to extract dates
            date_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*-\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\s*-\s*(Present|Current)'
            date_match = re.search(date_pattern, paragraph, re.IGNORECASE)
            if date_match:
                exp_entry["date_range"] = date_match.group(0).strip()
            
            # Extract description (bullet points)
            description_lines = []
            for line in lines[1:]:  # Skip the first line (title)
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    description_lines.append(line.strip())
            
            if description_lines:
                exp_entry["description"] = "\n".join(description_lines)
            
            # Add to experience list if we found something
            if exp_entry:
                experience.append(exp_entry)
        
        return experience
    
    def _extract_projects(self, text: str) -> List[Dict[str, str]]:
        """Extract project information from the resume text.
        
        Args:
            text: Resume text
            
        Returns:
            List of dictionaries containing project information
        """
        projects = []
        
        # Extract projects section
        projects_section = self._extract_section(text, "PROJECTS")
        
        if not projects_section:
            return projects
        
        # Split into paragraphs (likely different project entries)
        paragraphs = re.split(r'\n\s*\n', projects_section)
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            proj_entry = {}
            
            # Try to extract project name
            lines = paragraph.split('\n')
            if lines:
                proj_entry["name"] = lines[0].strip()
            
            # Extract description (bullet points)
            description_lines = []
            for line in lines[1:]:  # Skip the first line (name)
                if line.strip().startswith("-") or line.strip().startswith("•"):
                    description_lines.append(line.strip())
            
            if description_lines:
                proj_entry["description"] = "\n".join(description_lines)
            
            # Add to projects list if we found something
            if proj_entry:
                projects.append(proj_entry)
        
        return projects
    
    def _extract_section(self, text: str, section_type: str) -> str:
        """Extract a specific section from the resume text.
        
        Args:
            text: Resume text
            section_type: Type of section to extract (EDUCATION, EXPERIENCE, SKILLS, PROJECTS)
            
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
            "EDUCATION": r'(?i)\bEDUCATION\b|\bACADEMIC\b|\bQUALIFICATION\b',
            "EXPERIENCE": r'(?i)\bEXPERIENCE\b|\bEMPLOYMENT\b|\bWORK\s+HISTORY\b',
            "SKILLS": r'(?i)\bSKILLS\b|\bABILITIES\b|\bCOMPETENCIES\b',
            "PROJECTS": r'(?i)\bPROJECTS\b|\bPORTFOLIO\b'
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
    
    # Test with a sample resume
    with open("../data/dummy_data/sample_resume.txt", "r") as f:
        sample_resume = {"file_path": "sample_resume.txt", "file_name": "sample_resume.txt", "content": f.read()}
    
    parser = ResumeParser()
    parsed_resume = parser.parse_resume(sample_resume)
    
    print("\nExtracted Information:")
    print(f"Name: {parsed_resume['name']}")
    print(f"Email: {parsed_resume['email']}")
    print(f"Phone: {parsed_resume['phone']}")
    print(f"Location: {parsed_resume['location']}")
    print(f"Skills: {', '.join(parsed_resume['skills'])}")
    
    print("\nEducation:")
    for edu in parsed_resume['education']:
        print(f"- {edu.get('degree', '')} at {edu.get('institution', '')} ({edu.get('date', '')})")
    
    print("\nExperience:")
    for exp in parsed_resume['experience']:
        print(f"- {exp.get('title', '')} at {exp.get('company', '')} ({exp.get('date_range', '')})")
    
    print("\nProjects:")
    for proj in parsed_resume['projects']:
        print(f"- {proj.get('name', '')}")