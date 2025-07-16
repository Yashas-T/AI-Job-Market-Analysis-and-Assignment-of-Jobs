#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Module for Resume-Job Matching System.

This module provides functionality to generate visual representations
of resume-job matching and recommendation results.
"""

import logging
from typing import Dict, List, Any, Tuple
import os
import json
from pathlib import Path

# Try to import visualization libraries, but provide fallbacks if not available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    logging.warning("Visualization libraries not available. Install matplotlib, seaborn, and pandas for full visualization support.")

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """Class for generating visualizations of matching and recommendation results."""
    
    def __init__(self, output_dir="./output"):
        """Initialize the visualization engine.
        
        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = output_dir
        self._ensure_output_dir()
        self.visualization_available = VISUALIZATION_AVAILABLE
        logger.info(f"Initialized VisualizationEngine with output directory: {output_dir}")
        
        if not self.visualization_available:
            logger.warning("Visualization libraries not available. Some functions will be limited.")
    
    def _ensure_output_dir(self):
        """Ensure the output directory exists."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def generate_match_score_chart(self, recommendations, resume_id, save=True):
        """Generate a bar chart of job match scores for a resume.
        
        Args:
            recommendations: List of recommendation dictionaries
            resume_id: ID of the resume
            save: Whether to save the chart to a file
            
        Returns:
            Path to the saved chart file or None if visualization is not available
        """
        if not self.visualization_available:
            logger.warning("Cannot generate match score chart: visualization libraries not available.")
            return self._generate_text_report(recommendations, resume_id, "match_scores")
        
        try:
            # Extract job titles and scores
            job_titles = [rec["job"].get("title", f"Job {i}") for i, rec in enumerate(recommendations)]
            total_scores = [rec["score"] for rec in recommendations]
            skill_scores = [rec["skill_score"] for rec in recommendations]
            location_scores = [rec["location_score"] for rec in recommendations]
            embedding_scores = [rec["embedding_score"] for rec in recommendations]
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                "Job": job_titles,
                "Total Score": total_scores,
                "Skill Match": skill_scores,
                "Location Match": location_scores,
                "Content Match": embedding_scores
            })
            
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Create the grouped bar chart
            bar_width = 0.2
            x = np.arange(len(job_titles))
            
            plt.bar(x - bar_width*1.5, df["Total Score"], width=bar_width, label="Total Score", color="#3498db")
            plt.bar(x - bar_width/2, df["Skill Match"], width=bar_width, label="Skill Match", color="#2ecc71")
            plt.bar(x + bar_width/2, df["Location Match"], width=bar_width, label="Location Match", color="#e74c3c")
            plt.bar(x + bar_width*1.5, df["Content Match"], width=bar_width, label="Content Match", color="#f39c12")
            
            # Add labels and title
            plt.xlabel("Job Opportunities")
            plt.ylabel("Match Score (0-1)")
            plt.title(f"Job Match Scores for Resume: {resume_id}")
            plt.xticks(x, job_titles, rotation=45, ha="right")
            plt.ylim(0, 1.1)  # Scores are between 0 and 1
            plt.legend()
            plt.tight_layout()
            
            # Save the chart if requested
            if save:
                filename = f"match_scores_{self._sanitize_filename(resume_id)}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Saved match score chart to {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                plt.close()
                return None
            
        except Exception as e:
            logger.error(f"Error generating match score chart: {str(e)}")
            return self._generate_text_report(recommendations, resume_id, "match_scores")
    
    def generate_skill_gap_chart(self, skill_gap, resume_id, save=True):
        """Generate a bar chart of missing skills for a resume.
        
        Args:
            skill_gap: Skill gap analysis dictionary
            resume_id: ID of the resume
            save: Whether to save the chart to a file
            
        Returns:
            Path to the saved chart file or None if visualization is not available
        """
        if not self.visualization_available:
            logger.warning("Cannot generate skill gap chart: visualization libraries not available.")
            return self._generate_text_report({"skill_gap": skill_gap}, resume_id, "skill_gap")
        
        try:
            # Check if we have valid data
            if not skill_gap or "missing_skills" not in skill_gap or not skill_gap["missing_skills"]:
                logger.warning("No skill gap data available for visualization")
                return self._generate_text_report({"skill_gap": skill_gap}, resume_id, "skill_gap")
            
            # Extract skills and frequencies
            skills = skill_gap["missing_skills"]
            frequencies = [skill_gap["skill_frequency"][skill] for skill in skills]
            
            # Limit to top 10 skills for readability
            if len(skills) > 10:
                skills = skills[:10]
                frequencies = frequencies[:10]
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame({
                "Skill": skills,
                "Frequency": frequencies
            })
            
            # Set up the plot
            plt.figure(figsize=(12, 8))
            
            # Create the bar chart
            sns.barplot(x="Frequency", y="Skill", data=df, palette="viridis")
            
            # Add labels and title
            plt.xlabel("Number of Jobs Requiring Skill")
            plt.ylabel("Missing Skills")
            plt.title(f"Skill Gap Analysis for Resume: {resume_id}")
            plt.tight_layout()
            
            # Save the chart if requested
            if save:
                filename = f"skill_gap_{self._sanitize_filename(resume_id)}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Saved skill gap chart to {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                plt.close()
                return None
            
        except Exception as e:
            logger.error(f"Error generating skill gap chart: {str(e)}")
            return self._generate_text_report({"skill_gap": skill_gap}, resume_id, "skill_gap")
    
    def generate_skill_match_heatmap(self, recommendations, resume_id, save=True):
        """Generate a heatmap of skill matches for a resume.
        
        Args:
            recommendations: List of recommendation dictionaries
            resume_id: ID of the resume
            save: Whether to save the heatmap to a file
            
        Returns:
            Path to the saved heatmap file or None if visualization is not available
        """
        if not self.visualization_available:
            logger.warning("Cannot generate skill match heatmap: visualization libraries not available.")
            return self._generate_text_report(recommendations, resume_id, "skill_match")
        
        try:
            # Check if we have valid data
            if not recommendations:
                logger.warning("No recommendations available for visualization")
                return self._generate_text_report(recommendations, resume_id, "skill_match")
            
            # Collect all unique skills across all jobs
            all_skills = set()
            for rec in recommendations:
                all_skills.update(rec["matching_skills"])
                all_skills.update(rec["missing_skills"])
            
            all_skills = sorted(list(all_skills))
            
            # Create a matrix of skill matches
            job_titles = [rec["job"].get("title", f"Job {i}") for i, rec in enumerate(recommendations)]
            match_matrix = np.zeros((len(job_titles), len(all_skills)))
            
            for i, rec in enumerate(recommendations):
                for j, skill in enumerate(all_skills):
                    if skill in rec["matching_skills"]:
                        match_matrix[i, j] = 1  # Matching skill
                    elif skill in rec["missing_skills"]:
                        match_matrix[i, j] = -1  # Missing skill
            
            # Set up the plot
            plt.figure(figsize=(max(12, len(all_skills) * 0.5), max(8, len(job_titles) * 0.5)))
            
            # Create the heatmap
            cmap = sns.diverging_palette(10, 133, as_cmap=True)  # Red for missing, green for matching
            sns.heatmap(match_matrix, cmap=cmap, center=0, 
                      xticklabels=all_skills, yticklabels=job_titles,
                      cbar_kws={"label": "Skill Match Status"})
            
            # Add labels and title
            plt.xlabel("Skills")
            plt.ylabel("Job Opportunities")
            plt.title(f"Skill Match Heatmap for Resume: {resume_id}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            # Save the heatmap if requested
            if save:
                filename = f"skill_heatmap_{self._sanitize_filename(resume_id)}.png"
                filepath = os.path.join(self.output_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                logger.info(f"Saved skill match heatmap to {filepath}")
                plt.close()
                return filepath
            else:
                plt.show()
                plt.close()
                return None
            
        except Exception as e:
            logger.error(f"Error generating skill match heatmap: {str(e)}")
            return self._generate_text_report(recommendations, resume_id, "skill_match")
    
    def generate_html_report(self, recommendations, skill_gap, resume_id):
        """Generate an HTML report of job recommendations and skill gap analysis.
        
        Args:
            recommendations: List of recommendation dictionaries
            skill_gap: Skill gap analysis dictionary
            resume_id: ID of the resume
            
        Returns:
            Path to the saved HTML report
        """
        try:
            # Check if we have valid data
            if not recommendations:
                logger.warning("No recommendations available for report generation")
                return self._generate_text_report({"recommendations": recommendations, "skill_gap": skill_gap}, resume_id, "report")
            
            # Ensure skill_gap has required fields
            if not skill_gap:
                skill_gap = {"missing_skills": [], "skill_frequency": {}, "total_jobs_analyzed": 0}
            elif "total_jobs_analyzed" not in skill_gap:
                skill_gap["total_jobs_analyzed"] = len(recommendations)
            
            # Generate charts if visualization is available
            chart_paths = {}
            if self.visualization_available:
                chart_paths["match_scores"] = self.generate_match_score_chart(recommendations, resume_id)
                chart_paths["skill_gap"] = self.generate_skill_gap_chart(skill_gap, resume_id)
                chart_paths["skill_match"] = self.generate_skill_match_heatmap(recommendations, resume_id)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Job Recommendation Report for {resume_id}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .recommendation {{ border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 5px; }}
                    .job-title {{ font-size: 1.2em; font-weight: bold; color: #3498db; }}
                    .company {{ font-style: italic; color: #7f8c8d; }}
                    .score {{ font-weight: bold; color: #27ae60; }}
                    .skills {{ margin-top: 10px; }}
                    .matching-skills {{ color: #27ae60; }}
                    .missing-skills {{ color: #e74c3c; }}
                    .chart {{ margin: 20px 0; max-width: 100%; }}
                    .chart img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                    .skill-gap {{ margin-top: 20px; }}
                    table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Job Recommendation Report</h1>
                <p><strong>Resume ID:</strong> {resume_id}</p>
                
                <h2>Top Job Recommendations</h2>
            """
            
            # Add recommendations
            for i, rec in enumerate(recommendations):
                job = rec["job"]
                html_content += f"""
                <div class="recommendation">
                    <div class="job-title">{i+1}. {job.get('title', 'Unknown Job')}</div>
                    <div class="company">{job.get('company', '')}</div>
                    <div class="score">Match Score: {rec['score']:.2f}</div>
                    <p>{rec['explanation']}</p>
                    
                    <div class="skills">
                        <div class="matching-skills">
                            <strong>Matching Skills:</strong> {', '.join(rec['matching_skills']) if rec['matching_skills'] else 'None'}
                        </div>
                        <div class="missing-skills">
                            <strong>Missing Skills:</strong> {', '.join(rec['missing_skills']) if rec['missing_skills'] else 'None'}
                        </div>
                    </div>
                    
                    <table>
                        <tr>
                            <th>Total Score</th>
                            <th>Skill Match</th>
                            <th>Location Match</th>
                            <th>Content Match</th>
                        </tr>
                        <tr>
                            <td>{rec['score']:.2f}</td>
                            <td>{rec['skill_score']:.2f}</td>
                            <td>{rec['location_score']:.2f}</td>
                            <td>{rec['embedding_score']:.2f}</td>
                        </tr>
                    </table>
                </div>
                """
            
            # Add charts if available
            if self.visualization_available and chart_paths:
                html_content += "<h2>Visualization Charts</h2>"
                
                for chart_type, path in chart_paths.items():
                    if path:
                        # Convert absolute path to relative path for HTML
                        rel_path = os.path.relpath(path, self.output_dir)
                        title = {
                            "match_scores": "Job Match Scores",
                            "skill_gap": "Skill Gap Analysis",
                            "skill_match": "Skill Match Heatmap"
                        }.get(chart_type, chart_type.replace("_", " ").title())
                        
                        html_content += f"""
                        <div class="chart">
                            <h3>{title}</h3>
                            <img src="{rel_path}" alt="{title}">
                        </div>
                        """
            
            # Add skill gap analysis
            html_content += f"""
                <h2>Skill Gap Analysis</h2>
                <div class="skill-gap">
                    <p><strong>Total Jobs Analyzed:</strong> {skill_gap['total_jobs_analyzed']}</p>
                    <p><strong>Missing Skills (in order of importance):</strong></p>
                    <table>
                        <tr>
                            <th>Skill</th>
                            <th>Number of Jobs Requiring</th>
                        </tr>
            """
            
            for skill in skill_gap["missing_skills"]:
                frequency = skill_gap["skill_frequency"][skill]
                html_content += f"""
                        <tr>
                            <td>{skill}</td>
                            <td>{frequency}</td>
                        </tr>
                """
            
            html_content += """
                    </table>
                </div>
                
                <div style="margin-top: 30px; color: #7f8c8d; font-size: 0.8em;">
                    <p>Generated by Resume-Job Matching and Recommendation System</p>
                </div>
            </body>
            </html>
            """
            
            # Save the HTML report
            filename = f"report_{self._sanitize_filename(resume_id)}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"Saved HTML report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return self._generate_text_report({"recommendations": recommendations, "skill_gap": skill_gap}, resume_id, "report")
    
    def _generate_text_report(self, data, resume_id, report_type):
        """Generate a text-based report when visualization is not available.
        
        Args:
            data: Data to include in the report
            resume_id: ID of the resume
            report_type: Type of report
            
        Returns:
            Path to the saved text report
        """
        try:
            # Create a JSON report
            filename = f"{report_type}_{self._sanitize_filename(resume_id)}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved text report to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating text report: {str(e)}")
            return None
    
    def _sanitize_filename(self, filename):
        """Sanitize a string to be used as a filename.
        
        Args:
            filename: String to sanitize
            
        Returns:
            Sanitized string
        """
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length
        if len(filename) > 50:
            filename = filename[:47] + '...'
        
        return filename


# For testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if visualization libraries are available
    if not VISUALIZATION_AVAILABLE:
        print("Warning: Visualization libraries not available. Install matplotlib, seaborn, and pandas for full visualization support.")
        print("Continuing with limited functionality...")
    
    # Create sample data for testing
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
    
    sample_recommendations = [
        {
            "job": sample_job1,
            "score": 0.85,
            "embedding_score": 0.9,
            "skill_score": 0.8,
            "location_score": 0.7,
            "matching_skills": ["Python", "SQL", "Machine Learning"],
            "missing_skills": ["Statistics"],
            "explanation": "Data Scientist is a good match for your profile. You have 3 matching skills including Python, SQL, Machine Learning. Consider developing skills in Statistics to improve your match. The job location matches your preferred location. Overall, this is a good match for your profile."
        },
        {
            "job": sample_job2,
            "score": 0.75,
            "embedding_score": 0.8,
            "skill_score": 0.6,
            "location_score": 0.3,
            "matching_skills": ["Python"],
            "missing_skills": ["TensorFlow", "Deep Learning", "Computer Vision"],
            "explanation": "Machine Learning Engineer is a good match for your profile. You have 1 matching skills including Python. Consider developing skills in TensorFlow, Deep Learning, Computer Vision to improve your match. Overall, this is a good match for your profile."
        }
    ]
    
    sample_skill_gap = {
        "missing_skills": ["TensorFlow", "Deep Learning", "Computer Vision", "Statistics"],
        "skill_frequency": {"TensorFlow": 1, "Deep Learning": 1, "Computer Vision": 1, "Statistics": 1},
        "total_jobs_analyzed": 2
    }
    
    # Create visualization engine
    output_dir = "./output"
    viz_engine = VisualizationEngine(output_dir)
    
    # Generate visualizations
    if VISUALIZATION_AVAILABLE:
        print("\nGenerating visualizations...")
        match_chart_path = viz_engine.generate_match_score_chart(sample_recommendations, "sample_resume")
        skill_gap_chart_path = viz_engine.generate_skill_gap_chart(sample_skill_gap, "sample_resume")
        skill_match_heatmap_path = viz_engine.generate_skill_match_heatmap(sample_recommendations, "sample_resume")
        
        print(f"Match Score Chart: {match_chart_path}")
        print(f"Skill Gap Chart: {skill_gap_chart_path}")
        print(f"Skill Match Heatmap: {skill_match_heatmap_path}")
    
    # Generate HTML report
    print("\nGenerating HTML report...")
    report_path = viz_engine.generate_html_report(sample_recommendations, sample_skill_gap, "sample_resume")
    print(f"HTML Report: {report_path}")
    
    print("\nDone!")