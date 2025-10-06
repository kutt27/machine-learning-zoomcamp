#!/usr/bin/env python3
"""
Machine Learning Zoomcamp Progress Dashboard
Visualizes learning progress and generates reports
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import re

class ProgressDashboard:
    def __init__(self, progress_file="progress.json"):
        """Initialize the progress dashboard"""
        self.progress_file = Path(progress_file)
        self.data = self.load_progress()
        
        # Course structure
        self.modules = {
            "01-introduction": {
                "name": "Introduction to ML",
                "components": ["theory", "setup", "math", "notebooks"],
                "weight": 1.0
            },
            "02-regression": {
                "name": "Regression",
                "components": ["theory", "math", "notebooks", "exercises"],
                "weight": 1.2
            },
            "03-classification": {
                "name": "Classification", 
                "components": ["theory", "math", "notebooks"],
                "weight": 1.2
            },
            "04-evaluation": {
                "name": "Model Evaluation",
                "components": ["theory", "notebooks"],
                "weight": 1.0
            },
            "05-deployment": {
                "name": "Deployment",
                "components": ["theory", "notebooks"],
                "weight": 1.1
            },
            "06-trees": {
                "name": "Decision Trees",
                "components": ["theory", "notebooks"],
                "weight": 1.1
            },
            "08-deep-learning": {
                "name": "Deep Learning",
                "components": ["theory", "notebooks"],
                "weight": 1.3
            },
            "09-serverless": {
                "name": "Serverless",
                "components": ["theory", "notebooks"],
                "weight": 1.0
            },
            "10-kubernetes": {
                "name": "Kubernetes",
                "components": ["theory", "notebooks"],
                "weight": 1.0
            }
        }
        
        self.projects = {
            "customer-churn": {"name": "Customer Churn Prediction", "weight": 2.0},
            "house-prices": {"name": "House Price Prediction", "weight": 1.5},
            "image-classification": {"name": "Image Classification", "weight": 2.0},
            "recommendation": {"name": "Recommendation System", "weight": 1.5}
        }
    
    def load_progress(self):
        """Load progress data from JSON file"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        else:
            return self.create_default_progress()
    
    def create_default_progress(self):
        """Create default progress structure"""
        return {
            "start_date": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "modules": {},
            "projects": {},
            "exercises": {},
            "study_sessions": [],
            "goals": {},
            "notes": []
        }
    
    def save_progress(self):
        """Save progress data to JSON file"""
        self.data["last_updated"] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def update_module_progress(self, module_id, component, percentage):
        """Update progress for a specific module component"""
        if module_id not in self.data["modules"]:
            self.data["modules"][module_id] = {}
        
        self.data["modules"][module_id][component] = {
            "percentage": percentage,
            "updated": datetime.now().isoformat()
        }
        self.save_progress()
        print(f"Updated {module_id} - {component}: {percentage}%")
    
    def update_project_progress(self, project_id, percentage, notes=""):
        """Update project progress"""
        self.data["projects"][project_id] = {
            "percentage": percentage,
            "notes": notes,
            "updated": datetime.now().isoformat()
        }
        self.save_progress()
        print(f"Updated project {project_id}: {percentage}%")
    
    def add_study_session(self, topic, duration_minutes, notes=""):
        """Add a study session"""
        session = {
            "date": datetime.now().isoformat(),
            "topic": topic,
            "duration_minutes": duration_minutes,
            "notes": notes
        }
        self.data["study_sessions"].append(session)
        self.save_progress()
        print(f"Added study session: {topic} ({duration_minutes} min)")
    
    def calculate_module_progress(self, module_id):
        """Calculate overall progress for a module"""
        if module_id not in self.data["modules"]:
            return 0
        
        module_data = self.data["modules"][module_id]
        components = self.modules[module_id]["components"]
        
        total_progress = 0
        completed_components = 0
        
        for component in components:
            if component in module_data:
                total_progress += module_data[component]["percentage"]
                completed_components += 1
        
        if completed_components == 0:
            return 0
        
        return total_progress / len(components)
    
    def calculate_overall_progress(self):
        """Calculate overall course progress"""
        total_weighted_progress = 0
        total_weight = 0
        
        # Module progress
        for module_id, module_info in self.modules.items():
            progress = self.calculate_module_progress(module_id)
            weight = module_info["weight"]
            total_weighted_progress += progress * weight
            total_weight += weight
        
        # Project progress
        for project_id, project_info in self.projects.items():
            if project_id in self.data["projects"]:
                progress = self.data["projects"][project_id]["percentage"]
                weight = project_info["weight"]
                total_weighted_progress += progress * weight
                total_weight += weight
        
        return total_weighted_progress / total_weight if total_weight > 0 else 0
    
    def generate_progress_report(self):
        """Generate a comprehensive progress report"""
        print("=" * 60)
        print("🎓 MACHINE LEARNING ZOOMCAMP PROGRESS REPORT")
        print("=" * 60)
        
        # Overall progress
        overall_progress = self.calculate_overall_progress()
        print(f"\n📊 Overall Progress: {overall_progress:.1f}%")
        
        # Progress bar
        bar_length = 40
        filled_length = int(bar_length * overall_progress / 100)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"[{bar}] {overall_progress:.1f}%")
        
        # Module progress
        print(f"\n📚 Module Progress:")
        for module_id, module_info in self.modules.items():
            progress = self.calculate_module_progress(module_id)
            bar_length = 20
            filled_length = int(bar_length * progress / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"  {module_info['name']:<20} [{bar}] {progress:.1f}%")
        
        # Project progress
        print(f"\n🎯 Project Progress:")
        for project_id, project_info in self.projects.items():
            if project_id in self.data["projects"]:
                progress = self.data["projects"][project_id]["percentage"]
            else:
                progress = 0
            
            bar_length = 20
            filled_length = int(bar_length * progress / 100)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"  {project_info['name']:<25} [{bar}] {progress:.1f}%")
        
        # Study time analysis
        self.analyze_study_time()
        
        # Recent activity
        self.show_recent_activity()
    
    def analyze_study_time(self):
        """Analyze study time patterns"""
        if not self.data["study_sessions"]:
            print(f"\n⏰ Study Time: No sessions recorded")
            return
        
        sessions = self.data["study_sessions"]
        total_minutes = sum(session["duration_minutes"] for session in sessions)
        total_hours = total_minutes / 60
        
        print(f"\n⏰ Study Time Analysis:")
        print(f"  Total Study Time: {total_hours:.1f} hours ({len(sessions)} sessions)")
        print(f"  Average Session: {total_minutes/len(sessions):.1f} minutes")
        
        # Recent activity (last 7 days)
        recent_sessions = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        for session in sessions:
            session_date = datetime.fromisoformat(session["date"])
            if session_date >= cutoff_date:
                recent_sessions.append(session)
        
        if recent_sessions:
            recent_minutes = sum(session["duration_minutes"] for session in recent_sessions)
            print(f"  Last 7 Days: {recent_minutes/60:.1f} hours ({len(recent_sessions)} sessions)")
        else:
            print(f"  Last 7 Days: No study sessions")
    
    def show_recent_activity(self):
        """Show recent learning activity"""
        print(f"\n📈 Recent Activity:")
        
        # Recent updates
        recent_updates = []
        cutoff_date = datetime.now() - timedelta(days=7)
        
        # Check module updates
        for module_id, module_data in self.data["modules"].items():
            for component, comp_data in module_data.items():
                update_date = datetime.fromisoformat(comp_data["updated"])
                if update_date >= cutoff_date:
                    recent_updates.append({
                        "date": update_date,
                        "type": "module",
                        "item": f"{self.modules[module_id]['name']} - {component}",
                        "progress": comp_data["percentage"]
                    })
        
        # Check project updates
        for project_id, project_data in self.data["projects"].items():
            update_date = datetime.fromisoformat(project_data["updated"])
            if update_date >= cutoff_date:
                recent_updates.append({
                    "date": update_date,
                    "type": "project",
                    "item": self.projects[project_id]["name"],
                    "progress": project_data["percentage"]
                })
        
        # Sort by date
        recent_updates.sort(key=lambda x: x["date"], reverse=True)
        
        if recent_updates:
            for update in recent_updates[:5]:  # Show last 5 updates
                date_str = update["date"].strftime("%m/%d")
                print(f"  {date_str}: {update['item']} → {update['progress']}%")
        else:
            print(f"  No recent updates")
    
    def create_visual_dashboard(self):
        """Create visual dashboard with plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Machine Learning Zoomcamp Progress Dashboard', fontsize=16, fontweight='bold')
        
        # Module progress bar chart
        module_names = [info["name"] for info in self.modules.values()]
        module_progress = [self.calculate_module_progress(mid) for mid in self.modules.keys()]
        
        axes[0, 0].barh(module_names, module_progress, color='skyblue')
        axes[0, 0].set_xlabel('Progress (%)')
        axes[0, 0].set_title('Module Progress')
        axes[0, 0].set_xlim(0, 100)
        
        # Project progress pie chart
        project_names = [info["name"] for info in self.projects.values()]
        project_progress = []
        for pid in self.projects.keys():
            if pid in self.data["projects"]:
                project_progress.append(self.data["projects"][pid]["percentage"])
            else:
                project_progress.append(0)
        
        axes[0, 1].pie(project_progress, labels=project_names, autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Project Progress')
        
        # Study time over time
        if self.data["study_sessions"]:
            sessions_df = pd.DataFrame(self.data["study_sessions"])
            sessions_df['date'] = pd.to_datetime(sessions_df['date'])
            sessions_df['date_only'] = sessions_df['date'].dt.date
            
            daily_study = sessions_df.groupby('date_only')['duration_minutes'].sum() / 60
            
            axes[1, 0].plot(daily_study.index, daily_study.values, marker='o')
            axes[1, 0].set_xlabel('Date')
            axes[1, 0].set_ylabel('Study Hours')
            axes[1, 0].set_title('Daily Study Time')
            axes[1, 0].tick_params(axis='x', rotation=45)
        else:
            axes[1, 0].text(0.5, 0.5, 'No study sessions recorded', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Daily Study Time')
        
        # Overall progress gauge
        overall_progress = self.calculate_overall_progress()
        
        # Create a simple progress circle
        theta = np.linspace(0, 2*np.pi, 100)
        r = 1
        
        # Background circle
        axes[1, 1].plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=10)
        
        # Progress arc
        progress_theta = np.linspace(0, 2*np.pi * overall_progress/100, int(overall_progress))
        if len(progress_theta) > 0:
            axes[1, 1].plot(r * np.cos(progress_theta), r * np.sin(progress_theta), 
                           'green', linewidth=10)
        
        # Add percentage text
        axes[1, 1].text(0, 0, f'{overall_progress:.1f}%', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
        axes[1, 1].set_xlim(-1.5, 1.5)
        axes[1, 1].set_ylim(-1.5, 1.5)
        axes[1, 1].set_aspect('equal')
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Overall Progress')
        
        plt.tight_layout()
        plt.savefig('progress_dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visual dashboard saved as 'progress_dashboard.png'")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='ML Zoomcamp Progress Dashboard')
    parser.add_argument('--progress-file', default='progress.json', 
                       help='Progress data file (default: progress.json)')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate progress report')
    
    # Visual command
    visual_parser = subparsers.add_parser('visual', help='Create visual dashboard')
    
    # Update commands
    update_parser = subparsers.add_parser('update', help='Update progress')
    update_subparsers = update_parser.add_subparsers(dest='update_type')
    
    # Update module
    module_parser = update_subparsers.add_parser('module', help='Update module progress')
    module_parser.add_argument('module_id', help='Module ID (e.g., 01-introduction)')
    module_parser.add_argument('component', help='Component (theory, notebooks, etc.)')
    module_parser.add_argument('percentage', type=float, help='Progress percentage (0-100)')
    
    # Update project
    project_parser = update_subparsers.add_parser('project', help='Update project progress')
    project_parser.add_argument('project_id', help='Project ID')
    project_parser.add_argument('percentage', type=float, help='Progress percentage (0-100)')
    project_parser.add_argument('--notes', default='', help='Optional notes')
    
    # Add study session
    study_parser = update_subparsers.add_parser('study', help='Add study session')
    study_parser.add_argument('topic', help='Study topic')
    study_parser.add_argument('duration', type=int, help='Duration in minutes')
    study_parser.add_argument('--notes', default='', help='Optional notes')
    
    args = parser.parse_args()
    
    dashboard = ProgressDashboard(args.progress_file)
    
    if args.command == 'report':
        dashboard.generate_progress_report()
    elif args.command == 'visual':
        dashboard.create_visual_dashboard()
    elif args.command == 'update':
        if args.update_type == 'module':
            dashboard.update_module_progress(args.module_id, args.component, args.percentage)
        elif args.update_type == 'project':
            dashboard.update_project_progress(args.project_id, args.percentage, args.notes)
        elif args.update_type == 'study':
            dashboard.add_study_session(args.topic, args.duration, args.notes)
    else:
        # Default: show report
        dashboard.generate_progress_report()

if __name__ == "__main__":
    main()
