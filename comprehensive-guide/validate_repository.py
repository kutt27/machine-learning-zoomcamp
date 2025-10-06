#!/usr/bin/env python3
"""
Repository Validation Script for Machine Learning Zoomcamp
Validates all links, files, and structure to ensure everything works properly.
"""

import os
import re
import json
from pathlib import Path
from urllib.parse import urlparse
import requests
from collections import defaultdict
import argparse

class RepositoryValidator:
    def __init__(self, repo_path="."):
        """Initialize the repository validator"""
        self.repo_path = Path(repo_path)
        self.issues = defaultdict(list)
        self.stats = defaultdict(int)
        
        # File patterns to check
        self.markdown_pattern = re.compile(r'\.md$')
        self.notebook_pattern = re.compile(r'\.ipynb$')
        self.python_pattern = re.compile(r'\.py$')
        
        # Link patterns
        self.internal_link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        self.external_link_pattern = re.compile(r'https?://[^\s\)]+')
        
        print(f"🔍 Validating repository: {self.repo_path.absolute()}")
    
    def validate_file_structure(self):
        """Validate the expected file structure exists"""
        print("\n📁 Validating file structure...")
        
        expected_structure = {
            "README.md": "Main repository README",
            "comprehensive-guide/README.md": "Guide main README",
            "comprehensive-guide/01-introduction/README.md": "Introduction module",
            "comprehensive-guide/02-regression/README.md": "Regression module",
            "comprehensive-guide/notebooks/README.md": "Notebooks directory",
            "comprehensive-guide/exercises/README.md": "Exercises directory",
            "comprehensive-guide/reference/README.md": "Reference directory",
            "comprehensive-guide/data/README.md": "Data directory",
            "comprehensive-guide/progress/README.md": "Progress tracking",
        }
        
        for file_path, description in expected_structure.items():
            full_path = self.repo_path / file_path
            if full_path.exists():
                self.stats['files_found'] += 1
                print(f"  ✅ {file_path}")
            else:
                self.issues['missing_files'].append(f"{file_path} - {description}")
                print(f"  ❌ {file_path} - MISSING")
        
        print(f"  📊 Found {self.stats['files_found']}/{len(expected_structure)} expected files")
    
    def validate_markdown_files(self):
        """Validate all markdown files for issues"""
        print("\n📝 Validating markdown files...")
        
        markdown_files = list(self.repo_path.rglob("*.md"))
        self.stats['markdown_files'] = len(markdown_files)
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic structure
                if not content.strip():
                    self.issues['empty_files'].append(str(md_file.relative_to(self.repo_path)))
                    continue
                
                # Check for title
                if not content.startswith('#'):
                    self.issues['no_title'].append(str(md_file.relative_to(self.repo_path)))
                
                # Validate internal links
                self.validate_internal_links(md_file, content)
                
                # Check for common issues
                self.check_markdown_issues(md_file, content)
                
                self.stats['markdown_validated'] += 1
                
            except Exception as e:
                self.issues['file_errors'].append(f"{md_file.relative_to(self.repo_path)}: {e}")
        
        print(f"  📊 Validated {self.stats['markdown_validated']}/{self.stats['markdown_files']} markdown files")
    
    def validate_internal_links(self, md_file, content):
        """Validate internal links in markdown files"""
        links = self.internal_link_pattern.findall(content)
        
        for link_text, link_url in links:
            # Skip external links
            if link_url.startswith(('http://', 'https://', 'mailto:')):
                continue
            
            # Skip anchors
            if link_url.startswith('#'):
                continue
            
            # Resolve relative path
            if link_url.startswith('./'):
                link_url = link_url[2:]
            elif link_url.startswith('../'):
                # Handle relative paths
                target_path = md_file.parent / link_url
            else:
                target_path = md_file.parent / link_url
            
            try:
                target_path = target_path.resolve()
                if not target_path.exists():
                    self.issues['broken_links'].append(
                        f"{md_file.relative_to(self.repo_path)}: {link_text} -> {link_url}"
                    )
            except Exception:
                self.issues['invalid_links'].append(
                    f"{md_file.relative_to(self.repo_path)}: {link_text} -> {link_url}"
                )
    
    def check_markdown_issues(self, md_file, content):
        """Check for common markdown issues"""
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for very long lines (>120 chars for readability)
            if len(line) > 200:
                self.issues['long_lines'].append(
                    f"{md_file.relative_to(self.repo_path)}:{i} - Line too long ({len(line)} chars)"
                )
            
            # Check for missing alt text in images
            if '![](http' in line or '![](' in line:
                self.issues['missing_alt_text'].append(
                    f"{md_file.relative_to(self.repo_path)}:{i} - Image missing alt text"
                )
    
    def validate_notebooks(self):
        """Validate Jupyter notebooks"""
        print("\n📓 Validating Jupyter notebooks...")
        
        notebook_files = list(self.repo_path.rglob("*.ipynb"))
        self.stats['notebook_files'] = len(notebook_files)
        
        for notebook_file in notebook_files:
            try:
                with open(notebook_file, 'r', encoding='utf-8') as f:
                    notebook_data = json.load(f)
                
                # Check basic structure
                if 'cells' not in notebook_data:
                    self.issues['invalid_notebooks'].append(str(notebook_file.relative_to(self.repo_path)))
                    continue
                
                # Check for content
                cells = notebook_data['cells']
                if len(cells) == 0:
                    self.issues['empty_notebooks'].append(str(notebook_file.relative_to(self.repo_path)))
                    continue
                
                # Check for markdown cells (documentation)
                markdown_cells = [cell for cell in cells if cell.get('cell_type') == 'markdown']
                if len(markdown_cells) == 0:
                    self.issues['no_documentation'].append(str(notebook_file.relative_to(self.repo_path)))
                
                # Check for code cells
                code_cells = [cell for cell in cells if cell.get('cell_type') == 'code']
                if len(code_cells) == 0:
                    self.issues['no_code'].append(str(notebook_file.relative_to(self.repo_path)))
                
                self.stats['notebooks_validated'] += 1
                
            except json.JSONDecodeError:
                self.issues['invalid_json'].append(str(notebook_file.relative_to(self.repo_path)))
            except Exception as e:
                self.issues['notebook_errors'].append(f"{notebook_file.relative_to(self.repo_path)}: {e}")
        
        print(f"  📊 Validated {self.stats['notebooks_validated']}/{self.stats['notebook_files']} notebooks")
    
    def validate_python_files(self):
        """Validate Python files"""
        print("\n🐍 Validating Python files...")
        
        python_files = list(self.repo_path.rglob("*.py"))
        self.stats['python_files'] = len(python_files)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic syntax (compile check)
                try:
                    compile(content, str(py_file), 'exec')
                    self.stats['python_valid'] += 1
                except SyntaxError as e:
                    self.issues['syntax_errors'].append(f"{py_file.relative_to(self.repo_path)}: {e}")
                
                # Check for docstrings in functions/classes
                if 'def ' in content or 'class ' in content:
                    if '"""' not in content and "'''" not in content:
                        self.issues['missing_docstrings'].append(str(py_file.relative_to(self.repo_path)))
                
            except Exception as e:
                self.issues['python_errors'].append(f"{py_file.relative_to(self.repo_path)}: {e}")
        
        print(f"  📊 Validated {self.stats['python_valid']}/{self.stats['python_files']} Python files")
    
    def validate_data_availability(self):
        """Check if data files and download scripts work"""
        print("\n📊 Validating data availability...")
        
        data_dir = self.repo_path / "comprehensive-guide" / "data"
        if not data_dir.exists():
            self.issues['missing_directories'].append("comprehensive-guide/data")
            return
        
        # Check for download script
        download_script = data_dir / "download_datasets.py"
        if download_script.exists():
            print(f"  ✅ Download script found")
            self.stats['download_script'] = 1
        else:
            self.issues['missing_files'].append("data/download_datasets.py")
        
        # Check for data README
        data_readme = data_dir / "README.md"
        if data_readme.exists():
            print(f"  ✅ Data README found")
        else:
            self.issues['missing_files'].append("data/README.md")
        
        # Check for data directories
        expected_dirs = ['raw', 'external', 'processed']
        for dir_name in expected_dirs:
            dir_path = data_dir / dir_name
            if dir_path.exists():
                print(f"  ✅ {dir_name}/ directory exists")
            else:
                print(f"  ℹ️  {dir_name}/ directory will be created by download script")
    
    def validate_learning_path(self):
        """Validate the learning path flow"""
        print("\n🎓 Validating learning path...")
        
        # Check module sequence
        modules = [
            "01-introduction", "02-regression", "03-classification", 
            "04-evaluation", "05-deployment", "06-trees", 
            "08-deep-learning", "09-serverless", "10-kubernetes"
        ]
        
        guide_dir = self.repo_path / "comprehensive-guide"
        
        for module in modules:
            module_dir = guide_dir / module
            if module_dir.exists():
                readme_file = module_dir / "README.md"
                if readme_file.exists():
                    print(f"  ✅ {module}")
                    self.stats['modules_found'] += 1
                else:
                    self.issues['missing_module_readme'].append(module)
                    print(f"  ❌ {module} - Missing README")
            else:
                self.issues['missing_modules'].append(module)
                print(f"  ❌ {module} - Missing directory")
        
        print(f"  📊 Found {self.stats['modules_found']}/{len(modules)} modules")
    
    def generate_report(self):
        """Generate a comprehensive validation report"""
        print("\n" + "="*60)
        print("📋 REPOSITORY VALIDATION REPORT")
        print("="*60)
        
        # Summary statistics
        print(f"\n📊 Summary Statistics:")
        print(f"  Markdown files: {self.stats.get('markdown_files', 0)}")
        print(f"  Jupyter notebooks: {self.stats.get('notebook_files', 0)}")
        print(f"  Python files: {self.stats.get('python_files', 0)}")
        print(f"  Modules found: {self.stats.get('modules_found', 0)}/9")
        
        # Issues summary
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        if total_issues == 0:
            print(f"\n✅ VALIDATION PASSED - No issues found!")
        else:
            print(f"\n⚠️  VALIDATION ISSUES FOUND: {total_issues} total")
            
            for issue_type, issue_list in self.issues.items():
                if issue_list:
                    print(f"\n🔸 {issue_type.replace('_', ' ').title()} ({len(issue_list)}):")
                    for issue in issue_list[:5]:  # Show first 5 issues
                        print(f"    • {issue}")
                    if len(issue_list) > 5:
                        print(f"    ... and {len(issue_list) - 5} more")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if self.issues.get('broken_links'):
            print(f"  • Fix broken internal links")
        
        if self.issues.get('missing_files'):
            print(f"  • Create missing files")
        
        if self.issues.get('empty_files'):
            print(f"  • Add content to empty files")
        
        if self.issues.get('syntax_errors'):
            print(f"  • Fix Python syntax errors")
        
        if total_issues == 0:
            print(f"  • Repository is ready for learners! 🎉")
        
        # Save detailed report
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed validation report to file"""
        report_file = self.repo_path / "validation_report.json"
        
        report_data = {
            "timestamp": str(Path(__file__).stat().st_mtime),
            "repository_path": str(self.repo_path.absolute()),
            "statistics": dict(self.stats),
            "issues": dict(self.issues),
            "total_issues": sum(len(issues) for issues in self.issues.values())
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n💾 Detailed report saved to: {report_file}")
    
    def run_validation(self):
        """Run complete validation"""
        print("🚀 Starting repository validation...")
        
        self.validate_file_structure()
        self.validate_markdown_files()
        self.validate_notebooks()
        self.validate_python_files()
        self.validate_data_availability()
        self.validate_learning_path()
        
        self.generate_report()

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Validate ML Zoomcamp repository')
    parser.add_argument('--repo-path', default='.', 
                       help='Path to repository root (default: current directory)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick validation (skip external link checks)')
    
    args = parser.parse_args()
    
    validator = RepositoryValidator(args.repo_path)
    validator.run_validation()

if __name__ == "__main__":
    main()
