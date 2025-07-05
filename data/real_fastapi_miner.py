"""
Real FastAPI GitHub Mining Script

This script actually mines real FastAPI repositories from GitHub,
extracts code patterns, and generates training examples.
NO HARD-CODING - everything comes from real repositories.
"""

import json
import requests
import time
import re
import ast
import base64
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
import argparse
from collections import defaultdict


@dataclass
class ExtractedPattern:
    """A real code pattern extracted from GitHub."""
    code: str
    pattern_type: str
    repo_name: str
    file_path: str
    http_method: str
    has_validation: bool
    has_auth: bool
    has_database: bool
    complexity_score: int


class RealFastAPIMiner:
    """Actually mines FastAPI repositories from GitHub."""
    
    def __init__(self, github_token: str):
        """Initialize with GitHub token."""
        self.github_token = github_token
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "FastAPI-Dataset-Miner"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Storage for extracted patterns
        self.extracted_patterns: List[ExtractedPattern] = []
        
    def search_fastapi_repositories(self, max_repos: int = 50) -> List[Dict]:
        """Search for real FastAPI repositories on GitHub."""
        self.logger.info(f"🔍 Searching for FastAPI repositories (max: {max_repos})...")
        
        search_queries = [
            "fastapi stars:>100 language:python",
            "fastapi crud stars:>50 language:python", 
            "fastapi example stars:>30 language:python",
            "fastapi tutorial stars:>20 language:python",
            "fastapi rest api stars:>50 language:python"
        ]
        
        all_repos = []
        
        for query in search_queries:
            self.logger.info(f"🔍 Searching: {query}")
            
            try:
                url = "https://api.github.com/search/repositories"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": 20
                }
                
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                repos = data.get("items", [])
                
                # Filter for quality repos
                quality_repos = [repo for repo in repos if self._is_quality_repo(repo)]
                all_repos.extend(quality_repos)
                
                self.logger.info(f"✅ Found {len(quality_repos)} quality repos in this search")
                
                # Rate limiting
                time.sleep(2)
                
            except requests.exceptions.RequestException as e:
                self.logger.error(f"❌ Error searching repositories: {e}")
                continue
        
        # Remove duplicates and limit
        unique_repos = []
        seen_names = set()
        for repo in all_repos:
            if repo["full_name"] not in seen_names:
                unique_repos.append(repo)
                seen_names.add(repo["full_name"])
                
        final_repos = unique_repos[:max_repos]
        self.logger.info(f"🎯 Selected {len(final_repos)} unique repositories for mining")
        
        return final_repos
    
    def _is_quality_repo(self, repo: Dict) -> bool:
        """Filter for quality repositories."""
        # Skip forks and archived repos
        if repo.get("fork", False) or repo.get("archived", False):
            return False
            
        # Require minimum stars
        if repo.get("stargazers_count", 0) < 5:
            return False
            
        # Check for FastAPI in description or name
        description = (repo.get("description") or "").lower()
        name = repo.get("name", "").lower()
        
        fastapi_keywords = ["fastapi", "fast api", "api", "rest", "backend"]
        
        return any(keyword in description or keyword in name for keyword in fastapi_keywords)
    
    def mine_repository(self, repo: Dict) -> List[ExtractedPattern]:
        """Mine a single repository for FastAPI patterns."""
        self.logger.info(f"⛏️  Mining repository: {repo['full_name']}")
        
        patterns = []
        
        try:
            # Get repository file tree
            tree_url = f"https://api.github.com/repos/{repo['full_name']}/git/trees/{repo['default_branch']}?recursive=1"
            response = self.session.get(tree_url, timeout=30)
            response.raise_for_status()
            
            tree_data = response.json()
            
            # Find Python files that likely contain FastAPI code
            python_files = []
            for item in tree_data.get("tree", []):
                if (item.get("type") == "blob" and 
                    item.get("path", "").endswith(".py") and 
                    self._is_likely_fastapi_file(item.get("path", ""))):
                    python_files.append(item)
            
            self.logger.info(f"📁 Found {len(python_files)} potential FastAPI files")
            
            # Process each file
            for file_item in python_files[:10]:  # Limit to prevent rate limiting
                file_patterns = self._extract_patterns_from_file(repo, file_item)
                patterns.extend(file_patterns)
                
                # Rate limiting
                time.sleep(1)
                
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Error mining repository {repo['full_name']}: {e}")
        
        self.logger.info(f"✅ Extracted {len(patterns)} patterns from {repo['full_name']}")
        return patterns
    
    def _is_likely_fastapi_file(self, file_path: str) -> bool:
        """Check if file path suggests FastAPI code."""
        path_lower = file_path.lower()
        
        # Skip test files and migrations
        if any(skip in path_lower for skip in ["test", "migration", "__pycache__"]):
            return False
        
        # Look for FastAPI-related patterns
        fastapi_indicators = [
            "main.py", "app.py", "server.py", "api.py",
            "router", "route", "endpoint", "handler", 
            "crud", "model", "schema", "auth"
        ]
        
        return any(indicator in path_lower for indicator in fastapi_indicators)
    
    def _extract_patterns_from_file(self, repo: Dict, file_item: Dict) -> List[ExtractedPattern]:
        """Extract FastAPI patterns from a single file."""
        patterns = []
        
        try:
            # Get file content
            file_url = f"https://api.github.com/repos/{repo['full_name']}/contents/{file_item['path']}"
            response = self.session.get(file_url, timeout=30)
            response.raise_for_status()
            
            file_data = response.json()
            
            # Decode content
            if file_data.get("encoding") == "base64":
                content = base64.b64decode(file_data["content"]).decode("utf-8")
            else:
                content = file_data.get("content", "")
            
            # Skip if not FastAPI-related
            if not self._contains_fastapi_code(content):
                return patterns
            
            self.logger.info(f"📄 Processing file: {file_item['path']}")
            
            # Extract patterns using AST
            try:
                tree = ast.parse(content)
                file_patterns = self._extract_patterns_from_ast(tree, content, repo, file_item['path'])
                patterns.extend(file_patterns)
                
            except SyntaxError as e:
                self.logger.warning(f"⚠️  Syntax error in {file_item['path']}: {e}")
                
        except (requests.exceptions.RequestException, UnicodeDecodeError) as e:
            self.logger.warning(f"⚠️  Error processing file {file_item['path']}: {e}")
            
        return patterns
    
    def _contains_fastapi_code(self, content: str) -> bool:
        """Check if content contains FastAPI code."""
        fastapi_indicators = [
            "from fastapi import",
            "import fastapi",
            "FastAPI(",
            "@app.get",
            "@app.post",
            "@app.put",
            "@app.delete"
        ]
        
        return any(indicator in content for indicator in fastapi_indicators)
    
    def _extract_patterns_from_ast(self, tree: ast.AST, content: str, repo: Dict, file_path: str) -> List[ExtractedPattern]:
        """Extract FastAPI patterns from AST."""
        patterns = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function has FastAPI route decorator
                for decorator in node.decorator_list:
                    if self._is_fastapi_route_decorator(decorator):
                        # Extract the function code
                        try:
                            start_line = node.lineno - 1
                            end_line = getattr(node, 'end_lineno', start_line + 10)
                            
                            # Include decorator lines
                            decorator_start = min(d.lineno - 1 for d in node.decorator_list)
                            
                            function_code = '\n'.join(lines[decorator_start:end_line])
                            
                            # Analyze the pattern
                            http_method = self._get_http_method(decorator)
                            has_validation = self._has_pydantic_validation(function_code)
                            has_auth = self._has_authentication(function_code)
                            has_database = self._has_database_usage(function_code)
                            complexity = self._calculate_complexity(function_code)
                            
                            pattern = ExtractedPattern(
                                code=function_code,
                                pattern_type=f"route_{http_method}",
                                repo_name=repo['full_name'],
                                file_path=file_path,
                                http_method=http_method,
                                has_validation=has_validation,
                                has_auth=has_auth,
                                has_database=has_database,
                                complexity_score=complexity
                            )
                            
                            patterns.append(pattern)
                            
                        except Exception as e:
                            self.logger.warning(f"⚠️  Error extracting function: {e}")
                            
        return patterns
    
    def _is_fastapi_route_decorator(self, decorator: ast.expr) -> bool:
        """Check if decorator is a FastAPI route decorator."""
        if isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr in ["get", "post", "put", "delete", "patch", "options"]
        return False
    
    def _get_http_method(self, decorator: ast.expr) -> str:
        """Extract HTTP method from decorator."""
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
            return decorator.func.attr
        return "unknown"
    
    def _has_pydantic_validation(self, code: str) -> bool:
        """Check if code uses Pydantic validation."""
        pydantic_indicators = ["BaseModel", "Field", "validator", "EmailStr", "constr"]
        return any(indicator in code for indicator in pydantic_indicators)
    
    def _has_authentication(self, code: str) -> bool:
        """Check if code uses authentication."""
        auth_indicators = ["Depends", "HTTPBearer", "OAuth2", "jwt", "token", "auth"]
        return any(indicator in code for indicator in auth_indicators)
    
    def _has_database_usage(self, code: str) -> bool:
        """Check if code uses database."""
        db_indicators = ["Session", "query", "db.", "database", "orm", "sqlalchemy"]
        return any(indicator in code for indicator in db_indicators)
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate complexity score for code."""
        lines = len(code.split('\n'))
        
        complexity_factors = [
            ("async def", 2),
            ("try:", 3),
            ("except", 2),
            ("HTTPException", 2),
            ("Depends", 3),
            ("BaseModel", 2),
            ("Session", 3),
            ("jwt", 4),
            ("background_tasks", 4),
            ("WebSocket", 5)
        ]
        
        score = lines
        for pattern, weight in complexity_factors:
            if pattern in code:
                score += weight
        
        return min(score, 50)  # Cap at 50
    
    def generate_training_examples(self, patterns: List[ExtractedPattern]) -> List[Dict]:
        """Generate training examples from extracted patterns."""
        self.logger.info(f"🔄 Generating training examples from {len(patterns)} patterns...")
        
        examples = []
        
        for pattern in patterns:
            # Generate instruction based on pattern characteristics
            instruction = self._generate_instruction(pattern)
            input_text = self._generate_input_text(pattern)
            
            # Clean up the code
            cleaned_code = self._clean_code(pattern.code)
            
            # Categorize
            category = self._categorize_pattern(pattern)
            difficulty = self._assess_difficulty(pattern)
            tags = self._generate_tags(pattern)
            
            example = {
                "instruction": instruction,
                "input": input_text,
                "output": cleaned_code,
                "category": category,
                "difficulty": difficulty,
                "tags": tags,
                "source_repo": pattern.repo_name,
                "source_file": pattern.file_path,
                "complexity_score": pattern.complexity_score
            }
            
            examples.append(example)
        
        # Remove duplicates
        unique_examples = self._deduplicate_examples(examples)
        
        self.logger.info(f"✅ Generated {len(unique_examples)} unique training examples")
        return unique_examples
    
    def _generate_instruction(self, pattern: ExtractedPattern) -> str:
        """Generate instruction text for pattern."""
        base_instruction = f"Create a FastAPI {pattern.http_method.upper()} endpoint"
        
        features = []
        if pattern.has_validation:
            features.append("with Pydantic validation")
        if pattern.has_auth:
            features.append("with authentication")
        if pattern.has_database:
            features.append("with database integration")
        
        if features:
            return f"{base_instruction} {' and '.join(features)}"
        
        return base_instruction
    
    def _generate_input_text(self, pattern: ExtractedPattern) -> str:
        """Generate input text for pattern."""
        if pattern.has_validation and pattern.has_database:
            return "Include proper request validation and database operations"
        elif pattern.has_validation:
            return "Include proper request validation"
        elif pattern.has_database:
            return "Include database operations"
        elif pattern.has_auth:
            return "Include authentication and authorization"
        else:
            return "Include proper error handling and response formatting"
    
    def _clean_code(self, code: str) -> str:
        """Clean up extracted code."""
        # Remove empty lines at start and end
        lines = code.split('\n')
        
        # Find first and last non-empty lines
        start_idx = 0
        end_idx = len(lines) - 1
        
        while start_idx < len(lines) and lines[start_idx].strip() == "":
            start_idx += 1
        
        while end_idx >= 0 and lines[end_idx].strip() == "":
            end_idx -= 1
        
        if start_idx <= end_idx:
            cleaned_lines = lines[start_idx:end_idx + 1]
            return '\n'.join(cleaned_lines)
        
        return code
    
    def _categorize_pattern(self, pattern: ExtractedPattern) -> str:
        """Categorize the pattern."""
        if pattern.has_auth:
            return "authentication"
        elif pattern.has_database:
            return "database"
        elif pattern.has_validation:
            return "validation"
        else:
            return "endpoints"
    
    def _assess_difficulty(self, pattern: ExtractedPattern) -> str:
        """Assess difficulty level."""
        if pattern.complexity_score > 30:
            return "advanced"
        elif pattern.complexity_score > 15:
            return "intermediate"
        else:
            return "beginner"
    
    def _generate_tags(self, pattern: ExtractedPattern) -> List[str]:
        """Generate tags for pattern."""
        tags = ["fastapi", pattern.http_method]
        
        if pattern.has_validation:
            tags.append("pydantic")
        if pattern.has_auth:
            tags.append("authentication")
        if pattern.has_database:
            tags.append("database")
        
        return tags
    
    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples."""
        seen_outputs = set()
        unique_examples = []
        
        for example in examples:
            # Create a simple hash of the output
            output_hash = hash(example["output"])
            if output_hash not in seen_outputs:
                seen_outputs.add(output_hash)
                unique_examples.append(example)
        
        return unique_examples
    
    def mine_dataset(self, max_repos: int = 30, output_file: str = "data/fastapi_mined_dataset.json") -> List[Dict]:
        """Mine complete dataset from GitHub."""
        self.logger.info("🚀 Starting FastAPI dataset mining from GitHub...")
        
        # Step 1: Search for repositories
        repos = self.search_fastapi_repositories(max_repos)
        
        if not repos:
            self.logger.error("❌ No repositories found!")
            return []
        
        # Step 2: Mine each repository
        all_patterns = []
        for repo in repos:
            patterns = self.mine_repository(repo)
            all_patterns.extend(patterns)
            
            # Rate limiting between repos
            time.sleep(2)
        
        self.logger.info(f"⛏️  Total patterns extracted: {len(all_patterns)}")
        
        if not all_patterns:
            self.logger.error("❌ No patterns extracted!")
            return []
        
        # Step 3: Generate training examples
        examples = self.generate_training_examples(all_patterns)
        
        # Step 4: Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 Dataset saved to: {output_path}")
        
        # Step 5: Print statistics
        self._print_statistics(examples)
        
        return examples
    
    def _print_statistics(self, examples: List[Dict]):
        """Print dataset statistics."""
        categories = defaultdict(int)
        difficulties = defaultdict(int)
        repos = defaultdict(int)
        
        for example in examples:
            categories[example["category"]] += 1
            difficulties[example["difficulty"]] += 1
            repos[example["source_repo"]] += 1
        
        self.logger.info("\n📊 Dataset Statistics:")
        self.logger.info(f"📈 Total examples: {len(examples)}")
        self.logger.info(f"🏷️  Categories: {dict(categories)}")
        self.logger.info(f"💪 Difficulties: {dict(difficulties)}")
        self.logger.info(f"📚 Source repositories: {len(repos)}")
        self.logger.info(f"🔝 Top repos: {dict(list(repos.items())[:5])}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Mine FastAPI dataset from GitHub")
    parser.add_argument("--github_token", type=str, required=True, help="GitHub API token")
    parser.add_argument("--max_repos", type=int, default=30, help="Maximum repositories to mine")
    parser.add_argument("--output", type=str, default="data/fastapi_mined_dataset.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize miner
    miner = RealFastAPIMiner(args.github_token)
    
    # Mine dataset
    dataset = miner.mine_dataset(args.max_repos, args.output)
    
    if dataset:
        print(f"\n🎉 Successfully mined {len(dataset)} FastAPI examples from GitHub!")
        print(f"📁 Dataset saved to: {args.output}")
        print("🚀 Ready for fine-tuning!")
    else:
        print("❌ Failed to mine dataset")


if __name__ == "__main__":
    main()