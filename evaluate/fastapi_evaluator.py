"""
FastAPI Code Evaluator

This module evaluates FastAPI code by checking:
1. Syntax correctness
2. FastAPI patterns and best practices
3. Code structure and organization
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

class FastAPIBestPractices(Enum):
    """
    Enum for FastAPI best practices patterns.
    """
    PYDANTIC_MODELS = r"class\s+\w+(\w*Model|\w*Schema|\w*Request|\w*Response)"
    RESPONSE_MODEL = r"@\w+\.\w+\([^)]*response_model\s*="
    STATUS_CODES = r"(status_code\s*=|status\.\w+)"
    DEPENDENCIES = r"Depends\([^)]+\)"
    ASYNC_DEF = r"async\s+def"
    PATH_PARAMS = r"{[^}]+}"
    QUERY_PARAMS = r"(\w+\s*:\s*Optional|\w+\s*=\s*Query\()"
    BODY_MODELS = r"(\w+\s*:\s*\w+Model|\w+\s*:\s*\w+Schema)"

def extract_code_from_markdown(text: str) -> str:
    """
    Extract Python code from markdown-formatted text with code blocks.
    """
    code_blocks = re.findall(r'```(?:python)?(.*?)```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()

    if 'from fastapi import' in text:
        code = text[text.find('from fastapi import'):]
        if 'This endpoint' in code:
            code = code[:code.find('This endpoint')]
        return code.strip()
    return text.strip()

@dataclass
class EvaluationResult:
    """
    Stores the evaluation results for a single test case.
    """
    prompt: str
    response: str
    is_valid_python: bool = False
    has_imports: bool = False
    has_router: bool = False
    has_endpoint: bool = False
    has_type_hints: bool = False
    has_docstring: bool = False
    has_error_handling: bool = False
    has_pydantic_models: bool = False
    has_response_model: bool = False
    has_status_codes: bool = False
    has_dependencies: bool = False
    has_async_def: bool = False
    has_path_params: bool = False
    has_query_params: bool = False
    has_body_models: bool = False
    required_imports: Set[str] = field(default_factory=set)
    missing_imports: Set[str] = field(default_factory=set)
    extracted_endpoints: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    error_message: Optional[str] = None
    
    def calculate_score(self) -> float:
        """
        Calculate weighted score based on various criteria.
        """
        weights = {
            'is_valid_python': 1.0,
            'has_imports': 0.8,
            'has_router': 0.8,
            'has_endpoint': 1.0,
            'has_type_hints': 0.7,
            'has_docstring': 0.3,
            'has_error_handling': 0.9,
            'has_pydantic_models': 0.6,
            'has_response_model': 0.6,
            'has_status_codes': 0.7,
            'has_dependencies': 0.5,
            'has_async_def': 0.3,
            'has_path_params': 0.4,
            'has_query_params': 0.4,
            'has_body_models': 0.5
        }
        
        total_weight = sum(weights.values())
        weighted_sum = sum(
            weights[attr] * getattr(self, attr)
            for attr in weights.keys()
        )
        
        return weighted_sum / total_weight

class FastAPIEvaluator:
    """
    Evaluates FastAPI code generation responses.
    """
    
    def __init__(self):
        self.import_patterns = {
            "fastapi": [r"from\s+fastapi\s+import", r"import\s+fastapi"],
            "FastAPI": [r"FastAPI"],
            "APIRouter": [r"APIRouter"],
            "HTTPException": [r"HTTPException"],
            "status": [r"status\.HTTP_[0-9]+"],
            "Response": [r"Response"],
            "Request": [r"Request"],
            "Depends": [r"Depends"],
            "Body": [r"Body"],
            "Query": [r"Query"],
            "Path": [r"Path"],
            "BaseModel": [r"BaseModel"],
        }
    
    def evaluate_response(self, prompt: str, response: str) -> EvaluationResult:
        """
        Evaluates a single response.
        """
        code = extract_code_from_markdown(response)
        result = EvaluationResult(prompt=prompt, response=code)
        
        try:
            ast.parse(code)
            result.is_valid_python = True
        except SyntaxError as e:
            result.error_message = f"Invalid Python syntax: {str(e)}"
            return result

        for import_name, patterns in self.import_patterns.items():
            if any(re.search(pattern, code) for pattern in patterns):
                result.required_imports.add(import_name)
            else:
                result.missing_imports.add(import_name)
        result.has_imports = len(result.required_imports) > 0

        result.has_router = bool(re.search(r"(app\s*=\s*FastAPI\(\)|router\s*=\s*APIRouter\(\))", code))

        endpoint_pattern = r"@\s*(app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]\s*[,)]"
        endpoints = re.finditer(endpoint_pattern, code)
        result.extracted_endpoints = []
        
        for match in endpoints:
            decorator, method, path = match.groups()
            result.extracted_endpoints.append({
                "decorator": decorator,
                "method": method.upper(),
                "path": path
            })
        
        result.has_endpoint = len(result.extracted_endpoints) > 0

        type_hint_pattern = r"def\s+\w+\s*\([^)]*:\s*\w+[\[\],\s]*\w*"
        result.has_type_hints = bool(re.search(type_hint_pattern, code))

        docstring_pattern = r'("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')'
        result.has_docstring = bool(re.search(docstring_pattern, code))

        error_patterns = [
            r"HTTPException",
            r"try\s*:",
            r"raise\s+\w+",
            r"status\.HTTP_[45]\d\d",
            r"status_code\s*=\s*[45]\d\d"
        ]
        result.has_error_handling = any(bool(re.search(pattern, code)) for pattern in error_patterns)

        for practice in FastAPIBestPractices:
            attr_name = f"has_{practice.name.lower()}"
            if hasattr(result, attr_name):
                setattr(result, attr_name, bool(re.search(practice.value, code)))

        result.score = result.calculate_score()
        
        return result
    
    def format_evaluation_result(self, result: EvaluationResult) -> str:
        """
        Formats the evaluation result into a readable string.
        """
        output = []
        output.append("Evaluation Results:")

        output.append("\nBasic Checks:")
        output.append(f"✓ Valid Python: {result.is_valid_python}")
        if result.error_message:
            output.append(f"Error: {result.error_message}")
        output.append(f"✓ Has Router/App: {result.has_router}")

        output.append("\nImport Analysis:")
        output.append(f"✓ Has Required Imports: {result.has_imports}")
        if result.required_imports:
            output.append("  Found imports:")
            for imp in sorted(result.required_imports):
                output.append(f"    • {imp}")
        if result.missing_imports:
            output.append("  Missing recommended imports:")
            for imp in sorted(result.missing_imports):
                output.append(f"    • {imp}")

        output.append("\nEndpoint Analysis:")
        output.append(f"✓ Has Endpoints: {result.has_endpoint}")
        if result.extracted_endpoints:
            output.append("  Endpoints found:")
            for endpoint in result.extracted_endpoints:
                output.append(f"    • {endpoint['method']} {endpoint['path']}")

        output.append("\nCode Quality:")
        output.append(f"✓ Has Type Hints: {result.has_type_hints}")
        output.append(f"✓ Has Docstrings: {result.has_docstring}")
        output.append(f"✓ Has Error Handling: {result.has_error_handling}")

        output.append("\nFastAPI Best Practices:")
        output.append(f"✓ Uses Pydantic Models: {result.has_pydantic_models}")
        output.append(f"✓ Specifies Response Models: {result.has_response_model}")
        output.append(f"✓ Uses Status Codes: {result.has_status_codes}")
        output.append(f"✓ Uses Dependencies: {result.has_dependencies}")
        output.append(f"✓ Uses Async Functions: {result.has_async_def}")
        output.append(f"✓ Has Path Parameters: {result.has_path_params}")
        output.append(f"✓ Has Query Parameters: {result.has_query_params}")
        output.append(f"✓ Uses Request/Response Models: {result.has_body_models}")

        output.append(f"\nOverall Score: {result.score:.2%}")
        
        return "\n".join(output)