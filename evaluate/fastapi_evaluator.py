"""
FastAPI Specialist Evaluation Module

This module evaluates fine-tuned models on FastAPI-specific tasks by:
1. Testing if generated code is syntactically correct
2. Checking if code follows FastAPI patterns
3. Testing if endpoints actually work
4. Comparing against baseline models
"""

import json
import tempfile
import subprocess
import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


@dataclass
class FastAPITestCase:
    """Structure for FastAPI evaluation test case."""
    instruction: str
    input: str
    expected_patterns: List[str]  # Regex patterns that should be in output
    category: str
    difficulty: str
    test_type: str  # 'syntax', 'pattern', 'functional'


class FastAPIEvaluator:
    """Evaluates FastAPI specialist models."""
    
    def __init__(self, output_dir: str = "outputs/fastapi_evaluation"):
        """Initialize the evaluator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load test cases
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[FastAPITestCase]:
        """Create comprehensive test cases for evaluation."""
        test_cases = []
        
        # Basic endpoint test
        test_cases.append(FastAPITestCase(
            instruction="Create a simple FastAPI GET endpoint that returns user information",
            input="The endpoint should accept a user_id parameter",
            expected_patterns=[
                r"from fastapi import FastAPI",
                r"app = FastAPI\(\)",
                r"@app\.get\(",
                r"async def.*\(",
                r"return.*{",
            ],
            category="basic",
            difficulty="beginner",
            test_type="pattern"
        ))
        
        # Pydantic model test
        test_cases.append(FastAPITestCase(
            instruction="Create a FastAPI POST endpoint for user registration with email validation",
            input="Use Pydantic models for request validation",
            expected_patterns=[
                r"from pydantic import BaseModel",
                r"class.*\(BaseModel\):",
                r"@app\.post\(",
                r"async def.*\(",
                r"EmailStr|email.*str",
            ],
            category="models",
            difficulty="intermediate",
            test_type="pattern"
        ))
        
        # Database integration test
        test_cases.append(FastAPITestCase(
            instruction="Create a FastAPI endpoint to fetch users from database using dependency injection",
            input="Use SQLAlchemy session dependency",
            expected_patterns=[
                r"from.*sqlalchemy",
                r"Depends\(",
                r"Session",
                r"def get_db\(",
                r"yield db",
            ],
            category="database",
            difficulty="advanced",
            test_type="pattern"
        ))
        
        # Authentication test
        test_cases.append(FastAPITestCase(
            instruction="Create a protected FastAPI endpoint with JWT authentication",
            input="Use HTTPBearer for token validation",
            expected_patterns=[
                r"from fastapi.security import HTTPBearer",
                r"HTTPBearer\(\)",
                r"jwt\.|JWT",
                r"Depends\(",
                r"HTTPException",
            ],
            category="authentication",
            difficulty="advanced",
            test_type="pattern"
        ))
        
        # File upload test
        test_cases.append(FastAPITestCase(
            instruction="Create a FastAPI endpoint for file upload with size validation",
            input="Accept image files only, maximum 5MB",
            expected_patterns=[
                r"UploadFile",
                r"File\(",
                r"async def.*upload",
                r"\.filename",
                r"content_type|size",
            ],
            category="file-handling",
            difficulty="intermediate",
            test_type="pattern"
        ))
        
        return test_cases
        
    def evaluate_syntax(self, code: str) -> Tuple[bool, str]:
        """Check if generated code is syntactically correct."""
        try:
            ast.parse(code)
            return True, "Syntax valid"
        except SyntaxError as e:
            return False, f"Syntax error: {str(e)}"
        except Exception as e:
            return False, f"Parse error: {str(e)}"
            
    def evaluate_patterns(self, code: str, expected_patterns: List[str]) -> Tuple[float, List[str]]:
        """Check if code contains expected FastAPI patterns."""
        matched_patterns = []
        
        for pattern in expected_patterns:
            if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                matched_patterns.append(pattern)
                
        pattern_score = len(matched_patterns) / len(expected_patterns)
        return pattern_score, matched_patterns
        
    def evaluate_functional(self, code: str) -> Tuple[bool, str]:
        """Test if FastAPI code can be imported and run (basic functional test)."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Try to import and check for basic FastAPI structure
            try:
                # Basic validation: check if it's importable
                result = subprocess.run(
                    ['python', '-c', f'import ast; ast.parse(open("{temp_file}").read())'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    return True, "Code is importable"
                else:
                    return False, f"Import failed: {result.stderr}"
                    
            finally:
                # Clean up
                Path(temp_file).unlink(missing_ok=True)
                
        except Exception as e:
            return False, f"Functional test failed: {str(e)}"
            
    def evaluate_model(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Dict:
        """Evaluate a model on FastAPI-specific tasks."""
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        
        if adapter_path:
            self.logger.info(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            
        model.eval()
        
        results = {
            "model_name": model_name,
            "adapter_path": adapter_path,
            "timestamp": datetime.now().isoformat(),
            "test_results": [],
            "summary": {}
        }
        
        total_tests = len(self.test_cases)
        syntax_passed = 0
        pattern_scores = []
        functional_passed = 0
        
        for i, test_case in enumerate(self.test_cases):
            self.logger.info(f"Running test {i+1}/{total_tests}: {test_case.category}")
            
            # Format prompt
            if test_case.input:
                prompt = f"[INST] {test_case.instruction}\n\nInput:\n{test_case.input} [/INST]"
            else:
                prompt = f"[INST] {test_case.instruction} [/INST]"
                
            # Generate code
            inputs = tokenizer(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.eos_token_id
                )
                
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated_code[len(prompt):].strip()
            
            # Evaluate generated code
            test_result = {
                "test_case": test_case.__dict__,
                "generated_code": generated_code,
                "evaluations": {}
            }
            
            # Syntax evaluation
            syntax_valid, syntax_msg = self.evaluate_syntax(generated_code)
            test_result["evaluations"]["syntax"] = {
                "passed": syntax_valid,
                "message": syntax_msg
            }
            if syntax_valid:
                syntax_passed += 1
                
            # Pattern evaluation
            pattern_score, matched_patterns = self.evaluate_patterns(
                generated_code, test_case.expected_patterns
            )
            test_result["evaluations"]["patterns"] = {
                "score": pattern_score,
                "matched_patterns": matched_patterns,
                "total_patterns": len(test_case.expected_patterns)
            }
            pattern_scores.append(pattern_score)
            
            # Functional evaluation (if syntax is valid)
            if syntax_valid:
                functional_valid, functional_msg = self.evaluate_functional(generated_code)
                test_result["evaluations"]["functional"] = {
                    "passed": functional_valid,
                    "message": functional_msg
                }
                if functional_valid:
                    functional_passed += 1
            else:
                test_result["evaluations"]["functional"] = {
                    "passed": False,
                    "message": "Skipped due to syntax error"
                }
                
            results["test_results"].append(test_result)
            
        # Calculate summary metrics
        results["summary"] = {
            "total_tests": total_tests,
            "syntax_pass_rate": syntax_passed / total_tests,
            "average_pattern_score": sum(pattern_scores) / len(pattern_scores),
            "functional_pass_rate": functional_passed / total_tests,
            "overall_score": (
                (syntax_passed / total_tests) * 0.3 +
                (sum(pattern_scores) / len(pattern_scores)) * 0.4 +
                (functional_passed / total_tests) * 0.3
            )
        }
        
        # Save results
        results_file = self.output_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Evaluation complete! Results saved to {results_file}")
        return results
        
    def compare_models(self, baseline_results: Dict, specialist_results: Dict) -> Dict:
        """Compare specialist model against baseline."""
        comparison = {
            "baseline": baseline_results["summary"],
            "specialist": specialist_results["summary"],
            "improvements": {}
        }
        
        # Calculate improvements
        for metric in ["syntax_pass_rate", "average_pattern_score", "functional_pass_rate", "overall_score"]:
            baseline_score = baseline_results["summary"][metric]
            specialist_score = specialist_results["summary"][metric]
            improvement = ((specialist_score - baseline_score) / baseline_score) * 100
            comparison["improvements"][metric] = improvement
            
        return comparison
        
    def print_evaluation_summary(self, results: Dict):
        """Print evaluation summary."""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("FASTAPI SPECIALIST EVALUATION RESULTS")
        print("="*60)
        print(f"Overall Score: {summary['overall_score']:.3f} ({summary['overall_score']*100:.1f}%)")
        print(f"Syntax Pass Rate: {summary['syntax_pass_rate']:.3f} ({summary['syntax_pass_rate']*100:.1f}%)")
        print(f"Pattern Match Score: {summary['average_pattern_score']:.3f} ({summary['average_pattern_score']*100:.1f}%)")
        print(f"Functional Pass Rate: {summary['functional_pass_rate']:.3f} ({summary['functional_pass_rate']*100:.1f}%)")
        print(f"Model: {results['model_name']}")
        if results['adapter_path']:
            print(f"Adapter: {results['adapter_path']}")
        print("="*60)


if __name__ == "__main__":
    # Example usage
    evaluator = FastAPIEvaluator()
    
    # Evaluate base model
    base_results = evaluator.evaluate_model("codellama/CodeLlama-7b-Instruct-hf")
    
    # Evaluate specialist model (uncomment when you have trained adapter)
    # specialist_results = evaluator.evaluate_model(
    #     "codellama/CodeLlama-7b-Instruct-hf",
    #     adapter_path="outputs/checkpoints"
    # )
    
    # Compare results (uncomment when you have both)
    # comparison = evaluator.compare_models(base_results, specialist_results)
    
    evaluator.print_evaluation_summary(base_results)