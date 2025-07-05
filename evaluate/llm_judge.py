"""
LLM-as-Judge FastAPI Evaluator

Uses GPT-4 to evaluate FastAPI code quality, correctness, and best practices.
Much more reliable than custom pattern matching.
"""

import json
import asyncio
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import openai
from openai import OpenAI


@dataclass
class FastAPITask:
    """FastAPI evaluation task."""
    instruction: str
    input: str
    category: str
    difficulty: str
    expected_features: List[str]  # What the code should include


class LLMJudgeEvaluator:
    """Evaluates FastAPI code using GPT-4 as judge."""
    
    def __init__(self, openai_api_key: str, output_dir: str = "outputs/fastapi_evaluation"):
        """Initialize with OpenAI API key."""
        self.client = OpenAI(api_key=openai_api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Load evaluation tasks
        self.tasks = self._create_evaluation_tasks()
        
    def _create_evaluation_tasks(self) -> List[FastAPITask]:
        """Create FastAPI evaluation tasks."""
        tasks = []
        
        # Basic endpoint
        tasks.append(FastAPITask(
            instruction="Create a FastAPI GET endpoint that returns user information by user ID",
            input="The endpoint should accept user_id as a path parameter and return user details",
            category="basic_endpoint",
            difficulty="beginner",
            expected_features=["GET decorator", "path parameter", "return statement", "async function"]
        ))
        
        # POST with Pydantic
        tasks.append(FastAPITask(
            instruction="Create a FastAPI POST endpoint for user registration with email validation",
            input="Use Pydantic models for request validation and return success/error response",
            category="pydantic_validation",
            difficulty="intermediate",
            expected_features=["POST decorator", "Pydantic BaseModel", "email validation", "error handling"]
        ))
        
        # Database integration
        tasks.append(FastAPITask(
            instruction="Create a FastAPI endpoint to get all users from database using dependency injection",
            input="Use SQLAlchemy session dependency and return list of users",
            category="database_integration",
            difficulty="intermediate",
            expected_features=["dependency injection", "database session", "SQLAlchemy query", "proper cleanup"]
        ))
        
        # Authentication
        tasks.append(FastAPITask(
            instruction="Create a protected FastAPI endpoint that requires JWT authentication",
            input="Use HTTPBearer security and validate JWT tokens before accessing protected resource",
            category="authentication",
            difficulty="advanced",
            expected_features=["HTTPBearer", "JWT validation", "protected decorator", "error handling"]
        ))
        
        # File upload
        tasks.append(FastAPITask(
            instruction="Create a FastAPI endpoint for file upload with validation",
            input="Accept image files only, validate file type and size, save to uploads directory",
            category="file_upload",
            difficulty="intermediate",
            expected_features=["UploadFile", "file validation", "file saving", "error handling"]
        ))
        
        # Background tasks
        tasks.append(FastAPITask(
            instruction="Create a FastAPI endpoint that processes data in background",
            input="Accept user data, immediately return response, process email sending in background",
            category="background_tasks",
            difficulty="advanced",
            expected_features=["BackgroundTasks", "background function", "immediate response", "task scheduling"]
        ))
        
        # Error handling
        tasks.append(FastAPITask(
            instruction="Create a FastAPI endpoint with proper error handling and custom exceptions",
            input="Handle different error types (404, 400, 500) with custom error responses",
            category="error_handling",
            difficulty="intermediate",
            expected_features=["HTTPException", "custom exceptions", "error responses", "status codes"]
        ))
        
        # Query parameters
        tasks.append(FastAPITask(
            instruction="Create a FastAPI endpoint with query parameters and pagination",
            input="Accept optional query parameters for filtering and pagination (page, limit, sort)",
            category="query_parameters",
            difficulty="intermediate",
            expected_features=["query parameters", "optional parameters", "pagination", "filtering"]
        ))
        
        return tasks
        
    def judge_code_quality(self, instruction: str, generated_code: str, expected_features: List[str]) -> Dict:
        """Use GPT-4 to judge FastAPI code quality."""
        
        judge_prompt = f"""
You are an expert FastAPI developer evaluating generated code. Please evaluate the following FastAPI code based on:

1. CORRECTNESS: Does the code correctly implement the requested functionality?
2. FASTAPI BEST PRACTICES: Does it follow FastAPI conventions and best practices?
3. CODE QUALITY: Is the code well-structured, readable, and maintainable?
4. COMPLETENESS: Does it include all expected features?

TASK: {instruction}

EXPECTED FEATURES: {', '.join(expected_features)}

GENERATED CODE:
```python
{generated_code}
```

Please provide your evaluation as a JSON object with the following structure:
{{
    "correctness_score": <1-10>,
    "best_practices_score": <1-10>,
    "code_quality_score": <1-10>,
    "completeness_score": <1-10>,
    "overall_score": <1-10>,
    "explanation": "Brief explanation of the evaluation",
    "missing_features": ["list of missing expected features"],
    "suggestions": ["list of improvement suggestions"]
}}

Be strict but fair in your evaluation. A score of 10 means perfect, 7-9 is good, 5-6 is acceptable, below 5 needs improvement.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert FastAPI developer and code reviewer."},
                    {"role": "user", "content": judge_prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # Parse JSON response
            response_text = response.choices[0].message.content
            
            # Extract JSON from response (handle cases where GPT includes extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            json_str = response_text[json_start:json_end]
            
            evaluation = json.loads(json_str)
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in LLM judging: {str(e)}")
            return {
                "correctness_score": 0,
                "best_practices_score": 0,
                "code_quality_score": 0,
                "completeness_score": 0,
                "overall_score": 0,
                "explanation": f"Evaluation failed: {str(e)}",
                "missing_features": [],
                "suggestions": []
            }
    
    def evaluate_model(
        self,
        model_name: str,
        adapter_path: Optional[str] = None,
        temperature: float = 0.1,
        max_new_tokens: int = 512
    ) -> Dict:
        """Evaluate model using LLM-as-judge."""
        
        self.logger.info(f"Evaluating model: {model_name}")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
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
            "task_results": [],
            "summary": {}
        }
        
        total_scores = {
            "correctness": [],
            "best_practices": [],
            "code_quality": [],
            "completeness": [],
            "overall": []
        }
        
        for i, task in enumerate(self.tasks):
            self.logger.info(f"Evaluating task {i+1}/{len(self.tasks)}: {task.category}")
            
            # Format prompt (same as your existing format)
            if task.input:
                prompt = f"[INST] {task.instruction}\n\nInput:\n{task.input} [/INST]"
            else:
                prompt = f"[INST] {task.instruction} [/INST]"
                
            # Generate code
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
                
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_code = generated_code[len(prompt):].strip()
            
            # Judge the code
            evaluation = self.judge_code_quality(
                task.instruction,
                generated_code,
                task.expected_features
            )
            
            # Store results
            task_result = {
                "task": task.__dict__,
                "generated_code": generated_code,
                "evaluation": evaluation
            }
            results["task_results"].append(task_result)
            
            # Accumulate scores
            total_scores["correctness"].append(evaluation["correctness_score"])
            total_scores["best_practices"].append(evaluation["best_practices_score"])
            total_scores["code_quality"].append(evaluation["code_quality_score"])
            total_scores["completeness"].append(evaluation["completeness_score"])
            total_scores["overall"].append(evaluation["overall_score"])
            
        # Calculate summary statistics
        results["summary"] = {
            "total_tasks": len(self.tasks),
            "average_correctness": sum(total_scores["correctness"]) / len(total_scores["correctness"]),
            "average_best_practices": sum(total_scores["best_practices"]) / len(total_scores["best_practices"]),
            "average_code_quality": sum(total_scores["code_quality"]) / len(total_scores["code_quality"]),
            "average_completeness": sum(total_scores["completeness"]) / len(total_scores["completeness"]),
            "average_overall": sum(total_scores["overall"]) / len(total_scores["overall"]),
            "score_distribution": {
                "excellent (8-10)": sum(1 for score in total_scores["overall"] if score >= 8),
                "good (6-7)": sum(1 for score in total_scores["overall"] if 6 <= score < 8),
                "needs_improvement (<6)": sum(1 for score in total_scores["overall"] if score < 6)
            }
        }
        
        # Save results
        results_file = self.output_dir / f"llm_judge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Evaluation complete! Results saved to {results_file}")
        return results
        
    def print_evaluation_summary(self, results: Dict):
        """Print evaluation summary."""
        summary = results["summary"]
        
        print("\n" + "="*60)
        print("🎯 FASTAPI SPECIALIST EVALUATION (LLM-as-Judge)")
        print("="*60)
        print(f"📊 Overall Score: {summary['average_overall']:.1f}/10")
        print(f"✅ Correctness: {summary['average_correctness']:.1f}/10")
        print(f"🏆 Best Practices: {summary['average_best_practices']:.1f}/10")
        print(f"💎 Code Quality: {summary['average_code_quality']:.1f}/10")
        print(f"📝 Completeness: {summary['average_completeness']:.1f}/10")
        print(f"🤖 Model: {results['model_name']}")
        if results['adapter_path']:
            print(f"🔧 Adapter: {results['adapter_path']}")
        print()
        print("📈 Score Distribution:")
        dist = summary['score_distribution']
        print(f"   Excellent (8-10): {dist['excellent (8-10)']}/{summary['total_tasks']}")
        print(f"   Good (6-7): {dist['good (6-7)']}/{summary['total_tasks']}")
        print(f"   Needs Improvement (<6): {dist['needs_improvement (<6)']}/{summary['total_tasks']}")
        print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate FastAPI specialist using LLM-as-judge")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--base_model", type=str, default="codellama/CodeLlama-7b-Instruct-hf", help="Base model")
    parser.add_argument("--adapter_path", type=str, help="Path to fine-tuned adapter")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    
    args = parser.parse_args()
    
    evaluator = LLMJudgeEvaluator(args.openai_api_key)
    
    results = evaluator.evaluate_model(
        model_name=args.base_model,
        adapter_path=args.adapter_path,
        temperature=args.temperature
    )
    
    evaluator.print_evaluation_summary(results)
