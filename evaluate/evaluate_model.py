"""
Model evaluation module using lighteval for HumanEval benchmarking.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import torch
from datetime import datetime

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel
from lighteval.main_accelerate import main as lighteval_main
from lighteval.models.model_config import BaseModelConfig
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.logging.evaluation_tracker import EvaluationTracker


class ModelEvaluator:
    """Handles model evaluation using lighteval and HumanEval."""
    
    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        load_8bit: bool = False,
        output_dir: str = "outputs/evaluation"
    ):
        """
        Initialize the model evaluator.
        
        Args:
            base_model_name: Name or path of the base model
            adapter_path: Optional path to the fine-tuned LoRA adapter
            device: Device to run evaluation on
            load_8bit: Whether to load the model in 8-bit mode
            output_dir: Directory to save evaluation results
        """
        self.base_model_name = base_model_name
        self.adapter_path = adapter_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_8bit = load_8bit
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._load_model()
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load the model and tokenizer."""
        self.logger.info(f"Loading base model: {self.base_model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name,
            trust_remote_code=True,
            padding_side="left"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load base model
        model_kwargs = {
            "device_map": "auto" if self.device == "cuda" else None,
            "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            "trust_remote_code": True,
        }
        
        if self.load_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            **model_kwargs
        )
        
        # Load LoRA adapter if provided
        if self.adapter_path:
            self.logger.info(f"Loading LoRA adapter from: {self.adapter_path}")
            model = PeftModel.from_pretrained(model, self.adapter_path)
            
        model.eval()
        
        return model, tokenizer
        
    def generate_code(self, prompt: str, max_new_tokens: int = 256, temperature: float = 0.1) -> str:
        """
        Generate code completion for a given prompt.
        
        Args:
            prompt: Input prompt for code generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated code completion
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=False
        )
        
        if self.device == "cuda":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode and extract only the generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = generated_text[len(prompt):].strip()
        
        return completion
        
    def evaluate_humaneval(
        self,
        num_samples: int = 1,
        temperature: float = 0.1,
        max_new_tokens: int = 256
    ) -> Dict:
        """
        Evaluate the model on HumanEval benchmark.
        
        Args:
            num_samples: Number of samples to generate per problem
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting HumanEval evaluation...")
        
        try:
            # Import HumanEval dataset
            from datasets import load_dataset
            humaneval = load_dataset("openai_humaneval")["test"]
            
            results = []
            correct_solutions = 0
            total_problems = len(humaneval)
            
            for i, example in enumerate(humaneval):
                self.logger.info(f"Processing problem {i+1}/{total_problems}")
                
                prompt = example["prompt"]
                canonical_solution = example["canonical_solution"]
                test_cases = example["test"]
                
                # Generate solution
                generated_code = self.generate_code(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature
                )
                
                # Combine prompt and generated code
                full_solution = prompt + generated_code
                
                # Test the solution
                is_correct = self._test_solution(full_solution, test_cases)
                if is_correct:
                    correct_solutions += 1
                    
                results.append({
                    "task_id": example["task_id"],
                    "prompt": prompt,
                    "generated_code": generated_code,
                    "full_solution": full_solution,
                    "canonical_solution": canonical_solution,
                    "is_correct": is_correct,
                })
                
            # Calculate Pass@1 score
            pass_at_1 = correct_solutions / total_problems
            
            evaluation_results = {
                "pass_at_1": pass_at_1,
                "correct_solutions": correct_solutions,
                "total_problems": total_problems,
                "detailed_results": results,
                "model_info": {
                    "base_model": self.base_model_name,
                    "adapter_path": self.adapter_path,
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens
                }
            }
            
            # Save results
            results_file = self.output_dir / f"humaneval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
                
            self.logger.info(f"Evaluation complete! Pass@1: {pass_at_1:.3f}")
            self.logger.info(f"Results saved to: {results_file}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {str(e)}")
            raise
            
    def _test_solution(self, solution: str, test_cases: str) -> bool:
        """
        Test if a solution passes the given test cases.
        
        Args:
            solution: Generated code solution
            test_cases: Test cases to run
            
        Returns:
            True if solution passes all tests, False otherwise
        """
        try:
            # Create a safe execution environment
            exec_globals = {}
            
            # Execute the solution
            exec(solution, exec_globals)
            
            # Execute the test cases
            exec(test_cases, exec_globals)
            
            return True
            
        except Exception:
            return False
            
    def print_evaluation_summary(self, results: Dict):
        """Print a summary of evaluation results."""
        print("\n" + "="*60)
        print("🎯 HUMANEVAL EVALUATION RESULTS")
        print("="*60)
        print(f"📊 Pass@1 Score: {results['pass_at_1']:.3f} ({results['pass_at_1']*100:.1f}%)")
        print(f"✅ Correct Solutions: {results['correct_solutions']}/{results['total_problems']}")
        print(f"🤖 Model: {results['model_info']['base_model']}")
        if results['model_info']['adapter_path']:
            print(f"🔧 Adapter: {results['model_info']['adapter_path']}")
        print(f"🌡️  Temperature: {results['model_info']['temperature']}")
        print(f"📝 Max Tokens: {results['model_info']['max_new_tokens']}")
        print("="*60)


def main():
    """Main evaluation function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on HumanEval")
    parser.add_argument(
        "--base_model",
        type=str,
        default="codellama/CodeLlama-7b-Instruct-hf",
        help="Base model name or path"
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Path to fine-tuned LoRA adapter"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--load_8bit",
        action="store_true",
        help="Load model in 8-bit mode"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/evaluation",
        help="Directory to save evaluation results"
    )
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        base_model_name=args.base_model,
        adapter_path=args.adapter_path,
        device=args.device,
        load_8bit=args.load_8bit,
        output_dir=args.output_dir
    )
    
    # Run evaluation
    results = evaluator.evaluate_humaneval(
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )
    
    # Print summary
    evaluator.print_evaluation_summary(results)


if __name__ == "__main__":
    main()