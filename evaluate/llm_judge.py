import os
import openai
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

openai.api_key = os.getenv('OPENAI_API_KEY')

@dataclass
class FastAPIReviewCriteria:
    """
    Criteria for evaluating FastAPI code.
    """
    routing_patterns: float = 0.0  # Proper use of routes, HTTP methods
    type_hints: float = 0.0  # Python type hints and Pydantic models
    error_handling: float = 0.0  # HTTPException, status codes
    input_validation: float = 0.0  # Request models, query/path params
    response_models: float = 0.0  # Response schemas, status codes
    dependency_injection: float = 0.0  # Depends, proper DI patterns
    documentation: float = 0.0  # Docstrings, OpenAPI/Swagger
    best_practices: float = 0.0  # FastAPI conventions
    total_score: float = 0.0
    suggestions: List[str] = None
    detailed_feedback: Dict[str, str] = field(default_factory=dict)
    criterion_weights: Dict[str, float] = field(default_factory=lambda: {
        'routing_patterns': 0.15,
        'type_hints': 0.15,
        'error_handling': 0.15,
        'input_validation': 0.15,
        'response_models': 0.10,
        'dependency_injection': 0.10,
        'documentation': 0.10,
        'best_practices': 0.10
    })

class GPTFastAPIJudge:
    """
    Uses GPT to evaluate FastAPI code quality.
    """
    
    def __init__(self):
        self.system_prompt = """You are an expert FastAPI code reviewer. You will evaluate code based on these criteria:

        ROUTING (15%): Path operations, HTTP methods, route organization
        TYPES (15%): Type hints, Pydantic models, type safety
        ERRORS (15%): HTTPException, status codes, error responses
        VALIDATION (15%): Request models, query/path params, input validation
        RESPONSES (10%): Response schemas, status codes, response typing
        DEPENDENCIES (10%): Depends usage, dependency patterns
        DOCS (10%): Docstrings, OpenAPI/Swagger docs, comments
        PRACTICES (10%): FastAPI conventions, code organization

        For each criterion, provide a score (0-100) and brief feedback.
        Format your response exactly as shown in the user's prompt."""

    def evaluate(self, prompt: str, code: str) -> FastAPIReviewCriteria:
        """
        Evaluates FastAPI code using GPT.
        """
        try:
            evaluation_prompt = f"""You are evaluating FastAPI code. For each criterion below, provide a score from 0 to 100 and a brief explanation.

            Code to evaluate:
            ```python
            {code}
            ```

            Format your response EXACTLY like this (including the exact headings):

            ROUTING (Score: X/100):
            Brief feedback here

            TYPES (Score: X/100):
            Brief feedback here

            ERRORS (Score: X/100):
            Brief feedback here

            VALIDATION (Score: X/100):
            Brief feedback here

            RESPONSES (Score: X/100):
            Brief feedback here

            DEPENDENCIES (Score: X/100):
            Brief feedback here

            DOCS (Score: X/100):
            Brief feedback here

            PRACTICES (Score: X/100):
            Brief feedback here

            SUGGESTIONS:
            1. First suggestion
            2. Second suggestion
            """

            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": evaluation_prompt}
            ]

            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1,
                max_tokens=1000
            )

            evaluation_text = response.choices[0].message.content

            result = FastAPIReviewCriteria()
            result.suggestions = []

            lines = evaluation_text.split('\n')
            current_section = None
            feedback_buffer = []
            
            criteria_map = {
                'ROUTING': 'routing_patterns',
                'TYPES': 'type_hints',
                'ERRORS': 'error_handling',
                'VALIDATION': 'input_validation',
                'RESPONSES': 'response_models',
                'DEPENDENCIES': 'dependency_injection',
                'DOCS': 'documentation',
                'PRACTICES': 'best_practices'
            }
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                section_found = False
                for header, attr_name in criteria_map.items():
                    if line.startswith(header):
                        section_found = True

                        if current_section and feedback_buffer:
                            result.detailed_feedback[current_section] = ' '.join(feedback_buffer)
                            feedback_buffer = []
                        
                        current_section = attr_name

                        try:
                            score_start = line.find("Score:") + 6
                            score_end = line.find("/")
                            if score_start > 6 and score_end > score_start:
                                score = float(line[score_start:score_end].strip())
                                setattr(result, current_section, score)
                        except (ValueError, AttributeError):
                            pass
                        break
                
                if not section_found:
                    if current_section and not line.startswith("SUGGESTIONS:"):

                        if not any(line.startswith(h) for h in criteria_map.keys()):
                            feedback_buffer.append(line)
                    elif line.startswith("SUGGESTIONS:"):

                        if current_section and feedback_buffer:
                            result.detailed_feedback[current_section] = ' '.join(feedback_buffer)
                        current_section = None
                        feedback_buffer = []
                    elif line.startswith(("1.", "2.", "3.")):

                        suggestion = line.lstrip("123. ").strip()
                        if suggestion and suggestion != "First suggestion" and suggestion != "Second suggestion":
                            result.suggestions.append(suggestion)

            if current_section and feedback_buffer:
                result.detailed_feedback[current_section] = ' '.join(feedback_buffer)

            weighted_sum = 0
            for criterion, weight in result.criterion_weights.items():
                criterion_score = getattr(result, criterion, 0)
                weighted_sum += criterion_score * weight
                
            result.total_score = weighted_sum
            
            return result
            
        except Exception as e:
            result = FastAPIReviewCriteria()
            result.detailed_feedback = f"Evaluation failed: {str(e)}"
            result.score = 0
            result.suggestions = ["Could not complete evaluation"]
            return result

def format_gpt_evaluation(result: FastAPIReviewCriteria) -> str:
    """
    Formats the GPT evaluation result into a readable string.
    """
    output = []
    output.append("GPT FastAPI Code Review")
    output.append("=" * 40)

    output.append(f"Overall Score: {result.total_score:.1f}/100")

    output.append("\nCriteria Scores:")
    criteria = {
        "Routing Patterns": result.routing_patterns,
        "Type Hints": result.type_hints,
        "Error Handling": result.error_handling,
        "Input Validation": result.input_validation,
        "Response Models": result.response_models,
        "Dependency Injection": result.dependency_injection,
        "Documentation": result.documentation,
        "Best Practices": result.best_practices
    }
    
    for criterion_name, score in criteria.items():
        check_mark = "✓" if score >= 70 else "✗"
        output.append(f"{check_mark} {criterion_name}: {score:.1f}/100 (Weight: {result.criterion_weights[criterion_name.lower().replace(' ', '_')]:.2f})")
        if criterion_name.lower().replace(' ', '_') in result.detailed_feedback:
            output.append(f"   └─ {result.detailed_feedback[criterion_name.lower().replace(' ', '_')]}")

    if result.suggestions:
        output.append("\nSuggested Improvements:")
        for i, suggestion in enumerate(result.suggestions, 1):
            output.append(f"{i}. {suggestion}")
    
    return "\n".join(output)