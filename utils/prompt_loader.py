import os
from pathlib import Path

def load_prompt(filename: str) -> str:
    """
    Loads a static LLM prompt template from the centralized prompts/ directory.
    
    Args:
        filename (str): The filename including extension (e.g. 'domain_classifier.txt')
        
    Returns:
        str: Raw prompt string ready for f-string formatting.
    """
    prompt_dir = Path(__file__).parent.parent / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    
    prompt_path = prompt_dir / filename
    
    if not prompt_path.exists():
        raise FileNotFoundError(f"Critical prompt template missing from disk: {prompt_path}")
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()
