#!/usr/bin/env python3
"""
Test script to verify dataset enhancement is working correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.prepare_dataset import DatasetPreparator
import json

def test_dataset_enhancement():
    """Test the dataset enhancement functionality."""
    print("Testing Dataset Enhancement...")
    print("=" * 50)
    
    # Initialize the preparator
    preparator = DatasetPreparator()
    
    # Test cases for enhancement
    test_cases = [
        {
            "name": "Basic endpoint without imports",
            "code": '''@app.get("/")
def root():
    return {"message": "Hello, World!"}'''
        },
        {
            "name": "Endpoint with HTTPException",
            "code": '''@app.get("/users/{user_id}")
def get_user(user_id: int):
    if user_id < 0:
        raise HTTPException(status_code=404, detail="User not found")
    return {"user_id": user_id}'''
        },
        {
            "name": "Endpoint with Session dependency",
            "code": '''@app.get("/projects/{org_id}")
def list_projects(org_id: int, db: Session = Depends(get_db)):
    projects = db.query(Project).filter(Project.org_id == org_id).all()
    return projects'''
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}:")
        print("Original code:")
        print("-" * 30)
        print(test_case['code'])
        print("-" * 30)
        
        enhanced = preparator.enhance_code_snippet(test_case['code'])
        print("Enhanced code:")
        print("-" * 30)
        print(enhanced)
        print("-" * 30)
        
        # Check if enhancements are correct
        checks = {
            "Has FastAPI import": "from fastapi import" in enhanced,
            "Has app instance": "app = FastAPI()" in enhanced,
            "Has HTTPException": "HTTPException" in enhanced,
            "Has Session import": "from sqlalchemy.orm import Session" in enhanced if "Session" in test_case['code'] else True
        }
        
        print("Checks:")
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
        
        print("\n" + "=" * 50)

def test_actual_dataset():
    """Test the actual dataset loading and enhancement."""
    print("Testing Actual Dataset...")
    print("=" * 50)
    
    preparator = DatasetPreparator()
    
    # Load first few examples
    with open("data/data/fastapi_mined_dataset.json", 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    print(f"Total examples in dataset: {len(raw_data)}")
    
    # Test first 3 examples
    for i in range(min(3, len(raw_data))):
        example = raw_data[i]
        print(f"\nExample {i+1}:")
        print(f"Instruction: {example['instruction']}")
        print(f"Category: {example['category']}")
        print(f"Difficulty: {example['difficulty']}")
        
        print("Original output:")
        print("-" * 30)
        print(example['output'])
        print("-" * 30)
        
        enhanced = preparator.enhance_code_snippet(example['output'])
        print("Enhanced output:")
        print("-" * 30)
        print(enhanced)
        print("-" * 30)
        
        # Check quality
        has_imports = "from fastapi import" in enhanced
        has_app = "app = FastAPI()" in enhanced
        has_endpoint = "@app." in enhanced or "@router." in enhanced
        
        print(f"Quality checks:")
        print(f"  ✓ Has imports: {has_imports}")
        print(f"  ✓ Has app instance: {has_app}")
        print(f"  ✓ Has endpoint: {has_endpoint}")
        
        print("\n" + "=" * 50)

if __name__ == "__main__":
    test_dataset_enhancement()
    test_actual_dataset() 