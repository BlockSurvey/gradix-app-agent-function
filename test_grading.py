"""Test script for grading functionality"""
import asyncio
import json
from controllers.grading_controller import GradingController
from agents.rubric_agent import RubricAgent


def test_rubric_agent():
    """Test the RubricAgent directly"""
    print("Testing RubricAgent...")
    agent = RubricAgent()
    
    test_rubric = agent.create_detailed_rubric(
        name="Essay Writing Assessment",
        grading_type="Essay",
        criteria="Grammar, structure, argument quality, use of evidence, and creativity"
    )
    
    print("Generated Rubric:")
    print(json.dumps(test_rubric, indent=2))
    print("-" * 50)


def test_grading_controller():
    """Test the GradingController"""
    print("Testing GradingController...")
    controller = GradingController()
    
    # Test data
    test_grading = {
        "name": "Research Paper Evaluation",
        "type": "Academic Paper",
        "criteria": "Research methodology, citation quality, argument coherence, writing clarity, and originality"
    }
    
    # Create grading
    result = controller.create_grading(test_grading)
    print(f"Create Grading Result: {json.dumps(result, indent=2, default=str)}")
    
    if result["success"]:
        grading_id = result["grading_id"]
        
        # Get grading
        get_result = controller.get_grading(grading_id)
        print(f"Get Grading Result: {json.dumps(get_result, indent=2, default=str)}")
        
        # Generate rubric
        rubric_result = controller.generate_rubric(grading_id)
        print(f"Generate Rubric Result: {json.dumps(rubric_result, indent=2, default=str)}")
        
        # List gradings
        list_result = controller.list_gradings(5)
        print(f"List Gradings Result: {json.dumps(list_result, indent=2, default=str)}")
    
    print("-" * 50)


if __name__ == "__main__":
    print("Starting Grading System Tests")
    print("=" * 50)
    
    # Test RubricAgent
    test_rubric_agent()
    
    # Test GradingController
    test_grading_controller()
    
    print("Tests completed!")