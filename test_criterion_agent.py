"""
Test script for CriterionEvaluationAgent
"""
import asyncio
import json
from agents.criterion_evaluation_agent import CriterionEvaluationAgent

async def test_criterion_agent():
    """Test the criterion evaluation agent"""
    
    # Sample criterion configuration
    criterion_details = {
        "description": "Technical quality and implementation",
        "weight": 0.3,
        "levels": [
            {"level": 1, "description": "Poor implementation with major issues"},
            {"level": 2, "description": "Basic implementation with several issues"},
            {"level": 3, "description": "Satisfactory implementation meeting core requirements"},
            {"level": 4, "description": "Strong implementation with good practices"},
            {"level": 5, "description": "Exceptional implementation exceeding all requirements"}
        ]
    }
    
    # Sample application data
    application_data = {
        "id": "test-app-001",
        "name": "Sample Application",
        "description": "A test application for evaluation",
        "github_url": "https://github.com/example/repo",
        "website_url": "https://example.com",
        "technical_details": {
            "language": "Python",
            "framework": "FastAPI",
            "database": "PostgreSQL"
        }
    }
    
    # Sample dataset context
    dataset_context = {
        "requirements": [
            "Must have clean code architecture",
            "Should follow best practices",
            "Documentation should be comprehensive"
        ],
        "evaluation_focus": "Technical implementation quality"
    }
    
    try:
        print("Initializing CriterionEvaluationAgent...")
        agent = CriterionEvaluationAgent(
            criterion_name="Technical Execution",
            criterion_details=criterion_details
        )
        
        print(f"Evaluating criterion: {agent.criterion_name}")
        print("This may take a moment...")
        
        # Run the evaluation
        result = await agent.evaluate_criterion(
            application_data=application_data,
            dataset_context=dataset_context
        )
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        # Display results
        print(f"Criterion: {result.get('criterion')}")
        print(f"Score: {result.get('grade_score', 'N/A')}")
        print(f"Agent Type: {result.get('agent_type')}")
        print(f"Model Used: {result.get('model_used')}")
        
        if result.get('grade_evidence'):
            print("\nEvidence:")
            for i, evidence in enumerate(result.get('grade_evidence', []), 1):
                print(f"  {i}. {evidence}")
        
        if result.get('grade_feedback'):
            print(f"\nFeedback:\n{result.get('grade_feedback')}")
        
        if result.get('error'):
            print(f"\nError: {result.get('error')}")
            
        # Save full results to file
        with open('test_criterion_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print("\nFull results saved to test_criterion_results.json")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting CriterionEvaluationAgent test...")
    asyncio.run(test_criterion_agent())