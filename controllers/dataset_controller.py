import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd
from fastapi import HTTPException, UploadFile

from controllers.grading_controller import GradingController
from database.firebase_db import firestore_client
from logger.bs_logger import bs_logger


class DatasetController:
    def __init__(self):
        self.db = firestore_client
        self.datasets_collection = "datasets"
        self.applications_subcollection = "applications"

    async def upload_dataset(self, file: UploadFile) -> Dict[str, Any]:
        """
        Upload and process CSV dataset file, storing dataset metadata and individual applications
        """
        try:
            # Validate file type
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="File must be a CSV")
            
            # Generate unique dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Read CSV file
            contents = await file.read()
            
            # Parse CSV using pandas
            try:
                df = pd.read_csv(pd.io.common.StringIO(contents.decode('utf-8')))
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
            
            # Get basic dataset information
            num_applications = len(df)
            columns = df.columns.tolist()
            
            # Dataset metadata
            dataset_metadata = {
                "id": dataset_id,
                "filename": file.filename,
                "file_size": len(contents),
                "content_type": file.content_type,
                "columns": columns,
                "num_columns": len(columns),
                "num_applications": num_applications,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store dataset metadata in Firebase
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_ref.set(dataset_metadata)
            
            # Store individual applications in subcollection
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            
            # Process each row and store as individual application
            for index, row in df.iterrows():
                application_id = str(uuid.uuid4())
                application_data = {
                    "id": application_id,
                    "row_index": index,
                    "data": row.to_dict(),
                    "created_at": datetime.utcnow()
                }
                applications_ref.document(application_id).set(application_data)
            
            bs_logger.info(f"Dataset {dataset_id} uploaded successfully with {num_applications} applications")
            
            return {
                "dataset_id": dataset_id,
                "message": "Dataset uploaded successfully",
                "metadata": dataset_metadata
            }
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error uploading dataset: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Get dataset metadata by ID
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_doc = dataset_ref.get()
            
            if not dataset_doc.exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            return dataset_doc.to_dict()
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error retrieving dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_datasets(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get list of all datasets with pagination
        """
        try:
            query = self.db.collection(self.datasets_collection).order_by("created_at", direction="DESCENDING")
            
            if offset > 0:
                # Get the document to start after for pagination
                docs = query.limit(offset).get()
                if docs:
                    query = query.start_after(docs[-1])
            
            datasets = query.limit(limit).get()
            
            return [doc.to_dict() for doc in datasets]
            
        except Exception as e:
            bs_logger.error(f"Error retrieving datasets: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def get_applications(self, dataset_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get applications from a specific dataset with pagination
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            
            # Check if dataset exists
            if not dataset_ref.get().exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            query = applications_ref.order_by("row_index")
            
            if offset > 0:
                docs = query.limit(offset).get()
                if docs:
                    query = query.start_after(docs[-1])
            
            applications = query.limit(limit).get()
            
            return [doc.to_dict() for doc in applications]
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error retrieving applications for dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def delete_dataset(self, dataset_id: str) -> Dict[str, str]:
        """
        Delete a dataset and all its applications
        """
        try:
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            
            # Check if dataset exists
            if not dataset_ref.get().exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            # Delete all applications in subcollection
            applications_ref = dataset_ref.collection(self.applications_subcollection)
            applications = applications_ref.get()
            
            for app_doc in applications:
                app_doc.reference.delete()
            
            # Delete the dataset document
            dataset_ref.delete()
            
            bs_logger.info(f"Dataset {dataset_id} deleted successfully")
            
            return {"message": f"Dataset {dataset_id} deleted successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error(f"Error deleting dataset {dataset_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    async def summarize_dataset(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset using only dataset metadata
        """
        try:
            # Get dataset metadata
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_doc = dataset_ref.get()
            
            if not dataset_doc.exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset_data = dataset_doc.to_dict()
            
            # Extract information from dataset metadata
            columns = dataset_data.get("columns", [])
            num_applications = dataset_data.get("num_applications", 0)
            filename = dataset_data.get("filename", "")
            
            if not columns:
                raise HTTPException(status_code=400, detail="No column information available in dataset")
            
            # Analyze columns based on column names only
            column_analysis = {}
            for column in columns:
                # Detect column purpose/category based on name
                column_lower = column.lower()
                analysis = {
                    "name": column,
                    "category": self._detect_column_category(column_lower)
                }
                column_analysis[column] = analysis
            
            # Detect dataset purpose based on column names
            column_names = [col.lower() for col in columns]
            dataset_purpose, confidence = self._detect_dataset_purpose(column_names)
            
            # Generate insights based on metadata
            insights = self._generate_metadata_insights(columns, num_applications, filename)
            
            summary = {
                "dataset_id": dataset_id,
                "basic_info": {
                    "filename": filename,
                    "total_applications": num_applications,
                    "total_columns": len(columns),
                    "created_at": dataset_data.get("created_at"),
                    "file_size": dataset_data.get("file_size"),
                    "content_type": dataset_data.get("content_type")
                },
                "purpose_analysis": {
                    "detected_purpose": dataset_purpose,
                    "confidence": confidence,
                    "description": self._get_purpose_description(dataset_purpose)
                },
                "column_analysis": column_analysis,
                "column_categories": self._categorize_columns(column_analysis),
                "insights": insights,
                "recommendations": self._generate_metadata_recommendations(dataset_purpose, column_analysis, insights)
            }
            
            bs_logger.info("Dataset %s summarized successfully using metadata only", dataset_id)
            return summary
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error("Error summarizing dataset %s: %s", dataset_id, str(e))
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def _detect_column_category(self, column_lower: str) -> str:
        """Detect column category based on column name"""
        if any(term in column_lower for term in ['id', 'uuid', 'key', 'reference']):
            return "identifier"
        elif any(term in column_lower for term in ['name', 'title', 'label', 'full_name', 'first_name', 'last_name']):
            return "personal_info"
        elif any(term in column_lower for term in ['email', 'mail', '@']):
            return "contact"
        elif any(term in column_lower for term in ['phone', 'mobile', 'tel', 'contact']):
            return "contact"
        elif any(term in column_lower for term in ['age', 'birth', 'dob', 'born']):
            return "demographic"
        elif any(term in column_lower for term in ['date', 'time', 'created', 'updated', 'timestamp']):
            return "temporal"
        elif any(term in column_lower for term in ['score', 'grade', 'rating', 'points', 'mark']):
            return "metric"
        elif any(term in column_lower for term in ['status', 'state', 'condition', 'approved', 'rejected']):
            return "status"
        elif any(term in column_lower for term in ['address', 'location', 'city', 'country', 'state', 'zip', 'postal']):
            return "location"
        elif any(term in column_lower for term in ['experience', 'skill', 'qualification', 'education', 'degree']):
            return "qualification"
        elif any(term in column_lower for term in ['salary', 'wage', 'pay', 'compensation', 'amount', 'price', 'cost']):
            return "financial"
        elif any(term in column_lower for term in ['description', 'comment', 'note', 'message', 'text', 'content']):
            return "text_content"
        else:
            return "general"
    
    def _detect_dataset_purpose(self, column_names: List[str]) -> tuple[str, float]:
        """Detect dataset purpose based on column names"""
        dataset_purpose = "general"
        confidence = 0.5
        
        # Application/Job-related dataset
        job_terms = ['name', 'email', 'experience', 'skill', 'qualification', 'resume', 'cv', 'position', 'role', 'salary']
        job_score = sum(1 for term in job_terms if any(term in col for col in column_names))
        
        # Student/Academic dataset
        academic_terms = ['student', 'grade', 'score', 'course', 'subject', 'gpa', 'university', 'school', 'education']
        academic_score = sum(1 for term in academic_terms if any(term in col for col in column_names))
        
        # Survey/Feedback dataset
        survey_terms = ['rating', 'feedback', 'response', 'survey', 'question', 'answer', 'satisfaction']
        survey_score = sum(1 for term in survey_terms if any(term in col for col in column_names))
        
        # Customer/User dataset
        customer_terms = ['customer', 'user', 'client', 'account', 'purchase', 'order', 'subscription']
        customer_score = sum(1 for term in customer_terms if any(term in col for col in column_names))
        
        # Employee/HR dataset
        hr_terms = ['employee', 'department', 'manager', 'hire', 'position', 'salary', 'performance']
        hr_score = sum(1 for term in hr_terms if any(term in col for col in column_names))
        
        # Determine the most likely purpose
        scores = {
            "job_applications": job_score,
            "academic_records": academic_score,
            "survey_responses": survey_score,
            "customer_data": customer_score,
            "hr_records": hr_score
        }
        
        max_score = max(scores.values())
        if max_score >= 3:
            dataset_purpose = max(scores, key=scores.get)
            confidence = min(0.9, 0.5 + (max_score * 0.1))
        elif max_score >= 2:
            dataset_purpose = max(scores, key=scores.get)
            confidence = 0.6 + (max_score * 0.05)
        
        return dataset_purpose, confidence
    
    def _categorize_columns(self, column_analysis: Dict) -> Dict[str, List[str]]:
        """Group columns by their categories"""
        categories = {}
        for col_name, analysis in column_analysis.items():
            category = analysis["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(col_name)
        return categories
    
    def _generate_metadata_insights(self, columns: List[str], num_applications: int, filename: str) -> List[str]:
        """Generate insights based on dataset metadata"""
        insights = []
        
        # Column count insights
        if len(columns) > 20:
            insights.append(f"Large dataset with {len(columns)} columns - consider focusing on key attributes")
        elif len(columns) < 5:
            insights.append(f"Compact dataset with {len(columns)} columns - may need additional data enrichment")
        
        # Application count insights
        if num_applications > 1000:
            insights.append(f"Large dataset with {num_applications:,} records - suitable for statistical analysis")
        elif num_applications < 100:
            insights.append(f"Small dataset with {num_applications} records - may need more data for robust analysis")
        
        # File name insights
        filename_lower = filename.lower()
        if any(term in filename_lower for term in ['test', 'sample', 'demo']):
            insights.append("Filename suggests this might be test or sample data")
        elif any(term in filename_lower for term in ['export', 'backup', 'archive']):
            insights.append("Filename suggests this is exported or archived data")
        
        # Column name patterns
        id_columns = [col for col in columns if 'id' in col.lower()]
        if len(id_columns) > 3:
            insights.append(f"Multiple ID columns detected: {', '.join(id_columns)} - may indicate relational data")
        
        return insights
    
    def _generate_metadata_recommendations(self, purpose: str, column_analysis: Dict, insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on metadata analysis"""
        recommendations = []
        
        # Purpose-specific recommendations
        if purpose == "job_applications":
            recommendations.append("Consider implementing automated screening based on qualification columns")
            recommendations.append("Analyze skill patterns to identify top candidate profiles")
        elif purpose == "academic_records":
            recommendations.append("Use for academic performance tracking and student success prediction")
            recommendations.append("Consider grade trend analysis and course completion correlation")
        elif purpose == "survey_responses":
            recommendations.append("Focus on satisfaction metrics and response pattern analysis")
            recommendations.append("Consider implementing sentiment analysis on text responses")
        elif purpose == "customer_data":
            recommendations.append("Implement customer segmentation based on behavioral patterns")
            recommendations.append("Consider building customer lifetime value models")
        elif purpose == "hr_records":
            recommendations.append("Use for employee performance analysis and retention prediction")
            recommendations.append("Consider compensation analysis and career progression tracking")
        
        # Column-based recommendations
        categories = self._categorize_columns(column_analysis)
        
        if "identifier" in categories:
            recommendations.append("Use identifier columns for data linking and deduplication")
        
        if "metric" in categories:
            recommendations.append("Focus on metric columns for performance analysis and scoring")
        
        if "temporal" in categories:
            recommendations.append("Leverage temporal columns for trend analysis and time-based insights")
        
        if "location" in categories:
            recommendations.append("Consider geographic analysis and location-based segmentation")
        
        # General recommendations
        recommendations.append("Implement data validation rules to ensure data quality")
        recommendations.append("Consider creating data visualizations to explore relationships")
        
        return recommendations

    async def get_ai_dataset_summary(self, dataset_id: str) -> Dict[str, Any]:
        """
        Generate AI-powered dataset summary and save it to Firebase
        """
        try:
            # Import the agent here to avoid circular imports
            from agents.dataset_agent import DatasetAnalysisAgent

            # Get dataset metadata
            dataset_ref = self.db.collection(self.datasets_collection).document(dataset_id)
            dataset_doc = dataset_ref.get()
            
            if not dataset_doc.exists:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            dataset_data = dataset_doc.to_dict()
            
            # Check if summary already exists
            if dataset_data.get("ai_summary"):
                bs_logger.info("Returning existing AI summary for dataset %s", dataset_id)
                return {
                    "dataset_id": dataset_id,
                    "summary": dataset_data["ai_summary"],
                    "summary_generated_at": dataset_data.get("summary_generated_at"),
                    "cached": True
                }
            
            # Generate new AI summary
            bs_logger.info("Generating new AI summary for dataset %s", dataset_id)
            
            # Initialize the AI agent
            analysis_agent = DatasetAnalysisAgent()
            
            # Generate AI-powered analysis
            ai_analysis = analysis_agent.analyze_dataset(dataset_data)
            
            # Extract the paragraph summary from analysis sections
            summary_paragraphs = []
            
            if ai_analysis.get("analysis"):
                sections = ai_analysis["analysis"]
                # Combine key sections into a coherent summary
                if sections.get("executive_summary"):
                    summary_paragraphs.append(sections["executive_summary"])
                if sections.get("dataset_classification"):
                    summary_paragraphs.append(sections["dataset_classification"])
                if sections.get("machine_learning"):
                    summary_paragraphs.append(sections["machine_learning"])
                if sections.get("business_value"):
                    summary_paragraphs.append(sections["business_value"])
            
            # Create final summary paragraph
            final_summary = " ".join(summary_paragraphs).strip()
            
            if not final_summary:
                # Use full response if sections are empty
                final_summary = ai_analysis.get("full_response", "Dataset analysis pending.")
            
            # Update Firebase document with the summary
            from datetime import datetime
            update_data = {
                "ai_summary": final_summary,
                "summary_generated_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            dataset_ref.update(update_data)
            
            bs_logger.info("AI summary saved to Firebase for dataset %s", dataset_id)
            
            return {
                "dataset_id": dataset_id,
                "summary": final_summary,
                "summary_generated_at": update_data["summary_generated_at"],
                "cached": False,
                "message": "Summary generated and saved successfully"
            }
            
        except HTTPException:
            raise
        except Exception as e:
            bs_logger.error("Error generating AI dataset summary for %s: %s", dataset_id, str(e))
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    def _get_purpose_description(self, purpose: str) -> str:
        """Get description for detected dataset purpose"""
        descriptions = {
            "job_applications": "This dataset appears to contain job application data with candidate information, qualifications, and experience details.",
            "academic_records": "This dataset appears to contain academic records with student information, grades, and course details.",
            "survey_responses": "This dataset appears to contain survey or feedback responses with ratings and questionnaire data.",
            "customer_data": "This dataset appears to contain customer or user information with account and interaction details.",
            "general": "This dataset contains general tabular data. More analysis may be needed to determine its specific purpose."
        }
        return descriptions.get(purpose, descriptions["general"])
    
    def _generate_recommendations(self, purpose: str, column_analysis: Dict, insights: List[str]) -> List[str]:
        """Generate actionable recommendations based on dataset analysis"""
        recommendations = []
        
        # Purpose-specific recommendations
        if purpose == "job_applications":
            recommendations.append("Consider using this dataset for candidate screening, skill analysis, or recruitment pipeline optimization.")
            recommendations.append("Look for patterns in successful applications to improve hiring criteria.")
        elif purpose == "academic_records":
            recommendations.append("This dataset could be used for academic performance analysis and student success prediction.")
            recommendations.append("Consider analyzing grade distributions and course completion rates.")
        elif purpose == "survey_responses":
            recommendations.append("Analyze response patterns and satisfaction trends to derive actionable insights.")
            recommendations.append("Consider sentiment analysis on text responses if available.")
        elif purpose == "customer_data":
            recommendations.append("Use this dataset for customer segmentation and behavior analysis.")
            recommendations.append("Consider building customer lifetime value models or churn prediction.")
        
        # Data quality recommendations
        missing_data_mentioned = any("missing values" in insight for insight in insights)
        if missing_data_mentioned:
            recommendations.append("Address missing data through imputation or data collection improvements.")
        
        # High cardinality recommendations
        high_cardinality_mentioned = any("cardinality" in insight for insight in insights)
        if high_cardinality_mentioned:
            recommendations.append("Consider feature engineering or dimensionality reduction for high cardinality columns.")
        
        # General recommendations
        recommendations.append("Consider data visualization to better understand distributions and relationships.")
        recommendations.append("Implement data validation rules to maintain data quality over time.")
        
        return recommendations


class EvaluationController:
    """Controller for handling evaluation operations that combine grading and dataset processing."""
    
    def __init__(self):
        self.db = firestore_client
        self.grading_controller = GradingController()
        self.dataset_controller = DatasetController()
        self.evaluations_collection = "evaluations"
        self.results_collection = "evaluation_results"
    
    async def create_evaluation(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new evaluation that links grading criteria with a dataset
        
        Args:
            evaluation_data: Dictionary containing grading_id, dataset_id, and evaluation settings
            
        Returns:
            Dictionary with created evaluation ID and data
        """
        try:
            # Validate required fields
            required_fields = ["grading_id", "dataset_id", "name"]
            for field in required_fields:
                if field not in evaluation_data:
                    return {
                        "success": False,
                        "error": f"Missing required field: {field}"
                    }
            
            # Verify grading exists
            grading_result = self.grading_controller.get_grading(evaluation_data["grading_id"])
            if not grading_result["success"]:
                return {
                    "success": False,
                    "error": "Invalid grading_id: " + grading_result.get("error", "Grading not found")
                }
            
            # Verify dataset exists
            try:
                dataset_info = await self.dataset_controller.get_dataset(evaluation_data["dataset_id"])
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid dataset_id: {str(e)}"
                }
            
            # Generate unique evaluation ID
            evaluation_id = str(uuid.uuid4())
            
            # Prepare evaluation document
            evaluation_doc = {
                "id": evaluation_id,
                "name": evaluation_data["name"],
                "description": evaluation_data.get("description", ""),
                "grading_id": evaluation_data["grading_id"],
                "dataset_id": evaluation_data["dataset_id"],
                "status": "created",
                "settings": evaluation_data.get("settings", {}),
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
                "grading_info": grading_result["data"],
                "dataset_info": {
                    "filename": dataset_info.get("filename"),
                    "num_applications": dataset_info.get("num_applications"),
                    "columns": dataset_info.get("columns"),
                    "ai_summary": dataset_info.get("ai_summary")
                }
            }
            
            # Store evaluation in Firebase
            self.db.collection(self.evaluations_collection).document(evaluation_id).set(evaluation_doc)
            
            bs_logger.info(f"Evaluation {evaluation_id} created successfully")
            
            return {
                "success": True,
                "evaluation_id": evaluation_id,
                "data": evaluation_doc
            }
            
        except Exception as e:
            bs_logger.error(f"Error creating evaluation: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Get evaluation by ID"""
        try:
            doc = self.db.collection(self.evaluations_collection).document(evaluation_id).get()
            if doc.exists:
                return {
                    "success": True,
                    "data": doc.to_dict(),
                    "evaluation_id": evaluation_id
                }
            else:
                return {
                    "success": False,
                    "error": "Evaluation not found"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_evaluations(self, limit: int = 10, status: str = None) -> Dict[str, Any]:
        """List evaluations with optional status filter"""
        try:
            query = self.db.collection(self.evaluations_collection)
            
            if status:
                query = query.where("status", "==", status)
            
            docs = query.order_by("created_at", direction="DESCENDING").limit(limit).stream()
            evaluations = []
            
            for doc in docs:
                evaluation_data = doc.to_dict()
                evaluation_data["evaluation_id"] = doc.id
                evaluations.append(evaluation_data)
            
            return {
                "success": True,
                "evaluations": evaluations
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def start_evaluation(self, evaluation_id: str) -> Dict[str, Any]:
        """Start processing applications through the evaluation"""
        try:
            # Get evaluation details
            evaluation_result = await self.get_evaluation(evaluation_id)
            if not evaluation_result["success"]:
                return evaluation_result
            
            evaluation_data = evaluation_result["data"]
            
            # # Check if evaluation is in correct status
            # if evaluation_data["status"] != "created":
            #     return {
            #         "success": False,
            #         "error": f"Cannot start evaluation in status: {evaluation_data['status']}"
            #     }
            
            # # Update evaluation status to processing
            # self.db.collection(self.evaluations_collection).document(evaluation_id).update({
            #     "status": "processing",
            #     "started_at": datetime.now(timezone.utc),
            #     "updated_at": datetime.now(timezone.utc)
            # })

            # Get the first application from the dataset
            dataset_id = evaluation_data["dataset_id"]
            applications = await self.dataset_controller.get_applications(dataset_id, limit=1)
            if applications:
                first_app = applications[0]
                # result = await self.process_single_application(
                #     evaluation_id, first_app, evaluation_data["grading_info"], evaluation_data["dataset_info"])
                return {
                    "success": True,
                    "evaluation_id": evaluation_id,
                    "status": "processing",
                    "total_applications": len(applications),
                    # "first_application_result": result
                }
            else:
                return {
                    "success": False,
                    "error": "No applications found in dataset"
                }
            
        except Exception as e:
            bs_logger.error(f"Error starting evaluation {evaluation_id}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _score_application_criterion(self, app_data: Dict, criterion_name: str, criterion_data: Dict) -> float:
        """Score a single criterion for an application"""
        # Simple keyword-based scoring
        criterion_keywords = criterion_name.lower().split()
        app_text = " ".join([str(v).lower() for v in app_data.values() if v is not None])
        
        # Count keyword matches
        matches = sum(1 for keyword in criterion_keywords if keyword in app_text)
        
        # Return score based on matches (0-5 scale)
        if matches >= len(criterion_keywords):
            return 5.0  # Excellent
        elif matches >= len(criterion_keywords) * 0.7:
            return 4.0  # Good
        elif matches >= len(criterion_keywords) * 0.5:
            return 3.0  # Satisfactory
        elif matches >= len(criterion_keywords) * 0.3:
            return 2.0  # Needs improvement
        else:
            return 1.0  # Inadequate
    
    def _basic_application_score(self, app_data: Dict, criteria: str) -> float:
        """Basic scoring when no detailed rubric is available"""
        # Count non-empty fields
        non_empty_fields = sum(1 for v in app_data.values() if v is not None and str(v).strip())
        total_fields = len(app_data)
        
        if total_fields == 0:
            return 1.0
        
        # Completeness score
        completeness = non_empty_fields / total_fields
        
        # Keywords from criteria
        if criteria:
            criteria_keywords = criteria.lower().split()
            app_text = " ".join([str(v).lower() for v in app_data.values() if v is not None])
            keyword_matches = sum(1 for keyword in criteria_keywords if keyword in app_text)
            keyword_score = min(keyword_matches / max(len(criteria_keywords), 1), 1.0)
        else:
            keyword_score = 0.5  # Neutral if no criteria
        
        # Combined score (weighted)
        return (completeness * 0.6 + keyword_score * 0.4) * 5  # Scale to 1-5
    
    def _calculate_grade(self, total_score: float) -> str:
        """Convert numeric score to letter grade"""
        if total_score >= 4.5:
            return "A"
        elif total_score >= 3.5:
            return "B"
        elif total_score >= 2.5:
            return "C"
        elif total_score >= 1.5:
            return "D"
        else:
            return "F"
    
    def _generate_feedback(self, total_score: float, scores: Dict[str, float]) -> str:
        """Generate feedback based on scores"""
        if total_score >= 4.0:
            return "Excellent application that meets or exceeds expectations."
        elif total_score >= 3.0:
            return "Good application with strong qualifications."
        elif total_score >= 2.0:
            return "Satisfactory application with room for improvement."
        else:
            return "Application needs significant improvement to meet requirements."
