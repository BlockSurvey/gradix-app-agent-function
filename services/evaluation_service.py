# """
# Evaluation Service for processing applications through grading criteria
# """

# import asyncio
# from datetime import datetime, timezone
# from typing import Dict, Any, List
# from concurrent.futures import ThreadPoolExecutor, as_completed

# from database.firebase_db import firestore_client
# from logger.bs_logger import bs_logger
# from controllers.dataset_controller import EvaluationController


# class EvaluationService:
#     """Service for handling evaluation operations and batch processing."""
    
#     def __init__(self):
#         self.db = firestore_client
#         self.evaluation_controller = EvaluationController()
#         self.max_workers = 5  # Limit concurrent processing
    
#     async def process_evaluation_batch(self, evaluation_id: str, batch_size: int = 50) -> Dict[str, Any]:
#         """
#         Process a batch of applications for an evaluation
        
#         Args:
#             evaluation_id: ID of the evaluation to process
#             batch_size: Number of applications to process in each batch
            
#         Returns:
#             Dictionary with processing results and statistics
#         """
#         try:
#             # Get evaluation details
#             evaluation_result = self.evaluation_controller.get_evaluation(evaluation_id)
#             if not evaluation_result["success"]:
#                 return {
#                     "success": False,
#                     "error": evaluation_result["error"]
#                 }
            
#             evaluation_data = evaluation_result["data"]
            
#             # Get all applications from dataset
#             dataset_id = evaluation_data["dataset_id"]
#             applications = await self.evaluation_controller.dataset_controller.get_applications(
#                 dataset_id, limit=1000
#             )
            
#             if not applications:
#                 return {
#                     "success": False,
#                     "error": "No applications found in dataset"
#                 }
            
#             total_applications = len(applications)
#             processed_count = 0
#             successful_count = 0
#             failed_count = 0
#             results = []
            
#             bs_logger.info(f"Starting batch processing for evaluation {evaluation_id} with {total_applications} applications")
            
#             # Process applications in batches
#             for i in range(0, total_applications, batch_size):
#                 batch = applications[i:i + batch_size]
#                 batch_results = await self._process_application_batch(
#                     evaluation_id, 
#                     batch, 
#                     evaluation_data["grading_info"]
#                 )
                
#                 # Update counters
#                 processed_count += len(batch)
#                 for result in batch_results:
#                     if "error" not in result:
#                         successful_count += 1
#                     else:
#                         failed_count += 1
#                     results.append(result)
                
#                 # Update evaluation progress
#                 progress = (processed_count / total_applications) * 100
#                 self._update_evaluation_progress(evaluation_id, progress, processed_count)
                
#                 bs_logger.info(f"Processed batch {i//batch_size + 1}: {processed_count}/{total_applications} applications")
                
#                 # Small delay between batches to prevent overwhelming the system
#                 await asyncio.sleep(0.1)
            
#             # Update final evaluation status
#             final_status = "completed" if failed_count == 0 else "completed_with_errors"
#             self._update_evaluation_status(evaluation_id, final_status, {
#                 "total_processed": processed_count,
#                 "successful": successful_count,
#                 "failed": failed_count,
#                 "completion_rate": (successful_count / processed_count) * 100 if processed_count > 0 else 0
#             })
            
#             return {
#                 "success": True,
#                 "evaluation_id": evaluation_id,
#                 "statistics": {
#                     "total_applications": total_applications,
#                     "processed": processed_count,
#                     "successful": successful_count,
#                     "failed": failed_count,
#                     "success_rate": (successful_count / processed_count) * 100 if processed_count > 0 else 0
#                 },
#                 "status": final_status,
#                 "results": results[:10]  # Return first 10 results as sample
#             }
            
#         except Exception as e:
#             bs_logger.error(f"Error in batch processing for evaluation {evaluation_id}: {str(e)}")
#             self._update_evaluation_status(evaluation_id, "failed", {"error": str(e)})
#             return {
#                 "success": False,
#                 "error": str(e)
#             }
    
#     async def _process_application_batch(
#         self, 
#         evaluation_id: str, 
#         applications: List[Dict], 
#         grading_info: Dict
#     ) -> List[Dict[str, Any]]:
#         """Process a batch of applications concurrently"""
        
#         # Use ThreadPoolExecutor for concurrent processing
#         with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
#             # Submit all applications for processing
#             future_to_app = {}
#             for app in applications:
#                 future = executor.submit(
#                     self._process_single_application_sync,
#                     evaluation_id,
#                     app,
#                     grading_info
#                 )
#                 future_to_app[future] = app
            
#             # Collect results as they complete
#             results = []
#             for future in as_completed(future_to_app):
#                 try:
#                     result = future.result(timeout=30)  # 30 second timeout per application
#                     results.append(result)
#                 except Exception as e:
#                     app = future_to_app[future]
#                     error_result = {
#                         "evaluation_id": evaluation_id,
#                         "application_id": app.get("id"),
#                         "error": str(e),
#                         "processed_at": datetime.now(timezone.utc)
#                     }
#                     results.append(error_result)
#                     bs_logger.error(f"Error processing application {app.get('id')}: {str(e)}")
            
#             return results
    
#     def _process_single_application_sync(self, evaluation_id: str, application: Dict, grading_info: Dict) -> Dict[str, Any]:
#         """Synchronous wrapper for processing single application"""
#         try:
#             # Use the evaluation controller's processing logic
#             return asyncio.run(
#                 self.evaluation_controller.process_single_application(
#                     evaluation_id, application, grading_info
#                 )
#             )
#         except Exception as e:
#             return {
#                 "evaluation_id": evaluation_id,
#                 "application_id": application.get("id"),
#                 "error": str(e),
#                 "processed_at": datetime.now(timezone.utc)
#             }
    
#     def _update_evaluation_progress(self, evaluation_id: str, progress: float, processed_count: int):
#         """Update evaluation progress in database"""
#         try:
#             self.db.collection("evaluations").document(evaluation_id).update({
#                 "progress": progress,
#                 "processed_count": processed_count,
#                 "updated_at": datetime.now(timezone.utc)
#             })
#         except Exception as e:
#             bs_logger.error(f"Error updating evaluation progress: {str(e)}")
    
#     def _update_evaluation_status(self, evaluation_id: str, status: str, metadata: Dict = None):
#         """Update evaluation status and metadata"""
#         try:
#             update_data = {
#                 "status": status,
#                 "updated_at": datetime.now(timezone.utc)
#             }
            
#             if status in ["completed", "completed_with_errors"]:
#                 update_data["completed_at"] = datetime.now(timezone.utc)
            
#             if metadata:
#                 update_data["metadata"] = metadata
            
#             self.db.collection("evaluations").document(evaluation_id).update(update_data)
            
#         except Exception as e:
#             bs_logger.error(f"Error updating evaluation status: {str(e)}")
    
#     def get_evaluation_statistics(self, evaluation_id: str) -> Dict[str, Any]:
#         """Get comprehensive statistics for an evaluation"""
#         try:
#             # Get evaluation info
#             evaluation_result = self.evaluation_controller.get_evaluation(evaluation_id)
#             if not evaluation_result["success"]:
#                 return evaluation_result
            
#             evaluation_data = evaluation_result["data"]
            
#             # Get all results for this evaluation
#             results = self.evaluation_controller.get_evaluation_results(evaluation_id, limit=10000)
            
#             if not results["success"]:
#                 return results
            
#             result_data = results["results"]
            
#             if not result_data:
#                 return {
#                     "success": True,
#                     "evaluation_id": evaluation_id,
#                     "statistics": {
#                         "total_results": 0,
#                         "status": evaluation_data.get("status", "unknown")
#                     }
#                 }
            
#             # Calculate statistics
#             total_results = len(result_data)
#             scores = [r.get("total_score", 0) for r in result_data if "total_score" in r]
#             grades = [r.get("grade", "") for r in result_data if "grade" in r]
            
#             # Grade distribution
#             grade_distribution = {}
#             for grade in grades:
#                 grade_distribution[grade] = grade_distribution.get(grade, 0) + 1
            
#             # Score statistics
#             score_stats = {}
#             if scores:
#                 score_stats = {
#                     "average": sum(scores) / len(scores),
#                     "minimum": min(scores),
#                     "maximum": max(scores),
#                     "count": len(scores)
#                 }
            
#             return {
#                 "success": True,
#                 "evaluation_id": evaluation_id,
#                 "evaluation_name": evaluation_data.get("name", ""),
#                 "statistics": {
#                     "total_results": total_results,
#                     "status": evaluation_data.get("status", "unknown"),
#                     "progress": evaluation_data.get("progress", 0),
#                     "processed_count": evaluation_data.get("processed_count", 0),
#                     "score_statistics": score_stats,
#                     "grade_distribution": grade_distribution,
#                     "created_at": evaluation_data.get("created_at"),
#                     "started_at": evaluation_data.get("started_at"),
#                     "completed_at": evaluation_data.get("completed_at")
#                 }
#             }
            
#         except Exception as e:
#             bs_logger.error(f"Error getting evaluation statistics: {str(e)}")
#             return {
#                 "success": False,
#                 "error": str(e)
#             }