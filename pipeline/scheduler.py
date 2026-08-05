"""
Automated pipeline scheduler using APScheduler.

This module handles:
- Periodic model retraining
- Regular inference on new data
- Automated report generation
- Scheduled LLM insight generation
"""

import logging
from datetime import datetime
from typing import Callable, Dict, Any
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


class PipelineScheduler:
    """Manage automated pipeline execution."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scheduler = BackgroundScheduler()
        self.logger = logging.getLogger(__name__)
    
    def schedule_retrain(self, retrain_func: Callable) -> None:
        """Schedule periodic model retraining."""
        retrain_config = self.config['scheduling']['retrain']
        
        if not self.config['scheduling']['enabled']:
            self.logger.warning("Scheduling is disabled")
            return
        
        # Parse cron schedule
        trigger = CronTrigger(
            day_of_week=retrain_config['day_of_week'],
            hour=retrain_config['hour'],
            minute=retrain_config['minute']
        )
        
        self.scheduler.add_job(
            retrain_func,
            trigger=trigger,
            id='model_retrain',
            name='Model Retraining Job',
            replace_existing=True
        )
        
        self.logger.info(
            f"✅ Retraining scheduled for {retrain_config['day_of_week']} "
            f"at {retrain_config['hour']}:{retrain_config['minute']:02d}"
        )
    
    def schedule_inference(self, inference_func: Callable) -> None:
        """Schedule periodic inference on new data."""
        inference_config = self.config['scheduling']['inference']
        
        if not self.config['scheduling']['enabled']:
            self.logger.warning("Scheduling is disabled")
            return
        
        trigger = CronTrigger(
            day_of_week=inference_config['day_of_week'],
            hour=inference_config['hour'],
            minute=inference_config['minute']
        )
        
        self.scheduler.add_job(
            inference_func,
            trigger=trigger,
            id='model_inference',
            name='Model Inference Job',
            replace_existing=True
        )
        
        self.logger.info(
            f"✅ Inference scheduled for {inference_config['day_of_week']} "
            f"at {inference_config['hour']}:{inference_config['minute']:02d}"
        )
    
    def schedule_report_generation(self, report_func: Callable) -> None:
        """Schedule periodic report generation."""
        report_config = self.config['scheduling']['report']
        
        if not self.config['scheduling']['enabled']:
            self.logger.warning("Scheduling is disabled")
            return
        
        trigger = CronTrigger(
            day_of_week=report_config['day_of_week'],
            hour=report_config['hour'],
            minute=report_config['minute']
        )
        
        self.scheduler.add_job(
            report_func,
            trigger=trigger,
            id='report_generation',
            name='Report Generation Job',
            replace_existing=True
        )
        
        self.logger.info(
            f"✅ Report generation scheduled for {report_config['day_of_week']} "
            f"at {report_config['hour']}:{report_config['minute']:02d}"
        )
    
    def start(self) -> None:
        """Start the scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self.logger.info("🚀 Pipeline scheduler started")
            print("🚀 Pipeline scheduler started")
            print("\nScheduled jobs:")
            for job in self.scheduler.get_jobs():
                print(f"  - {job.name}: {job.trigger}")
    
    def stop(self) -> None:
        """Stop the scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown()
            self.logger.info("⏹️ Pipeline scheduler stopped")
            print("⏹️ Pipeline scheduler stopped")
    
    def get_jobs(self) -> list:
        """Get list of scheduled jobs."""
        return self.scheduler.get_jobs()
    
    def trigger_job_now(self, job_id: str) -> None:
        """Manually trigger a scheduled job."""
        job = self.scheduler.get_job(job_id)
        if job:
            job.func()
            self.logger.info(f"✅ Triggered job: {job_id}")
            print(f"✅ Triggered job: {job_id}")
        else:
            self.logger.warning(f"⚠️ Job not found: {job_id}")
            print(f"⚠️ Job not found: {job_id}")


class ErrorHandler:
    """Handle errors in scheduled jobs."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def wrap_job(self, func: Callable) -> Callable:
        """Wrap job function with error handling."""
        def wrapper():
            try:
                self.logger.info(f"Starting job: {func.__name__}")
                result = func()
                self.logger.info(f"✅ Job completed: {func.__name__}")
                return result
            except Exception as e:
                self.logger.error(f"❌ Job failed: {func.__name__} - {str(e)}", exc_info=True)
                # In production, send alert to monitoring system
                print(f"\n❌ Job Error: {str(e)}\n")
        
        return wrapper


def create_scheduler(config: Dict[str, Any]) -> PipelineScheduler:
    """Factory function to create scheduler."""
    scheduler = PipelineScheduler(config)
    return scheduler
