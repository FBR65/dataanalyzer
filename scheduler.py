#!/usr/bin/env python3
"""
Automated Excel Processing Scheduler
====================================

This script provides automated scheduling for Excel file processing.
It can be used with cron, Windows Task Scheduler, or run as a service.

Usage:
    python scheduler.py --config config.json
    python scheduler.py --source ./excel_files --interval 3600
"""

import os
import time
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict
import schedule
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

# Import custom tools
from tools.excel_processor import ExcelProcessor
from tools.duckdb_connector import DuckDBConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/scheduler.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ExcelFileHandler(FileSystemEventHandler):
    """File system event handler for Excel files"""

    def __init__(self, processor: ExcelProcessor, delay: int = 5):
        self.processor = processor
        self.delay = delay
        self.pending_files = set()
        self.last_modified = {}

    def on_modified(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
            self.pending_files.add(file_path)
            self.last_modified[file_path] = datetime.now()
            logger.info(f"Detected Excel file change: {file_path}")

    def on_created(self, event):
        if event.is_directory:
            return

        file_path = Path(event.src_path)
        if file_path.suffix.lower() in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
            self.pending_files.add(file_path)
            self.last_modified[file_path] = datetime.now()
            logger.info(f"Detected new Excel file: {file_path}")

    def process_pending_files(self):
        """Process files that haven't been modified for the delay period"""
        now = datetime.now()
        files_to_process = []

        for file_path in list(self.pending_files):
            if file_path in self.last_modified:
                if now - self.last_modified[file_path] > timedelta(seconds=self.delay):
                    files_to_process.append(file_path)
                    self.pending_files.remove(file_path)
                    del self.last_modified[file_path]

        for file_path in files_to_process:
            try:
                logger.info(f"Processing file: {file_path}")
                result = self.processor.process_excel_file(file_path)
                if result["status"] == "success":
                    logger.info(f"Successfully processed: {file_path}")
                else:
                    logger.error(
                        f"Failed to process {file_path}: {result.get('reason', 'Unknown error')}"
                    )
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")


class ExcelScheduler:
    """Main scheduler class for Excel processing"""

    def __init__(self, config_path: str = None):
        self.config = self.load_config(config_path)
        self.processor = None
        self.connector = None
        self.observer = None
        self.file_handler = None
        self.running = False

        # Initialize components
        self.setup_processor()
        self.setup_file_watcher()
        self.setup_scheduled_tasks()

    def load_config(self, config_path: str = None) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            "source_dir": "./excel_files",
            "duck_lake_path": "./ducklake",
            "duckdb_path": "./databases/scheduled_data.duckdb",
            "watch_mode": True,
            "batch_interval": 3600,  # 1 hour
            "file_delay": 5,  # 5 seconds
            "max_file_size_mb": 100,
            "cleanup_days": 30,
            "notifications": {"email": None, "webhook": None},
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                default_config.update(user_config)

        return default_config

    def setup_processor(self):
        """Initialize Excel processor"""
        try:
            self.processor = ExcelProcessor(
                source_dir=self.config["source_dir"],
                duck_lake_path=self.config["duck_lake_path"],
                duckdb_path=self.config["duckdb_path"],
                use_parquet=True,
                auto_clean=True,
            )

            self.connector = DuckDBConnector(
                duckdb_path=self.config["duckdb_path"],
                duck_lake_path=self.config["duck_lake_path"],
            )

            logger.info("Excel processor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Excel processor: {e}")
            raise

    def setup_file_watcher(self):
        """Setup file system watcher for real-time processing"""
        if not self.config["watch_mode"]:
            return

        try:
            self.file_handler = ExcelFileHandler(
                self.processor, delay=self.config["file_delay"]
            )

            self.observer = Observer()
            self.observer.schedule(
                self.file_handler, self.config["source_dir"], recursive=True
            )

            logger.info(f"File watcher setup for: {self.config['source_dir']}")

        except Exception as e:
            logger.error(f"Failed to setup file watcher: {e}")

    def setup_scheduled_tasks(self):
        """Setup scheduled tasks"""
        # Batch processing
        if self.config["batch_interval"] > 0:
            interval_minutes = self.config["batch_interval"] // 60
            schedule.every(interval_minutes).minutes.do(self.batch_process)
            logger.info(f"Scheduled batch processing every {interval_minutes} minutes")

        # Daily cleanup
        schedule.every().day.at("02:00").do(self.cleanup_old_files)
        logger.info("Scheduled daily cleanup at 02:00")

        # Pending files check (for watch mode)
        if self.config["watch_mode"]:
            schedule.every(30).seconds.do(self.check_pending_files)

    def batch_process(self):
        """Process all Excel files in batch"""
        try:
            logger.info("Starting batch processing...")
            results = self.processor.process_directory(recursive=True)

            logger.info(
                f"Batch processing complete: {results['processed']} processed, "
                f"{results['errors']} errors, {results['skipped']} skipped"
            )

            # Send notification if configured
            self.send_notification(
                f"Batch processing: {results['processed']} files processed"
            )

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            self.send_notification(f"Batch processing failed: {e}")

    def check_pending_files(self):
        """Check and process pending files in watch mode"""
        if self.file_handler:
            self.file_handler.process_pending_files()

    def cleanup_old_files(self):
        """Clean up old processed files"""
        try:
            logger.info("Starting cleanup of old files...")

            if self.processor:
                cleaned_count = self.processor.cleanup_old_files(
                    days_old=self.config["cleanup_days"]
                )
                logger.info(f"Cleaned up {cleaned_count} old files")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def send_notification(self, message: str):
        """Send notification via configured channels"""
        try:
            # Email notification
            if self.config["notifications"]["email"]:
                # Implementation depends on your email service
                logger.info(f"Email notification: {message}")

            # Webhook notification
            if self.config["notifications"]["webhook"]:
                import requests

                requests.post(
                    self.config["notifications"]["webhook"],
                    json={"message": message, "timestamp": datetime.now().isoformat()},
                )
                logger.info(f"Webhook notification sent: {message}")

        except Exception as e:
            logger.error(f"Failed to send notification: {e}")

    def get_status(self) -> Dict:
        """Get current scheduler status"""
        status = {
            "running": self.running,
            "config": self.config,
            "last_batch": getattr(self, "last_batch", None),
            "pending_files": len(self.file_handler.pending_files)
            if self.file_handler
            else 0,
        }

        if self.connector:
            try:
                tables = self.connector.list_tables()
                status["tables"] = len(tables)
                status["latest_table"] = tables[0]["table_name"] if tables else None
            except Exception as e:
                logger.error(f"Error getting status: {e}")

        return status

    def start(self):
        """Start the scheduler"""
        logger.info("Starting Excel processing scheduler...")
        self.running = True

        # Start file watcher
        if self.observer:
            self.observer.start()
            logger.info("File watcher started")

        # Start scheduled tasks
        def run_scheduled_tasks():
            while self.running:
                schedule.run_pending()
                time.sleep(1)

        schedule_thread = threading.Thread(target=run_scheduled_tasks)
        schedule_thread.daemon = True
        schedule_thread.start()

        logger.info("Scheduler started successfully")

        # Keep the main thread alive
        try:
            while self.running:
                time.sleep(60)
                logger.debug("Scheduler heartbeat")
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.stop()

    def stop(self):
        """Stop the scheduler"""
        logger.info("Stopping Excel processing scheduler...")
        self.running = False

        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("File watcher stopped")

        if self.processor:
            self.processor.close()

        if self.connector:
            self.connector.close()

        logger.info("Scheduler stopped")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Excel Processing Scheduler")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--source", help="Source directory for Excel files")
    parser.add_argument(
        "--interval", type=int, help="Batch processing interval (seconds)"
    )
    parser.add_argument("--watch", action="store_true", help="Enable file watching")
    parser.add_argument("--no-watch", action="store_true", help="Disable file watching")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")

    args = parser.parse_args()

    # Create configuration if arguments provided
    config = {}
    if args.source:
        config["source_dir"] = args.source
    if args.interval:
        config["batch_interval"] = args.interval
    if args.watch:
        config["watch_mode"] = True
    if args.no_watch:
        config["watch_mode"] = False

    # Save temporary config if needed
    config_path = args.config
    if config and not config_path:
        config_path = "temp_scheduler_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Create and start scheduler
    scheduler = ExcelScheduler(config_path)

    if args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2, default=str))
        return

    try:
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler error: {e}")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
