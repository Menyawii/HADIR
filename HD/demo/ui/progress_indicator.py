from datetime import datetime
import sys

class ProgressIndicator:
    """Handles visual feedback for user interactions"""
    @staticmethod
    def show_status(message: str, end: str = '\n'):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}", end=end)
        sys.stdout.flush()

    @staticmethod
    def show_success(message: str):
        print(f"\n✅ {message}")

    @staticmethod
    def show_error(message: str):
        print(f"\n❌ {message}")

    @staticmethod
    def show_warning(message: str):
        print(f"\n⚠️ {message}")