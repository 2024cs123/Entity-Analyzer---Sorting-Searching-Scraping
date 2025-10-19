import os
import pandas as pd
from io import StringIO
from typing import Tuple

from PyQt5.QtCore import QObject, pyqtSignal, QRunnable
from PyQt5.QtWidgets import (
    QDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QPushButton,
    QHBoxLayout,
    QMessageBox
)

from . import scraper_core

class AIScraperWorkerSignals(QObject):
    """Defines the signals available from a running worker thread."""
    progress = pyqtSignal(int, str) # percent, message
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    stopped = pyqtSignal()


class AIScraperWorker(QRunnable):
    """Worker thread to perform the full AI scraping pipeline."""

    def __init__(self, url: str, parse_description: str, temp_csv_path: str):
        super().__init__()
        self.signals = AIScraperWorkerSignals()
        self.url = url
        self.parse_description = parse_description
        self.temp_csv_path = temp_csv_path
        self.is_stopped = False

    def stop(self):
        """Sets the stop flag. The worker will check this flag and exit gracefully."""
        self.is_stopped = True

    def run(self):
        try:
            self.signals.progress.emit(5, "Scraping website...")
            html = scraper_core.scrape_website(self.url)
            
            self.signals.progress.emit(20, "Extracting and cleaning content...")
            body_content = scraper_core.extract_body_content(html)
            clean_text = scraper_core.clean_body_content(body_content)
            
            if not clean_text:
                raise Exception("Scraped content is empty or could not be cleaned.")

            self.signals.progress.emit(30, "Chunking content for LLM...")
            dom_chunks = scraper_core.split_dom_content(clean_text)
            
            self.signals.progress.emit(40, f"Parsing with LLM ({len(dom_chunks)} chunks)...")
            parsed_results_csv_string = scraper_core.parse_with_ollama(dom_chunks, self.parse_description)

            if parsed_results_csv_string == "No relevant information found." or not parsed_results_csv_string.strip():
                raise Exception("AI could not extract any relevant information from the page.")
            
            self.signals.progress.emit(80, "Converting LLM output to DataFrame...")
            
            # Converting the CSV string output directly to a DataFrame
            try:
                df = pd.read_csv(StringIO(parsed_results_csv_string), sep=',')
                
                # Save to CSV 
                df.to_csv(self.temp_csv_path, index=False) 
                
            except Exception as e:
                raise Exception(f"Failed to parse LLM CSV output. Check description and LLM consistency. Error: {e}")

            self.signals.progress.emit(100, "Scraping and Parsing Complete.")
            self.signals.finished.emit(df)

        except Exception as e:
            self.signals.error.emit(f"AI Scraper Error: {e}")
        finally:
            pass


class AIScraperDialog(QDialog):
    """A modal dialog for setting up AI scraping parameters (URL and Parse Description)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AI Web Scraper Configuration")
        self.setGeometry(200, 200, 600, 400)

        layout = QGridLayout(self)

        # URL Input
        layout.addWidget(QLabel("1. Target URL:"), 0, 0)
        self.url_input = QLineEdit("https://en.wikipedia.org/wiki/Comparison_sort")
        self.url_input.setPlaceholderText("Enter URL to scrape")
        layout.addWidget(self.url_input, 0, 1)

        # Parse Description Input
        layout.addWidget(QLabel("2. Parse Description (LLM Prompt):"), 1, 0)
        self.parse_input = QTextEdit()
        self.parse_input.setPlaceholderText("e.g., 'Extract the name and complexity (Big O) of all sorting algorithms mentioned in a table on this page.'")
        self.parse_input.setText("Extract a table of comparison sorting algorithms, listing the 'Algorithm Name' and 'Time Complexity' in CSV format.")
        layout.addWidget(self.parse_input, 1, 1)
        
        # Control Buttons
        self.start_btn = QPushButton("Start AI Scraper")
        self.cancel_btn = QPushButton("Cancel")
        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.start_btn)
        
        layout.addLayout(button_layout, 2, 0, 1, 2)

        # Connect Signals
        self.start_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

    def get_settings(self) -> Tuple[str, str]:
        """Returns the configured URL and parse description."""
        return self.url_input.text(), self.parse_input.toPlainText()