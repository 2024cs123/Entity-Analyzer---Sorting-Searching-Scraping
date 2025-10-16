import sys
import time
from Pagination import *
from Sort_Executor import *
from Scrapper import *
import pandas as pd
import random
import numpy as np
from typing import List, Callable, Tuple
from PyQt5.QtCore import (
    QThreadPool,
)

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableView,
    QPushButton,
    QComboBox,
    QLabel,
    QFileDialog,
    QHeaderView,
    QProgressBar,
    QLineEdit,
    QGridLayout,
    QMessageBox,
    QGroupBox,
    QSpinBox,
)


# ====================================================================
# 5. MAIN GUI APPLICATION
# ====================================================================


class SortingApp(QMainWindow):

    MAX_SCRAPE_ROWS = 25000  # For larger test

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entity Analyzer (Sorting, Searching & Scraping)")
        self.setGeometry(100, 100, 1400, 900)

        self.orignal_df = pd.DataFrame()
        self.current_df = pd.DataFrame()  # The current visible (filtered Data)
        self.model: PaginatedPandasModel = None
        self.threadPool = QThreadPool()
        self.scraper: ScraperWorker = None
        self.sorter = SortExecutor()

        self.central_widget = QWidget()
        self.centralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.centralWidget)

        # List to hold dynamic filter rows
        self.filter_rows: List[FilterRow] = []
        self.initial_columns = ["ID", "Value1", "Value2", "City", "Key"]

        self.setup_control_layout()
        self.setup_table_view()
        self.setup_status_label()
        self.setup_pagination_controls()

        # Loading an Initial dummy dataset
        self.create_dummy_data()

        # ----- SETUP METHODS -----

    def setup_control_layout(self):
        control_layout = QHBoxLayout()

        control_layout.addWidget(self.setup_data_load_group())
        control_layout.addWidget(self.setup_sorting_group())
        control_layout.addWidget(self.setup_searching_group())

        self.main_layout.addLayout(control_layout)

    def setup_data_load_group(self):
        group = QGroupBox("Data Source")
        layout = QVBoxLayout()

        self.url_input = QLineEdit("https://example.com/api/data")
        self.url_input.setPlaceholderText("Enter URL for Scraping")
        self.target_count_spin = QSpinBox()
        self.target_count_spin.setRange(100, self.MAX_SCRAPE_ROWS * 2)
        self.target_count_spin.setValue(self.MAX_SCRAPE_ROWS)
        self.target_count_spin.setPrefix("Max Rows: ")

        self.scrape_progress = QProgressBar()
        self.scrape_progress.setTextVisible = True

        scrape_control_layout = QHBoxLayout()
        self.scrape_start_btn = QPushButton("Start Scraping")
        self.scrape_pause_btn = QPushButton("Pause")
        self.scrape_stop_btn = QPushButton("Pause")
        scrape_control_layout.addWidget(self.scrape_start_btn)
        scrape_control_layout.addWidget(self.scrape_pause_btn)
        scrape_control_layout.addWidget(self.scrape_stop_btn)

        self.load_csv_btn = QPushButton("Load CSV File")

        layout.addWidget(self.url_input)
        layout.addWidget(self.target_count_spin)
        layout.addWidget(self.scrape_progress)
        layout.addLayout(scrape_control_layout)
        layout.addWidget(self.load_csv_btn)

        group.setLayout(layout)

        # Connect Signals
        self.scrape_start_btn.clicked.connect(self.start_scraping)
        self.scrape_pause_btn.clicked.connect(self.toggle_scrape_pause)
        self.scrape_stop_btn.clicked.connect(self.stop_scraping)
        self.load_csv_btn.clicked.connect(self.load_csv_data)

        return group

    def setup_sorting_group(self):
        group = QGroupBox("Sorting Options")
        layout = QGridLayout()

        # Row 0-1: Single Column and Algortihm Selection
        layout.addWidget(QLabel("Primarty Col:"), 0, 0)
        self.col_primary_combo = QComboBox()
        layout.addWidget(self.col_primary_combo, 0, 1)

        layout.addWidget(QLabel("Algorithm:"), 1, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(self.sorter.algorithm_map.keys()))
        self.algo_combo.currentIndexChanged.connect(self.check_sort_algorithm)
        layout.addWidget(self.algo_combo, 1, 1)

        # Row 2: Secondary Column (for Multi-column Sort)
        layout.addWidget(QLabel("Secondary Col:", 2, 0))
        self.col_secondary_combo = QComboBox()
        layout.addWidget(self.col_secondary_combo, 2, 1)

        # Row 3: Sort Button
        self.sort_btn = QPushButton("Run Sort")
        self.sort_btn.clicked.connect(self.run_sort)
        layout.addWidget(self.sort_btn, 3, 0, 1, 2)

        group.setLayout(layout)
        return group

    def setup_searching_group(self):
        group = QGroupBox("Search and Filter")
        layout = QGridLayout()

        self.filter_container_layout = QVBoxLayout()
        self.filter_container_widget = QWidget()
        self.filter_container_widget.setLayout(self.filter_container_layout)

        layout.addWidget(self.filter_container_widget, 0, 0, 1, 3)

        # Initial Filter Row
        self.add_filter_row(self.initial_columns)

    def set_up_table_view(self):
        self.table_view = QTableView()
        self.main_layout.addWidget(self.table_view)

    def setup_pagination_control(self):
        pagination_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Page")
        self.next_btn = QPushButton("Next Page")
        self.page_label = QLabel("Page 1 of 1")
        self.prev_btn.clicked.connect(lambda: self.change_page(-1))
        self.next_btn.clicked.connect(lambda: self.change_page(1))

        pagination_layout.addStretch()
        pagination_layout.addWidget(self.prev_btn)
        pagination_layout.addWidget(self.page_label)
        pagination_layout.addWidget(self.next_btn)
        pagination_layout.addStretch()

        self.main_layout.addLayout(pagination_layout)

    def setup_status_labels(self):
        status_panel = QHBoxLayout()
        self.complexity_label = QLabel("Compplexity: N/A")
        self.time_label = QLabel("Time Consumed: N/A")
        self.status_label = QLabel(
            f"Thread active: {self.threadPool.activeThreadCount()}"
        )

        status_panel.addWidget(self.complexity_label)
        status_panel.addWidget(self.time_label)
        status_panel.addWidget(self.status_label)

        self.main_layout.addLayout(status_panel)

    # ----- UTILITY METHODS -----

    def update_combo_box(self, columns: List[str]):
        """Updates all column combo box with current DataFrame columns."""
        for combo in [self.col_primary_combo, self.col_secondary_combo]:
            combo.clear()
            combo.addItems(columns)

        for filter_row in self.filter_rows:
            filter_row.col_combo.clear()
            filter_row.col_combo.addItems(columns)

    def load_data_to_table(self, df: pd.DataFrame):
        """Updates the DataFrame, the model, and the UI controls."""

        self.current_df = df

        if self.model is None:
            self.model = PaginatedPandasModel(self.current_df, parent=self)
            self.table_view.setModel(self.model)
        else:
            self.model.update_data(self.current_df)

        self.update_combo_box(self.current_df.columns.tolist())
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def update_pagination_ui(self):
        """Updates the page number label and button states."""
        if self.model is None:
            self.page_label.setText("Page 0 of 0 (No Data)")
            return

        self.page_label.setText(
            f"Page {self.model.page_number+1} of {self.model.total_pages} (Total: {self.model.total_rows})"
        )
        self.prev_btn.setEnabled(self.model.page_number > 0)
        self.next_btn.setEnabled(self.model.page_number < self.model.total_pages - 1)

    def change_page(self, direction: int):
        """Logic to move to the next or previous page."""
        if self.model:
            if self.model.set_page(self.model.page_number + direction):
                self.update_pagination_ui()

    # --- Data Loading / Scraper Methods ---

    def create_dumy_data(self):
        """Creating a small dummy dataset for initial view."""
        data = {
            "ID": np.arange(20),
            "Value1": np.random.randint(100, 1000, 20),
            "Value2": np.random.uniform(10.0, 50.0, 20).round(2),
            "City": ["NY", "LA", "CHI", "HOU", "PHI", "PHX", "SA", "SD", "DAL", "SJ"]
            * 2,
            "Key": np.random.randint(1000, 9999, 20),
        }

        df = pd.DataFrame(data)
        self.orignal_df = df
        self.load_data_to_table(df)
        self.time_label.setText(f"Initial Data Loaded: {len(df)} rows")

    def start_scraping(self):
        if self.scraper and self.scraper.is_running:
            QMessageBox.information(
                self, "Status", "Scraping is already running or paused."
            )
            return

        start_url = self.url_input.text()
        target_count = self.target_count_spin.value()

        self.scraper = ScraperWorker(start_url, target_count=target_count)
        self.scraper.signals.progress.connect(self.scrape_progress.setValue)
        self.scraper.signals.finished.connect(self.on_scrape_finished)
        self.scraper.signals.error.connect(self.on_scrape_error)

        self.scrape_progress.value(0)
        self.status_label.setText(
            f"Scraping started for {target_count} items from {start_url}..."
        )
        self.threadPool.start(self.scraper)

    def toggle_scrape_pause(self):
        if not self.scraper or not self.scraper.is_running:
            return

        if self.scraper.is_paused:
            self.scraper.resume()
            self.scrape_pause_btn.setText("Pause")
            self.status_label.setText("Scraping resumed.")
        else:
            self.scraper.pause()
            self.scrape_pause_btn.setText("Pause")
            self.status_label.setText("Scraping stopped.")

    def stop_scraping(self):
        if self.scraper:
            self.scraper.stop()
            self.scraper = None
            self.scrape_progress.setValue(0)
            self.status_label.setText(f"Scraping stopped.")

    def on_scrape_finished(self, data_list, headers):
        df_scraped = pd.DataFrame(data_list, columns=headers)
        self.orignal_df = df_scraped
        self.load_data_to_table(df_scraped)
        self.status_label.setText(
            f"Scraping complete. Loaded {len(df_scraped)} entities"
        )
        self.scraper = None

    def on_scraped_error(self, error_message):
        QMessageBox.critical(self, "Scraping Error", error_message)
        self.status_label.setText("Scraping failed.")
        self.scraper = None

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                df_loaded = pd.read_csv(file_path, nrows=self.MAX_SCRAPE_ROWS)
                self.orignal_df = df_loaded
                self.load_data_to_table(df_loaded)
                self.time_label.setText(f"CSV Loaded: {len(df_loaded)} rows")
                self.complexity_label.setText("Complexity: N/A")
            except Exception as e:
                QMessageBox.critical(self, "CSV Error", f"Error loading CSV: {e}")

    # --- Sorting Methods ---

    def check_sort_algorithm(self):
        """Disables non-comparison sorts if the primary column is not numeric."""

        selected_algo = self.algo_combo.currentText()
        selected_col_name = self.col_primary_combo.currentText()

        is_numeric = False
        if selected_col_name and not self.current_df.empty:
            is_numeric = pd.api.types.is_numeric_dtype(
                self.current_df[selected_col_name]
            )

        non_comparison_sorts = ["Counting Sort", "Radix Sort", "Bucket Sort"]
        is_non_comparison = selected_algo in non_comparison_sorts

        if not is_numeric and is_non_comparison:
            self.sort_btn.setEnabled(False)
            self.status_label.setText(
                f"Error: {selected_algo} is only for numeric Data. Select a numeric column or another algorithm."
            )

        else:
            self.sort_btn.setEnabled(True)
            self.status_label.setText(f"Ready to sort {len(self.current_df)} rows.")

    def run_sort(self):
        selected_col_primary = self.col_primary_combo.currentText()
        selected_col_secondary = self.col_secondary_combo.currentText()
        selected_algo = self.algo_combo.currentText()

        if self.current_df.empty:
            QMessageBox.warning(self, "Sort Error", "No data loaded to sort.")
            return

        column_names = [selected_col_primary]
        if selected_col_secondary and selected_col_secondary != selected_col_primary:
            column_names.append(selected_col_secondary)

        try:
            # The entire (filtered) current_df is sorted
            sorted_df, time_ms, complexity = self.sorter.execute_sort(
                self.current_df.copy(), column_names, selected_algo
            )

            # Update the UI
            self.load_data_to_table(sorted_df)
            self.current_df = sorted_df  # Keeping the newly sorted data as current
            self.complexity_label.setText(f"Complexity: {complexity}")
            self.time_label.setText(f"Time Consumed: {time_ms:.4f}ms")
            self.status_label.setText("Sort complete.")

        except TypeError as e:
            QMessageBox.critical(self, "Algorithm Error", str(e))
            self.check_sort_algorithm()  # Rechecking and disabling the button

        except Exception as e:
            QMessageBox.critical(
                self, "Sort Error", f"An unexpected Error occurred during sort: {e}"
            )

    # --- Searching Methods ---
    def add_filter_row(self, columns: List[str]):
        """Adds a new filter row widget to the composite filter area."""
        new_row = FilterRow(columns)
        self.filter_rows.append(new_row)
        self.filter_container_layout.addWidget(new_row)

    def apply_single_filter(
        self, df: pd.DataFrame, col_name: str, operator: str, value: str
    ) -> pd.Series:
        """Returrns a boolean mask for a single filter condition."""
        if col_name not in df.columns or not value:
            return pd.Series([True] * len(df))

        column = df[col_name]

        # 1. Numerical Comparison
        if operator in ["==", ">", "<", "!="]:
            try:
                num_col = pd.to_numeric(column, errors="coerce")
                num_val = float(value)
                if operator == "==":
                    return num_col == num_val
                if operator == ">":
                    return num_col > num_val
                if operator == "<":
                    return num_col < num_val
                if operator == "!=":
                    return num_col != num_val
            except ValueError:
                # If conversion fails, fall back to string comparison for '==' or '!='
                if operator in ["==", "!="]:
                    return (
                        (column.astype(str) == value)
                        if operator == "=="
                        else (column.astype(str) != value)
                    )
                return pd.Series([False] * len(df))  # Cannot apply >, < to non-numeric

        # 2. String Comparison
        column_str = column.astype(str).str
        if operator == "contains":
            return column_str.contains(value, case=False, na=False)
        if operator == "starts with":
            return column_str.startswith(value, na=False)
        if operator == "ends with":
            return column_str.endswith(value, na=False)

        return pd.Series([True] * len(df))

    def apply_search_filter(self):
        if self.orignal_df.empty:
            QMessageBox.warning(self, "Filter Error", "No data loaded to filter")
            return

        composite_op = self.composite_op.currentText()
        current_df = self.orignal_df

        if not self.filter_rows:
            self.load_data_to_table(current_df)
            return

        final_mask = pd.Series([True] * len(current_df))

        for i, filter_row in enumerate(self.filter_rows):
            col, op, val = filter_row.get_filter_data()

            if not val:
                continue  # skipping empty filters

            mask = self.apply_single_filter(current_df, col, op, val)
            if i == 0:
                final_mask = mask
            else:
                if composite_op == "AND":
                    final_mask &= mask
                elif composite_op == "OR":
                    final_mask |= mask

        filtered_df = current_df[final_mask].copy()

        self.load_data_to_table(filtered_df)
        self.time_label.setText(f"Filter Applied: {len(filtered_df)} rows remaining.")
        self.complexity_label.setText("Complexity: N/A (Searching)")
