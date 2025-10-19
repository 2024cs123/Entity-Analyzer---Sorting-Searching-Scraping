import sys
import pandas as pd
import numpy as np
from typing import List, Tuple
import os

from PyQt5.QtCore import (
    Qt,
    QThreadPool
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
    QScrollArea,
    QDialog
)
from PyQt5.QtGui import QPalette, QColor, QFont

# Import custom modules
from sort_core import SortExecutor
from data_model import PaginatedPandasModel, FilterRow
from scraper_gui import AIScraperDialog, AIScraperWorker


class SortingApp(QMainWindow):

    MAX_SCRAPE_ROWS = 25000

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entity Analyzer (Sorting, Searching & AI Scraping)")
        self.setGeometry(100, 100, 1400, 900)

        # Data Management
        self.orignal_df = pd.DataFrame()
        self.current_df = pd.DataFrame()
        self.model: PaginatedPandasModel = None
        self.sorter = SortExecutor()
        self.temp_csv_path = "ai_scraper_results_temp.csv" # Path to save LLM results

        # State Variables
        self.threadPool = QThreadPool()
        self.scraper_worker: AIScraperWorker = None
        self.sort_columns: List[str] = []
        self.filter_rows: List[FilterRow] = []
        self.initial_columns = ["ID", "Value1", "Value2", "City", "Key"]

        # 1. Central Widget and Main Layout 
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # 2. Setup UI
        self.setup_control_layout()
        self.setup_table_view()
        self.setup_pagination_controls() 
        self.setup_status_labels() 

        # 3. Applying Professional Stylesheet
        self.apply_stylesheet()

        # 4. Loading an Initial dummy dataset
        self.create_dummy_data()

    def apply_stylesheet(self):
        # [Stylesheet implementation - kept here for full class cohesion]
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(250, 250, 250))
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(220, 220, 220))
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.Highlight, QColor(66, 133, 244))
        palette.setColor(QPalette.HighlightedText, Qt.white)
        self.setPalette(palette)

        style = """
        QMainWindow { background-color: #f0f0f0; }
        QGroupBox { 
            font-weight: bold; 
            margin-top: 10px; 
            padding-top: 20px;
            border: 1px solid #c0c0c0;
            border-radius: 5px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 3px;
            color: #4285f4; 
        }
        QPushButton { 
            background-color: #e0e0e0; 
            border: 1px solid #c0c0c0; 
            padding: 5px; 
            border-radius: 3px;
        }
        QPushButton#AIScraperButton {
            background-color: #4285f4; /* Blue */
            color: white;
        }
        QPushButton#AIScraperButton:hover { background-color: #3b78e7; }
        QPushButton#ResetButton {
            background-color: #f4b400; /* Yellow/Orange */
        }
        QPushButton:hover { background-color: #d0d0d0; }
        QPushButton:pressed { background-color: #c0c0c0; }
        QComboBox, QLineEdit, QTextEdit {
            padding: 3px;
            border: 1px solid #c0c0c0;
            border-radius: 3px;
        }
        QTableView {
            selection-background-color: #4285f4;
            gridline-color: #e0e0e0;
            border: 1px solid #c0c0c0;
        }
        QHeaderView::section {
            background-color: #d8d8d8;
            padding: 4px;
            border: 1px solid #c0c0c0;
            font-weight: bold;
        }
        QLabel#SortListLabel {
            color: #c0392b; /* Red color for sort list */
            font-weight: bold;
        }
        """
        self.setStyleSheet(style)
        self.setFont(QFont("Arial", 10))

    # --- UI SETUP METHODS (Similar to previous consolidated code) ---
    
    def setup_control_layout(self):
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.setup_data_load_group())
        control_layout.addWidget(self.setup_sorting_group())
        control_layout.addWidget(self.setup_searching_group())
        self.main_layout.addLayout(control_layout)

    def setup_data_load_group(self):
        group = QGroupBox("Data Source & Control")
        layout = QVBoxLayout()
        
        self.ai_scrape_setup_btn = QPushButton("Start Web Scraper with LLM(Ollama)")
        self.ai_scrape_setup_btn.setObjectName("AIScraperButton")
        self.scrape_progress = QProgressBar()
        self.scrape_progress.setTextVisible(True)
        self.load_csv_btn = QPushButton("Load CSV File")
        self.reload_original_btn = QPushButton("Reset/Reload Original Table")
        self.reload_original_btn.setObjectName("ResetButton")

        layout.addWidget(self.ai_scrape_setup_btn)
        layout.addWidget(self.scrape_progress)
        layout.addWidget(self.load_csv_btn)
        layout.addWidget(self.reload_original_btn)
        group.setLayout(layout)

        self.ai_scrape_setup_btn.clicked.connect(self.show_ai_scraper_dialog)
        self.load_csv_btn.clicked.connect(self.load_csv_data)
        self.reload_original_btn.clicked.connect(self.reload_original_data)
        return group
    
    def setup_sorting_group(self):
        group = QGroupBox("Sorting Options")
        layout = QGridLayout()

        layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(self.sorter.algorithm_map.keys()))
        self.algo_combo.currentIndexChanged.connect(self.check_sort_algorithm)
        layout.addWidget(self.algo_combo, 0, 1, 1, 2)

        layout.addWidget(QLabel("Sort By:"), 1, 0)
        self.sort_list_label = QLabel("Select Column (Double-click header)")
        self.sort_list_label.setObjectName("SortListLabel")
        layout.addWidget(self.sort_list_label, 1, 1, 1, 2)
        
        self.sort_columns_clear_btn = QPushButton("Clear")
        self.sort_columns_clear_btn.clicked.connect(self.clear_sort_columns)
        layout.addWidget(self.sort_columns_clear_btn, 1, 3)

        self.sort_btn = QPushButton("Run Sort")
        self.sort_btn.clicked.connect(self.run_sort)
        self.sort_btn.setStyleSheet("margin-top : 10px;")
        layout.addWidget(self.sort_btn, 3, 0, 1, 4)

        group.setLayout(layout)
        return group

    def setup_searching_group(self):
        group = QGroupBox("Search and Filter")
        layout = QVBoxLayout()

        composite_op_layout = QHBoxLayout()
        composite_op_layout.addWidget(QLabel("Combine Filters (AND/OR):"))
        self.composite_op = QComboBox()
        self.composite_op.addItems(["AND", "OR"])
        composite_op_layout.addWidget(self.composite_op)
        composite_op_layout.addStretch(1)

        self.filter_container_layout = QVBoxLayout()
        self.filter_container_layout.addStretch(1) 

        self.filter_container_widget = QWidget()
        self.filter_container_widget.setLayout(self.filter_container_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.filter_container_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        filter_button_layout = QHBoxLayout()
        self.add_filter_btn = QPushButton("Add Filter")
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.add_filter_btn.clicked.connect(lambda: self.add_filter_row(self.current_df.columns.tolist()))
        self.apply_filter_btn.clicked.connect(self.apply_search_filter)

        filter_button_layout.addWidget(self.add_filter_btn)
        filter_button_layout.addWidget(self.apply_filter_btn)

        layout.addLayout(composite_op_layout)
        layout.addWidget(scroll_area, 1) 
        layout.addLayout(filter_button_layout)

        group.setLayout(layout)
        self.add_filter_row(self.initial_columns)
        return group
    
    def setup_table_view(self):
        self.table_view = QTableView()
        self.table_view.horizontalHeader().sectionDoubleClicked.connect(self.add_sort_column)
        self.main_layout.addWidget(self.table_view)

    def setup_pagination_controls(self):
        self.pagination_layout = QHBoxLayout()
        self.prev_page_btn = QPushButton("Previous Page")
        self.next_page_btn = QPushButton("Next Page")
        self.page_info_label = QLabel("Page 0 of 0")
        
        self.prev_page_btn.clicked.connect(lambda: self.set_page(self.model.page_number - 1))
        self.next_page_btn.clicked.connect(lambda: self.set_page(self.model.page_number + 1))

        self.pagination_layout.addWidget(self.prev_page_btn)
        self.pagination_layout.addWidget(self.page_info_label)
        self.pagination_layout.addWidget(self.next_page_btn)
        self.main_layout.addLayout(self.pagination_layout)

    def setup_status_labels(self):
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Status: Ready.")
        self.complexity_label = QLabel("Complexity: N/A")
        self.time_label = QLabel("Time Consumed: N/A")

        status_layout.addWidget(self.status_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.complexity_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.time_label)
        self.main_layout.addLayout(status_layout)


    # --- DATA MANAGEMENT & UTILITY METHODS ---

    def create_dummy_data(self):
        data = {
            "ID": np.arange(100),
            "Value1": np.random.randint(-500, 1000, 100), # Includes negatives
            "Value2": np.random.uniform(10.0, 50.0, 100).round(2),
            "City": ["NY", "LA", "CHI", "HOU", "PHI", "PHX", "SA", "SD", "DAL", "SJ"] * 10,
            "Key": np.random.randint(1000, 9999, 100),
        }
        df = pd.DataFrame(data)
        self.orignal_df = df
        self.load_data_to_table(df)
        self.time_label.setText(f"Initial Data Loaded: {len(df)} rows (Value1 has negatives)")

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

    def load_data_to_table(self, df: pd.DataFrame):
        h_scroll = self.table_view.horizontalScrollBar().value()
        v_scroll = self.table_view.verticalScrollBar().value()

        self.current_df = df

        if self.model is None:
            self.model = PaginatedPandasModel(self.current_df, parent=self)
            self.table_view.setModel(self.model)
        else:
            self.model.update_data(self.current_df)

        self.update_combo_box(self.current_df.columns.tolist())
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.update_pagination_ui()

        self.table_view.horizontalScrollBar().setValue(h_scroll)
        self.table_view.verticalScrollBar().setValue(v_scroll)
    
    def reload_original_data(self):
        """Resets the current table view back to the original loaded DataFrame."""
        if self.orignal_df.empty:
            QMessageBox.warning(self, "No Data", "No original data available to reload.")
            return

        # 1. Clear Filter UI
        for filter_row in self.filter_rows[:]:
            self.remove_filter_row(filter_row)
        self.add_filter_row(self.orignal_df.columns.tolist())

        # 2. Clear sort selection
        self.clear_sort_columns()

        # 3. Load the original data (which clears the current view, filter, and sort results)
        self.load_data_to_table(self.orignal_df.copy())
        self.time_label.setText(f"Original Data Reloaded: {len(self.orignal_df)} rows")
        self.complexity_label.setText("Complexity: N/A")

    def update_combo_box(self, columns: List[str]):
        """Updates all column combo box with current DataFrame columns."""
        for filter_row in self.filter_rows:
            current_col = filter_row.col_combo.currentText()
            filter_row.col_combo.clear()
            filter_row.col_combo.addItems(columns)
            if current_col in columns:
                filter_row.col_combo.setCurrentText(current_col)
    
    # --- PAGINATION ---

    def set_page(self, page_num):
        if self.model and self.model.set_page(page_num):
            self.update_pagination_ui()

    def update_pagination_ui(self):
        if self.model and self.model.total_pages > 0:
            current = self.model.page_number + 1
            total = self.model.total_pages
            self.page_info_label.setText(f"Page {current} of {total}")
            self.prev_page_btn.setEnabled(current > 1)
            self.next_page_btn.setEnabled(current < total)
        else:
            self.page_info_label.setText("Page 0 of 0")
            self.prev_page_btn.setEnabled(False)
            self.next_page_btn.setEnabled(False)


    # --- SORTING METHODS ---

    def add_sort_column(self, index):
        if self.current_df.empty: return
        col_name = self.current_df.columns[index]
        if col_name in self.sort_columns: self.sort_columns.remove(col_name)
        self.sort_columns.insert(0, col_name)
        if len(self.sort_columns) > 3: self.sort_columns = self.sort_columns[:3] 
        self.update_sort_ui()

    def clear_sort_columns(self):
        self.sort_columns = []
        self.update_sort_ui()

    def update_sort_ui(self):
        if not self.sort_columns:
            self.sort_list_label.setText("None (Double-click table header)")
            self.check_sort_algorithm()
            return

        sort_str = " > ".join(self.sort_columns)
        self.sort_list_label.setText(sort_str)
        self.check_sort_algorithm()
    
    def check_sort_algorithm(self):
        """Checks constraints for sorting algorithms."""
        selected_algo = self.algo_combo.currentText()
        non_comparison_sorts = ["Counting Sort", "Radix Sort", "Bucket Sort"]
        self.sort_btn.setEnabled(True)
        self.status_label.setText(f"Ready to sort {len(self.current_df)} rows with {selected_algo}.")

        if not self.sort_columns:
            self.sort_btn.setEnabled(False)
            self.status_label.setText("Select a column to sort by.")
            return

        if len(self.sort_columns) > 1:
            if selected_algo not in self.sorter.comparison_sorts and selected_algo != "Optimized (Pandas Timsort)":
                self.sort_btn.setEnabled(False)
                self.status_label.setText(
                    f"Error: Multi-column sort requires **Comparison Sorts** or **Optimized Timsort**."
                )
                return
        
        if selected_algo in non_comparison_sorts:
            selected_col_name = self.sort_columns[0]
            df_to_check = self.orignal_df if not self.orignal_df.empty else self.current_df
            is_numeric = pd.api.types.is_numeric_dtype(df_to_check[selected_col_name])
            
            if not is_numeric:
                self.sort_btn.setEnabled(False)
                self.status_label.setText(
                    f"Error: {selected_algo} only supports **numeric data**. Select a numeric column."
                )
                return
            
            if selected_algo == "Radix Sort":
                if (df_to_check[selected_col_name] < 0).any():
                    self.sort_btn.setEnabled(False)
                    self.status_label.setText(
                        f"Error: {selected_algo} only supports **non-negative integers**. Counting Sort supports negatives."
                    )
                    return
                if not pd.api.types.is_integer_dtype(df_to_check[selected_col_name]):
                    self.sort_btn.setEnabled(False)
                    self.status_label.setText(
                        f"Error: {selected_algo} only supports **integers**."
                    )
                    return


    def run_sort(self):
        selected_algo = self.algo_combo.currentText()
        column_names = self.sort_columns

        if self.current_df.empty or not column_names or not self.sort_btn.isEnabled():
            QMessageBox.critical(self, "Sort Error", "Cannot run sort. Check data, selected columns, and status bar.")
            return

        try:
            sorted_df, time_ms, complexity = self.sorter.execute_sort(
                self.current_df.copy(), column_names, selected_algo
            )

            self.load_data_to_table(sorted_df)
            self.current_df = sorted_df
            self.complexity_label.setText(f"Complexity: {complexity}")
            self.time_label.setText(f"Time Consumed: {time_ms:.4f}ms")
            self.status_label.setText("Sort complete.")

        except TypeError as e:
            QMessageBox.critical(self, "Algorithm Error", str(e))
            self.check_sort_algorithm()

        except Exception as e:
            QMessageBox.critical(
                self, "Sort Error", f"An unexpected Error occurred during sort: {e}"
            )


    # --- SEARCHING/FILTERING METHODS ---

    def add_filter_row(self, columns: List[str]):
        new_row = FilterRow(columns, self.remove_filter_row) 
        self.filter_rows.append(new_row)
        self.filter_container_layout.insertWidget(self.filter_container_layout.count() - 1, new_row)

    def remove_filter_row(self, row_widget: FilterRow):
        if row_widget in self.filter_rows:
            self.filter_rows.remove(row_widget)
            self.filter_container_layout.removeWidget(row_widget)
            row_widget.deleteLater()
            self.apply_search_filter()

    def apply_single_filter(
        self, df: pd.DataFrame, col_name: str, operator: str, value: str
    ) -> pd.Series:
        # [Implementation remains the same as previous file]
        if col_name not in df.columns or not value:
            return pd.Series([True] * len(df))

        column = df[col_name]
        if operator in ["==", ">", "<", "!="]:
            try:
                num_col = pd.to_numeric(column, errors="coerce")
                num_val = float(value)
                valid_mask = ~num_col.isna()
                result_mask = pd.Series([False] * len(df))
                
                if operator == "==":
                    result_mask[valid_mask] = num_col[valid_mask] == num_val
                elif operator == ">":
                    result_mask[valid_mask] = num_col[valid_mask] > num_val
                elif operator == "<":
                    result_mask[valid_mask] = num_col[valid_mask] < num_val
                elif operator == "!=":
                    result_mask[valid_mask] = num_col[valid_mask] != num_val
                
                if not valid_mask.all():
                    str_col = column.astype(str)
                    if operator == "==":
                        result_mask[~valid_mask] = str_col[~valid_mask] == value
                    elif operator == "!=":
                        result_mask[~valid_mask] = str_col[~valid_mask] != value

                return result_mask
            except ValueError:
                if operator in ["==", "!="]:
                    return (
                        (column.astype(str) == value)
                        if operator == "=="
                        else (column.astype(str) != value)
                    )
                return pd.Series([False] * len(df))

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
        current_df = self.orignal_df.copy() 

        active_filters = [
            filter_row.get_filter_data() for filter_row in self.filter_rows if filter_row.get_filter_data()[2]
        ]

        if not active_filters:
            self.load_data_to_table(self.orignal_df)
            self.time_label.setText(f"Filter Cleared: {len(self.orignal_df)} rows remaining.")
            self.complexity_label.setText("Complexity: N/A (Searching)")
            return

        final_mask = pd.Series([True] * len(current_df))

        for i, (col, op, val) in enumerate(active_filters):
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
        self.complexity_label.setText("Complexity: O(N*F) - N rows, F filters")


    # --- AI SCRAPER METHODS (Connecting GUI to Worker) ---

    def show_ai_scraper_dialog(self):
        """Displays the AI Scraper Setup Dialog."""
        if self.scraper_worker and self.threadPool.activeThreadCount() > 0:
            QMessageBox.information(
                self, "Status", "A scraping job is already running. Please wait."
            )
            return

        dialog = AIScraperDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            url, parse_description = dialog.get_settings()
            
            if not url or not parse_description:
                QMessageBox.warning(self, "Input Error", "URL and Parse Description cannot be empty.")
                return

            self.start_ai_scraping(url, parse_description)

    def start_ai_scraping(self, url: str, parse_description: str):
        self.scraper_worker = AIScraperWorker(url, parse_description, self.temp_csv_path)
        self.scraper_worker.signals.progress.connect(self.on_scrape_progress)
        self.scraper_worker.signals.finished.connect(self.on_scrape_finished)
        self.scraper_worker.signals.error.connect(self.on_scrape_error) 

        self.scrape_progress.setValue(0)
        self.ai_scrape_setup_btn.setEnabled(False)
        self.status_label.setText(
            f"AI Scraper started. Threads Active: {self.threadPool.activeThreadCount() + 1}"
        )
        self.threadPool.start(self.scraper_worker)

    def on_scrape_progress(self, percent: int, message: str):
        self.scrape_progress.setValue(percent)
        self.status_label.setText(f"Progress ({percent}%): {message}")

    def on_scrape_finished(self, df_scraped: pd.DataFrame):
        self.orignal_df = df_scraped
        self.load_data_to_table(df_scraped)
        self.ai_scrape_setup_btn.setEnabled(True)
        self.status_label.setText(
            f"AI Scraping complete. Loaded {len(df_scraped)} records. Results saved to '{self.temp_csv_path}'. Threads Active: {self.threadPool.activeThreadCount()}"
        )
        self.scraper_worker = None

    def on_scrape_error(self, error_message: str):
        QMessageBox.critical(self, "AI Scraper Error", error_message)
        self.ai_scrape_setup_btn.setEnabled(True)
        self.status_label.setText(f"AI Scraping failed. Threads Active: {self.threadPool.activeThreadCount()}")
        self.scraper_worker = None


# --- Main Execution ---

if __name__ == "__main__":
    # Setting recursion limit high enough for recursive sorts like Quicksort/Mergesort
    sys.setrecursionlimit(5000) 
    app = QApplication(sys.argv)
    ex = SortingApp()
    ex.show()
    sys.exit(app.exec_())