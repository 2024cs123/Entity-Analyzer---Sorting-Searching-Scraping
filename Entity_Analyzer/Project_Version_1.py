import sys
import time
import pandas as pd
import random
import numpy as np
from typing import List, Callable, Tuple
from PyQt5.QtCore import (
    QAbstractTableModel,
    Qt,
    QVariant,
    QModelIndex,
    QObject,
    pyqtSignal,
    QRunnable,
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
    QSpinBox,
    QScrollArea,
)
from PyQt5.QtGui import QPalette, QColor, QFont

# ====================================================================
# 1. ALGORITHM CORE LOGIC
# ====================================================================


class SortingLibrary:
    """A comprehensive library for sorting algorithms with runtime tracking"""

    def __init__(self):
        self.runtime = 0.0
        self._start_time = 0.0

    def _start_timer(self):
        self._start_time = time.perf_counter()

    def _stop_timer(self, arr: List) -> List:
        self.runtime = (time.perf_counter() - self._start_time) * 1000  # Time in ms
        return arr

    # --- O(n^2) SIMPLE SORTS ------------------------------------------

    def bubble_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            for j in range(0, n - i - 1):
                # The algorithms must sort based on the first element of the tuple, which is the value.
                if arr[j][0] > arr[j + 1][0]:
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]
        return self._stop_timer(arr)

    def insertion_sort(self, arr: List) -> List:
        self._start_timer()
        for i in range(1, len(arr)):
            key = arr[i]
            j = i - 1
            # Comparison on the first element of the tuple
            while j >= 0 and key[0] < arr[j][0]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key
        return self._stop_timer(arr)

    def selection_sort(self, arr: List) -> List:
        self._start_timer()
        n = len(arr)
        for i in range(n):
            min_idx = i
            # Comparison on the first element of the tuple
            for j in range(i + 1, n):
                if arr[j][0] < arr[min_idx][0]:
                    min_idx = j
            arr[i], arr[min_idx] = arr[min_idx], arr[i]
        return self._stop_timer(arr)

    # --- QUICKSORT VARIANTS (O(n log n) Average) ----------------------

    def _partition_standard(self, arr: List, low: int, high: int) -> int:
        # Comparison on the first element of the tuple
        pivot = arr[high]
        i = low - 1
        for j in range(low, high):
            if arr[j][0] <= pivot[0]:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    def quicksort_standard(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            pivot = self._partition_standard(arr, low, high)
            # high=-2 is a dummy value to indicate non-top-level recursive call
            self.quicksort_standard(arr, low, pivot - 1, high=-2)
            self.quicksort_standard(arr, pivot + 1, high, high=-2)
        return self._stop_timer(arr) if is_top_level else arr

    def _partition_randomized(self, arr: List, low: int, high: int) -> int:
        rand_pivot_idx = random.randint(low, high)
        arr[rand_pivot_idx], arr[high] = arr[high], arr[rand_pivot_idx]
        return self._partition_standard(arr, low, high)

    def quicksort_randomized(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            pi = self._partition_randomized(arr, low, high)
            self.quicksort_randomized(arr, low, pi - 1, high=-2)
            self.quicksort_randomized(arr, pi + 1, high, high=-2)
        return self._stop_timer(arr) if is_top_level else arr

    def _partition_dual_pivot(self, arr: List, low: int, high: int) -> tuple:
        # Comparison on the first element of the tuple
        if arr[low][0] > arr[high][0]:
            arr[low], arr[high] = arr[high], arr[low]
        pivot1, pivot2 = arr[low], arr[high]
        leftPivotIndex, rightPivotIndex, iterator = low + 1, high - 1, low + 1

        while iterator <= rightPivotIndex:
            if arr[iterator][0] < pivot1[0]:
                arr[iterator], arr[leftPivotIndex] = arr[leftPivotIndex], arr[iterator]
                leftPivotIndex += 1
            elif arr[iterator][0] >= pivot2[0]:
                while (
                    arr[rightPivotIndex][0] > pivot2[0] and iterator < rightPivotIndex
                ):
                    rightPivotIndex -= 1
                arr[iterator], arr[rightPivotIndex] = (
                    arr[rightPivotIndex],
                    arr[iterator],
                )
                rightPivotIndex -= 1
                if arr[iterator][0] < pivot1[0]:
                    arr[iterator], arr[leftPivotIndex] = (
                        arr[leftPivotIndex],
                        arr[iterator],
                    )
                    leftPivotIndex += 1
            iterator += 1
        rightPivotIndex += 1
        leftPivotIndex -= 1
        arr[low], arr[high], arr[leftPivotIndex], arr[rightPivotIndex] = (
            arr[leftPivotIndex],
            arr[rightPivotIndex],
            arr[low],
            arr[high],
        )
        return leftPivotIndex, rightPivotIndex

    def quicksort_dual_pivot(self, arr: List, low: int = 0, high: int = -1) -> List:
        is_top_level = high == -1
        if is_top_level:
            high = len(arr) - 1
            self._start_timer()

        if low < high:
            lp, gp = self._partition_dual_pivot(arr, low, high)
            self.quicksort_dual_pivot(arr, low, lp - 1, high=-2)
            self.quicksort_dual_pivot(arr, lp + 1, gp - 1, high=-2)
            self.quicksort_dual_pivot(arr, gp + 1, high, high=-2)

        return self._stop_timer(arr) if is_top_level else arr

    # --- MERGE SORT VARIANTS (O(n log n)) -----------------------------

    def _merge(self, arr: List, l: int, m: int, r: int):
        n1, n2 = m - l + 1, r - m
        L, R = arr[l : l + n1], arr[m + 1 : m + 1 + n2]
        i, j, k = 0, 0, l

        while i < n1 and j < n2:
            # Comparison on the first element of the tuple
            if L[i][0] <= R[j][0]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
        while i < n1:
            arr[k] = L[i]
            k += 1
            i += 1
        while j < n2:
            arr[k] = R[j]
            k += 1
            j += 1

    def merge_sort(self, arr: List, l: int = 0, r: int = -1) -> List:
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_timer()

        if l < r:
            m = l + (r - l) // 2
            self.merge_sort(arr, l, m, r=-2)
            self.merge_sort(arr, m + 1, r, r=-2)
            self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr

    def _insertion_sort_sublist(self, arr: List, l: int, r: int):
        """Helper for Insertion Sort on a sub-array (used by Hybrid and Bucket)."""
        # NOTE: Bucket/Hybrid sorts do not use tuples in their helpers, as they sort internal lists of values.
        for i in range(l + 1, r + 1):
            key = arr[i]
            j = i - 1
            while j >= l and key < arr[j]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    def merge_sort_hybrid(
        self, arr: List, l: int = 0, r: int = -1, k: int = 15
    ) -> List:
        """
        Implements Hybrid Merge Sort.
        Uses Insertion Sort for sub-arrays of size k or less.
        """
        is_top_level = r == -1
        if is_top_level:
            r = len(arr) - 1
            self._start_timer()

        if l < r:
            if r - l <= k:
                # Need a version of insertion sort that handles (value, index) tuples
                self._insertion_sort_sublist_tuples(arr, l, r)
            else:
                m = l + (r - l) // 2
                self.merge_sort_hybrid(arr, l, m, r=-2, k=k)
                self.merge_sort_hybrid(arr, m + 1, r, r=-2, k=k)
                self._merge(arr, l, m, r)

        return self._stop_timer(arr) if is_top_level else arr

    def _insertion_sort_sublist_tuples(self, arr: List, l: int, r: int):
        """Helper for Insertion Sort on a sub-array of tuples (used by Hybrid Merge)."""
        for i in range(l + 1, r + 1):
            key = arr[i]
            j = i - 1
            # Comparison on the first element of the tuple
            while j >= l and key[0] < arr[j][0]:
                arr[j + 1] = arr[j]
                j -= 1
            arr[j + 1] = key

    # --- NON-COMPARISON SORTS -----------------------------------------

    def counting_sort(self, arr: List) -> List:
        """Implements In-Place Counting Sort. Assumes list of (value, index) tuples."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        # Check for non-integers (Counting Sort requirement)
        if any(not isinstance(x[0], (int, np.integer)) for x in arr):
            raise TypeError("Counting Sort requires all elements to be integers.")

        # Extract values only
        values = [x[0] for x in arr]
        min_val, max_val = min(values), max(values)
        range_size = max_val - min_val + 1
        # Count array stores the count of *values*
        count = [0] * range_size
        # Output array stores the sorted (value, index) tuples
        output = [None] * len(arr)

        # 1. Store count of each value
        for value, _ in arr:
            count[value - min_val] += 1

        # 2. Store cumulative count
        for i in range(1, range_size):
            count[i] += count[i - 1]

        # 3. Build the output array (stable sort)
        # Iterate backward to ensure stability
        for value, index in reversed(arr):
            output_index = count[value - min_val] - 1
            output[output_index] = (value, index)
            count[value - min_val] -= 1

        # 4. Copy the output array to the input array (in-place)
        for i in range(len(arr)):
            arr[i] = output[i]

        return self._stop_timer(arr)

    def _counting_sort_by_digit(self, arr: List, exp: int, base: int = 10):
        """A stable helper for Radix Sort. Works on (value, index) tuples."""
        n = len(arr)
        output = [None] * n
        count = [0] * base

        # 1. Count occurrences of digits
        for i in range(n):
            index = (arr[i][0] // exp) % base
            count[index] += 1

        # 2. Store cumulative count
        for i in range(1, base):
            count[i] += count[i - 1]

        # 3. Build the output array (stable sort)
        i = n - 1
        while i >= 0:
            index = (arr[i][0] // exp) % base
            output[count[index] - 1] = arr[i]
            count[index] -= 1
            i -= 1

        # 4. Copy the output array to the input array (in-place)
        for i in range(n):
            arr[i] = output[i]

    def radix_sort(self, arr: List, base: int = 10) -> List:
        """Implements Radix Sort (LSD). O(d * (n + b)). Assumes list of (value, index) tuples."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        if any(not isinstance(x[0], (int, np.integer)) or x[0] < 0 for x in arr):
            raise TypeError(
                "Radix Sort requires all elements to be non-negative integers."
            )

        # Extract values only for max calculation
        max_val = max(x[0] for x in arr)
        exp = 1

        while max_val // exp > 0:
            self._counting_sort_by_digit(arr, exp, base)
            exp *= base

        return self._stop_timer(arr)

    def bucket_sort(self, arr: List, num_buckets=10) -> List:
        """Implements Bucket Sort. Average O(n). Assumes list of (value, index) tuples."""
        self._start_timer()
        if not arr:
            return self._stop_timer(arr)

        # Extract values only
        values = [x[0] for x in arr]
        if any(not isinstance(x, (int, float, np.number)) for x in values):
            raise TypeError("Bucket Sort requires all elements to be numeric.")

        min_val, max_val = min(values), max(values)
        if min_val == max_val:
            return self._stop_timer(arr)

        range_val = max_val - min_val
        # Buckets now store tuples: (value, original_index)
        buckets = [[] for _ in range(num_buckets)]

        # Avoid division by zero when min_val == max_val
        bucket_size = range_val / num_buckets if range_val > 0 else 1

        for x_tuple in arr:
            # Formula to get bucket index (normalized to [0, num_buckets-1])
            x_val = x_tuple[0]
            idx = int((x_val - min_val) / bucket_size)
            if idx == num_buckets:
                idx = num_buckets - 1  # Edge case for max_val
            buckets[idx].append(x_tuple)

        sorted_arr_idx = 0
        for bucket in buckets:
            # Sort the internal bucket list of (value, index) tuples
            self._insertion_sort_sublist_tuples(bucket, 0, len(bucket) - 1)

            for elem_tuple in bucket:
                arr[sorted_arr_idx] = elem_tuple
                sorted_arr_idx += 1

        return self._stop_timer(arr)


# ====================================================================
# 2. ALGORITHM & DATA LOGIC WRAPPER
# ====================================================================


class SortExecutor:
    """Wraps SotingLibrary algorithms to work with Pandas DataFrames."""

    def __init__(self):
        self.library = SortingLibrary()
        self.algorithm_map: dict[str, Tuple[str, Callable]] = {
            # Algorithm Name: (Complexity, Function Pointer)
            "Selection Sort": ("O(n^2)", self.library.selection_sort),
            "Bubble Sort": ("O(n^2)", self.library.bubble_sort),
            "Insertion Sort": ("O(n^2)", self.library.insertion_sort),
            "Quicksort (Standard)": ("O(N log N)", self.library.quicksort_standard),
            "Quicksort (Randomized)": ("O(N log N)", self.library.quicksort_randomized),
            "Quicksort (Dual Pivot)": ("O(N log N)", self.library.quicksort_dual_pivot),
            "Merge Sort": ("O(N log N)", self.library.merge_sort),
            "Merge Sort (Hybrid)": ("O(N log N)", self.library.merge_sort_hybrid),
            "Counting Sort": ("O(N+K)", self.library.counting_sort),
            "Radix Sort": ("O(d*(N+b))", self.library.radix_sort),
            "Bucket Sort": ("O(N)", self.library.bucket_sort),
            "Optimized (Pandas Timsort)": ("O(N log N)", self._pandas_sort_stub),
        }

    def _pandas_sort_stub(self, arr: List) -> List:
        """Dummy function for the Pandas sort, to ensure it doesn't run the custom logic."""
        self.library._start_timer()
        self.library.runtime = 0.0  # Resetting timing
        return arr

    def execute_sort(
        self, df: pd.DataFrame, column_names: List[str], algorithm_name: str
    ) -> Tuple[pd.DataFrame, float, str]:
        """Sorts the entire DataFrame based on one or more columns using the specifies algorithm"""

        if not column_names:
            return df, 0.0, "N/A"

        # 1. Handling Multi-column or Pandas Optimized Sorts
        # Multi-column sort requires the optimized Pandas method.

        if len(column_names) > 1 or algorithm_name == "Optimized (Pandas Timsort)":
            start_time = time.perf_counter()
            # Pandas sort is inherently multi-column and highly optimized TimSort (or merge sort)
            # We use 'mergesort' kind for consistency/demonstration as it's O(N log N)
            sorted_df = df.sort_values(
                by=column_names, ascending=True, kind="mergesort", ignore_index=True
            )
            end_time = time.perf_counter()
            time_ms = (end_time - start_time) * 1000
            complexity = "O(N log N) Pandas"
            return sorted_df, time_ms, complexity

        # 2. Handling Single-column custom algorithms

        col_name = column_names[0]
        complexity, sort_func = self.algorithm_map.get(algorithm_name)

        # Preparing the proxy list: (value, original_index)
        try:
            sort_key_series = (
                pd.to_numeric(df[col_name], errors="coerce")
                if pd.api.types.is_numeric_dtype(df[col_name])
                else df[col_name].astype(str)
            )
        except Exception:
            # Fallback for complex types or unconvertible data
            sort_key_series = df[col_name].astype(str)

        # Create a list of tuples: (value, original_index)
        sortable_list = list(zip(sort_key_series.to_list(), df.index.to_list()))

        try:
            # Execute the IN-PLACE custom sort algorithm on the list of tuples
            sorted_list_of_tuples = sort_func(sortable_list)
            time_ms = self.library.runtime

            # Extract the sorted original indices
            sorted_indices = [idx for val, idx in sorted_list_of_tuples]

            # Reindex the original DataFrame to get the sorted DataFrame
            sorted_df = df.loc[sorted_indices].reset_index(drop=True)

            return sorted_df, time_ms, complexity

        except TypeError as e:
            # Non-comparison sort attempted on wrong data type (handled by checks inside sort functions)
            raise TypeError(f"Algorithm Mismatch: {e}")

        except Exception as e:
            print(f"Sort Error: {e}")
            raise Exception(f"Sort Failed: {e}")


# ====================================================================
# 3. PYQT MODEL/VIEW WITH PAGINATION
# ====================================================================


class PaginatedPandasModel(QAbstractTableModel):
    """A data model for QTableView that wraps a Pandas DataFrame and implements pagination."""

    ROWS_PER_PAGE = 5000  # Max rows to display per page

    def __init__(self, data: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.full_data = data
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]
        self.total_pages = int(np.ceil(self.total_rows / self.ROWS_PER_PAGE))
        self._page_data = pd.DataFrame() # Initialize
        self.update_page_data()

    def update_data(self, data: pd.DataFrame):
        """Called when uderlying DataFrame is sorted or filtered."""
        self.beginResetModel()
        self.full_data = data
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]
        self.total_pages = int(np.ceil(self.total_rows / self.ROWS_PER_PAGE))
        if self.page_number >= self.total_pages and self.total_pages > 0:
            self.page_number = max(0, self.total_pages - 1)
        # Handle case for empty DataFrame
        if self.total_rows == 0:
            self.total_pages = 0
            self._page_data = pd.DataFrame()
        else:
            self.update_page_data()
        self.endResetModel()

    def update_page_data(self):
        """Calculate the slice of the DataFrame for the current page."""
        if self.total_rows == 0:
            self._page_data = pd.DataFrame()
            return
        start_row = self.page_number * self.ROWS_PER_PAGE
        end_row = min((self.page_number + 1) * self.ROWS_PER_PAGE, self.total_rows)
        self._page_data = self.full_data.iloc[start_row:end_row]

    def set_page(self, page_num):
        """Changes the current page."""
        if 0 <= page_num < self.total_pages:
            self.page_number = page_num
            self.update_page_data()
            self.layoutChanged.emit()  # Signal to the view to redraw
            return True
        return False

    def rowCount(self, parent=QModelIndex()):
        return self._page_data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._page_data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        # Corrected: index.isValid()
        if not index.isValid():
            return QVariant()

        if role == Qt.DisplayRole:
            # Ensure safe access to iloc
            if index.row() < self._page_data.shape[0] and index.column() < self._page_data.shape[1]:
                return str(self._page_data.iloc[index.row(), index.column()])
            return QVariant()
        return QVariant()

    def headerData(self, col, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._page_data.columns[col])
        return QVariant()


# ====================================================================
# 4. SCRAPING THREADING & COMPOSITE FILTER UI
# ====================================================================


class ScraperWorkerSignals(QObject):
    progress = pyqtSignal(int)
    # The 'finished' signal signature passes data_list, headers
    finished = pyqtSignal(list, list) 
    error = pyqtSignal(str)


class ScraperWorker(QRunnable):
    """Worker thread to perform scraping logic (using dummy data)."""

    def __init__(self, start_url, target_count=25000):
        super().__init__()
        self.signals = ScraperWorkerSignals()
        self.url = start_url
        self.target_count = target_count
        self.is_running = True
        self.is_paused = False
        self.scraped_data = []

        self.headers = ["ID", "Value1", "Value2", "City", "Key"]

    def run(self):
        # Dummy scraping simulation
        self.scraped_data = []
        for i in range(self.target_count):
            if not self.is_running:
                return

            while self.is_paused:
                time.sleep(0.1)
                if not self.is_running:
                    return

            if i % 100 == 0:
                self.signals.progress.emit(int(i / self.target_count * 100))

            # Simulate scraping a data row
            row = [
                i,
                random.randint(100, 1000),
                round(random.uniform(10.0, 50.0), 2),
                random.choice(
                    ["NY", "LA", "CHI", "HOU", "PHI", "PHX", "SA", "SD", "DAL", "SJ"]
                ),
                random.randint(1000, 9999),
            ]
            self.scraped_data.append(row)

        if self.is_running: # Only emit finished if not stopped prematurely
            self.signals.progress.emit(100)
            self.signals.finished.emit(self.scraped_data, self.headers)

    def stop(self):
        self.is_running = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False


class FilterRow(QWidget):
    """A reusable widget for a single filter condition."""

    def __init__(self, column_names: List[str], parent=None):
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.col_combo = QComboBox()
        self.col_combo.addItems(column_names)

        self.op_combo = QComboBox()
        self.op_combo.addItems(
            ["==", ">", "<", "!=", "contains", "starts with", "ends with"]
        )

        self.val_input = QLineEdit()
        self.val_input.setPlaceholderText("Value")

        self.layout.addWidget(self.col_combo)
        self.layout.addWidget(self.op_combo)
        self.layout.addWidget(self.val_input)

    def get_filter_data(self) -> Tuple[str, str, str]:
        return (
            self.col_combo.currentText(),
            self.op_combo.currentText(),
            self.val_input.text().strip(),
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
        self.sort_columns: List[str] = [] # List for multi-column sort

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
        """Applies a minimal professional stylesheet."""
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240)) # Light gray background
        palette.setColor(QPalette.WindowText, Qt.black)
        palette.setColor(QPalette.Base, Qt.white)
        palette.setColor(QPalette.AlternateBase, QColor(250, 250, 250))
        palette.setColor(QPalette.ToolTipBase, Qt.black)
        palette.setColor(QPalette.ToolTipText, Qt.black)
        palette.setColor(QPalette.Text, Qt.black)
        palette.setColor(QPalette.Button, QColor(220, 220, 220)) # Slightly darker buttons
        palette.setColor(QPalette.ButtonText, Qt.black)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Highlight, QColor(66, 133, 244)) # Google Blue
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
            color: #4285f4; /* A nice blue for titles */
        }
        QPushButton { 
            background-color: #e0e0e0; 
            border: 1px solid #c0c0c0; 
            padding: 5px; 
            border-radius: 3px;
        }
        QPushButton:hover { background-color: #d0d0d0; }
        QPushButton:pressed { background-color: #c0c0c0; }
        QComboBox, QLineEdit, QSpinBox {
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


    # ----- SETUP METHODS -----

    def setup_control_layout(self):
        control_layout = QHBoxLayout()

        control_layout.addWidget(self.setup_data_load_group())
        control_layout.addWidget(self.setup_sorting_group())
        control_layout.addWidget(self.setup_searching_group())
        
        # control_layout.addStretch(1)
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

        # Row 0: Algorithm Selection
        layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(list(self.sorter.algorithm_map.keys()))
        self.algo_combo.currentIndexChanged.connect(self.check_sort_algorithm)
        layout.addWidget(self.algo_combo, 0, 1)

        # Row 1: Sort Column List Display
        layout.addWidget(QLabel("Sort By:"), 1, 0)
        self.sort_list_label = QLabel("Select Column")
        self.sort_list_label.setObjectName("SortListLabel")
        layout.addWidget(self.sort_list_label, 1, 1)
        self.sort_columns_clear_btn = QPushButton("Clear")
        self.sort_columns_clear_btn.clicked.connect(self.clear_sort_columns)
        layout.addWidget(self.sort_columns_clear_btn, 1, 2,1,2)


        # Row 2: Sort Button
        self.sort_btn = QPushButton("Run Sort")
        self.sort_btn.clicked.connect(self.run_sort)
        self.sort_btn.setStyleSheet("margin-top : 10px;")
        layout.addWidget(self.sort_btn, 3, 0, 1, 3)

        group.setLayout(layout)
        return group

    def setup_searching_group(self):
        group = QGroupBox("Search and Filter")
        layout = QVBoxLayout()

        # Composite Operator
        composite_op_layout = QHBoxLayout()
        composite_op_layout.addWidget(QLabel("Combine Filters:"))
        self.composite_op = QComboBox()
        self.composite_op.addItems(["AND", "OR"])
        composite_op_layout.addWidget(self.composite_op)
        composite_op_layout.addStretch(1)

        # Filter Container (Scrollable)
        self.filter_container_layout = QVBoxLayout()
        self.filter_container_layout.addStretch(1) # Adds stretch to push items to the top

        self.filter_container_widget = QWidget()
        self.filter_container_widget.setLayout(self.filter_container_layout)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.filter_container_widget)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Filter Control Buttons
        filter_button_layout = QHBoxLayout()
        self.add_filter_btn = QPushButton("Add Filter")
        self.apply_filter_btn = QPushButton("Apply Filter")
        self.add_filter_btn.clicked.connect(lambda: self.add_filter_row(self.current_df.columns.tolist()))
        self.apply_filter_btn.clicked.connect(self.apply_search_filter)

        filter_button_layout.addWidget(self.add_filter_btn)
        filter_button_layout.addWidget(self.apply_filter_btn)

        # Arrange Group Layout
        layout.addLayout(composite_op_layout)
        layout.addWidget(scroll_area, 1) # Set stretch to 1
        layout.addLayout(filter_button_layout)


        group.setLayout(layout)

        # Initial Filter Row
        self.filter_rows: List[FilterRow] = []
        self.add_filter_row(self.initial_columns)

        return group

    def setup_table_view(self):
        self.table_view = QTableView()
        self.table_view.horizontalHeader().sectionDoubleClicked.connect(self.add_sort_column)
        self.main_layout.addWidget(self.table_view, 1)

    def setup_pagination_controls(self):
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
        self.complexity_label = QLabel("Complexity: N/A")
        self.time_label = QLabel("Time Consumed: N/A")
        self.status_label = QLabel(
            f"Threads Active: 0"
        )
#  f"Thread active: {self.threadPool.activeThreadCount()
        status_panel.addWidget(self.complexity_label)
        status_panel.addWidget(self.time_label)
        status_panel.addStretch(1)
        status_panel.addWidget(self.status_label)

        self.main_layout.addLayout(status_panel)

    # ----- UTILITY METHODS -----

    def update_combo_box(self, columns: List[str]):
        """Updates all column combo box with current DataFrame columns."""

        # Update Filter Rows
        for filter_row in self.filter_rows:
            current_col = filter_row.col_combo.currentText()
            filter_row.col_combo.clear()
            filter_row.col_combo.addItems(columns)
            if current_col in columns:
                filter_row.col_combo.setCurrentText(current_col)

    def load_data_to_table(self, df: pd.DataFrame):
        """Updates the DataFrame, the model, and the UI controls."""

        # Store the current scroll position before reset
        h_scroll = self.table_view.horizontalScrollBar().value()
        v_scroll = self.table_view.verticalScrollBar().value()

        self.current_df = df

        if self.model is None:
            self.model = PaginatedPandasModel(self.current_df, parent=self)
            self.table_view.setModel(self.model)
        else:
            self.model.update_data(self.current_df)

        self.update_combo_box(self.current_df.columns.tolist())
        # The section resize mode must be set on the header
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.update_pagination_ui()

        # Restore scroll position
        self.table_view.horizontalScrollBar().setValue(h_scroll)
        self.table_view.verticalScrollBar().setValue(v_scroll)

    def update_pagination_ui(self):
        """Updates the page number label and button states."""
        if self.model is None or self.model.total_pages == 0:
            self.page_label.setText("Page 0 of 0 (No Data)")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
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

    # --- DATA LOADING / SCRAPING METHOD ---

    def create_dummy_data(self):
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

    def toggle_scrape_pause(self):
        if not self.scraper or not self.scraper.is_running:
            return

        if self.scraper.is_paused:
            self.scraper.resume()
            self.scrape_pause_btn.setText("Pause")
            self.status_label.setText(f"Scraping resumed. Threads Active: {self.threadPool.activeThreadCount()}")
        else:
            self.scraper.pause()
            self.scrape_pause_btn.setText("Resume") 
            self.status_label.setText(f"Scraping paused. Threads Active: {self.threadPool.activeThreadCount()}")

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

        self.scrape_progress.setValue(0) 
        self.status_label.setText(
            f"Scraping started for {target_count} items from {start_url}... Threads Active: {self.threadPool.activeThreadCount() + 1}"
        )
        self.threadPool.start(self.scraper)

    def stop_scraping(self):
        """
        Method to stop the scraper thread.
        """
        if self.scraper:
            self.scraper.stop()
            time.sleep(0.1) 
            self.scraper = None
            self.scrape_progress.setValue(0)
            self.status_label.setText(f"Scraping stopped by user. Threads Active: {self.threadPool.activeThreadCount()}")
        else:
            self.status_label.setText(f"No active scraping job to stop. Threads Active: {self.threadPool.activeThreadCount()}")

    def on_scrape_finished(self, data_list, headers):
        df_scraped = pd.DataFrame(data_list, columns=headers)
        self.orignal_df = df_scraped
        self.load_data_to_table(df_scraped)
        self.status_label.setText(
            f"Scraping complete. Loaded {len(df_scraped)} entities. Threads Active: {self.threadPool.activeThreadCount()}"
        )
        self.scraper = None

    def on_scrape_error(self, error_message):
        QMessageBox.critical(self, "Scraping Error", error_message)
        self.status_label.setText(f"Scraping failed. Threads Active: {self.threadPool.activeThreadCount()}")
        self.scraper = None

    def load_csv_data(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open CSV File", "", "CSV Files (*.csv)"
        )

        if file_path:
            try:
                # Use a larger default nrows or self.MAX_SCRAPE_ROWS
                df_loaded = pd.read_csv(file_path, nrows=self.MAX_SCRAPE_ROWS)
                self.orignal_df = df_loaded
                self.load_data_to_table(df_loaded)
                self.time_label.setText(f"CSV Loaded: {len(df_loaded)} rows")
                self.complexity_label.setText("Complexity: N/A")
            except Exception as e:
                QMessageBox.critical(self, "CSV Error", f"Error loading CSV: {e}")

    # --- SORTING METHOD ---

    def add_sort_column(self, index):
        """Adds or moves a column to the list of columns to sort by via double-clicking the header."""
        if self.current_df.empty:
            return

        col_name = self.current_df.columns[index]

        # If already in the list, move to front (primary sort key)
        if col_name in self.sort_columns:
            self.sort_columns.remove(col_name)
            self.sort_columns.insert(0, col_name)
        else:
            self.sort_columns.insert(0, col_name)

        # Limit to a reasonable number of columns for sorting
        if len(self.sort_columns) > 3:
            self.sort_columns = self.sort_columns[:3] 
        self.update_sort_ui()

    def clear_sort_columns(self):
        """Clears the list of columns to sort by."""
        self.sort_columns = []
        self.update_sort_ui()

    def update_sort_ui(self):
        """Updates the label displaying the currently selected sort columns."""
        if not self.sort_columns:
            self.sort_list_label.setText("None (Double-click table header)")
            # Re-check algorithm for button state 
            self.check_sort_algorithm()
            return

        # Display the list of columns
        sort_str = " > ".join(self.sort_columns)
        self.sort_list_label.setText(sort_str)

        # Check algorithm constraint
        self.check_sort_algorithm()

    def check_sort_algorithm(self):
        """Disables non-comparison sorts if the primary column is not numeric OR if multi-column sort is selected."""

        selected_algo = self.algo_combo.currentText()
        non_comparison_sorts = ["Counting Sort", "Radix Sort", "Bucket Sort"]

        # Multi-column sort check
        if len(self.sort_columns) > 1:
            if selected_algo != "Optimized (Pandas Timsort)":
                self.sort_btn.setEnabled(False)
                self.status_label.setText(
                    "Error: Multi-column sort requires 'Optimized Timsort'."
                )
                return
        
        # Single-column sort check
        if len(self.sort_columns) == 1:
            selected_col_name = self.sort_columns[0]
            # Using original_df for dtypes as current_df might be empty after filter
            df_to_check = self.orignal_df if not self.orignal_df.empty else self.current_df

            is_numeric = df_to_check[selected_col_name].dtype in ['int64', 'float64', 'int32', 'float32']
            
            # Check for custom non-comparison sorts
            is_non_comparison_custom = selected_algo in non_comparison_sorts

            if is_non_comparison_custom and not is_numeric:
                self.sort_btn.setEnabled(False)
                self.status_label.setText(
                    f"Error: {selected_algo} only supports numeric data. Select a numeric column."
                )
                return

        # If constraints are met or no columns are selected (button remains disabled in run_sort)
        self.sort_btn.setEnabled(True)
        if len(self.sort_columns) > 0:
             self.status_label.setText(f"Ready to sort {len(self.current_df)} rows with {selected_algo}.")
        else:
            self.status_label.setText(f"Ready to sort {len(self.current_df)} rows.")

    def run_sort(self):
        selected_algo = self.algo_combo.currentText()
        column_names = self.sort_columns

        if self.current_df.empty:
            QMessageBox.warning(self, "Sort Error", "No data loaded to sort.")
            return

        if not column_names:
            QMessageBox.warning(self, "Sort Error", "Select at least one column to sort by (Double-click header).")
            return

        if not self.sort_btn.isEnabled():
             QMessageBox.critical(self, "Sort Error", "Sorting constraints violated. Check status bar for details.")
             return

        try:
            # The entire (filtered) current_df is sorted
            sorted_df, time_ms, complexity = self.sorter.execute_sort(
                self.current_df.copy(), column_names, selected_algo
            )

            # Update the UI
            self.load_data_to_table(sorted_df)
            # The result from execute_sort is already a copy, making it the new current_df
            self.current_df = sorted_df
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

    # --- SEARCHING METHODS ---
    def add_filter_row(self, columns: List[str]):
        """Adds a new filter row widget to the composite filter area."""
        new_row = FilterRow(columns)
        self.filter_rows.append(new_row)
        self.filter_container_layout.insertWidget(self.filter_container_layout.count() - 1, new_row)

    def apply_single_filter(
        self, df: pd.DataFrame, col_name: str, operator: str, value: str
    ) -> pd.Series:
        """Returns a boolean mask for a single filter condition."""
        if col_name not in df.columns or not value:
            return pd.Series([True] * len(df))

        column = df[col_name]

        # 1. Numerical Comparison
        if operator in ["==", ">", "<", "!="]:
            try:
                num_col = pd.to_numeric(column, errors="coerce")
                num_val = float(value)
                # Apply mask only where conversion was successful (not NaN)
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
                
                # For '==' and '!=', compare string where numeric conversion failed
                if not valid_mask.all():
                    str_col = column.astype(str)
                    if operator == "==":
                        result_mask[~valid_mask] = str_col[~valid_mask] == value
                    elif operator == "!=":
                        result_mask[~valid_mask] = str_col[~valid_mask] != value
                    # >, < remain False for non-numeric part

                return result_mask
            except ValueError:
                # Fallback: if value cannot be converted to float, only string comparison is possible for ==/!=
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
        self.complexity_label.setText("Complexity: N/A (Searching)")


def main():
    sys.setrecursionlimit(5000)
    app = QApplication(sys.argv)
    ex = SortingApp()
    ex.show()
    sys.exit(app.exec_())


"""///////////////////////////////////////////////////////////////"""
if __name__ == "__main__":
    main()