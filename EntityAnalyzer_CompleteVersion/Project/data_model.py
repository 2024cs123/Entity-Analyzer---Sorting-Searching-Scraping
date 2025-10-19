import pandas as pd
import numpy as np
from typing import List, Tuple

from PyQt5.QtCore import QAbstractTableModel, Qt, QVariant, QModelIndex, QObject, pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QComboBox, QLineEdit, QPushButton

class PaginatedPandasModel(QAbstractTableModel):
    """A data model for QTableView that wraps a Pandas DataFrame and implements pagination."""

    ROWS_PER_PAGE = 5000

    def __init__(self, data: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self.full_data = data
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]
        self.total_pages = int(np.ceil(self.total_rows / self.ROWS_PER_PAGE))
        self._page_data = pd.DataFrame()
        self.update_page_data()
        
    def update_data(self, data: pd.DataFrame):
        self.beginResetModel()
        self.full_data = data
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]
        self.total_pages = int(np.ceil(self.total_rows / self.ROWS_PER_PAGE))
        if self.total_rows == 0:
            self.total_pages = 0
            self._page_data = pd.DataFrame()
        else:
            self.update_page_data()
        self.endResetModel()

    def update_page_data(self):
        if self.total_rows == 0:
            self._page_data = pd.DataFrame()
            return
        start_row = self.page_number * self.ROWS_PER_PAGE
        end_row = min((self.page_number + 1) * self.ROWS_PER_PAGE, self.total_rows)
        self._page_data = self.full_data.iloc[start_row:end_row]

    def set_page(self, page_num):
        if 0 <= page_num < self.total_pages:
            self.page_number = page_num
            self.update_page_data()
            self.layoutChanged.emit()
            return True
        return False

    def rowCount(self, parent=QModelIndex()):
        return self._page_data.shape[0]

    def columnCount(self, parent=QModelIndex()):
        return self._page_data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid(): return QVariant()
        if role == Qt.DisplayRole:
            if index.row() < self._page_data.shape[0] and index.column() < self._page_data.shape[1]:
                return str(self._page_data.iloc[index.row(), index.column()])
            return QVariant()
        return QVariant()

    def headerData(self, col, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._page_data.columns[col])
        return QVariant()


class FilterRow(QWidget):
    """A reusable widget for a single filter condition with a remove button."""

    def __init__(self, column_names: List[str], remove_callback, parent=None):
        super().__init__(parent)
        self.remove_callback = remove_callback
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
        
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove_clicked)

        self.layout.addWidget(self.col_combo)
        self.layout.addWidget(self.op_combo)
        self.layout.addWidget(self.val_input)
        self.layout.addWidget(self.remove_btn)
        
    def _on_remove_clicked(self):
        """Calls the main application's method to safely remove this row."""
        self.remove_callback(self)

    def get_filter_data(self) -> Tuple[str, str, str]:
        return (
            self.col_combo.currentText(),
            self.op_combo.currentText(),
            self.val_input.text().strip(),
        )