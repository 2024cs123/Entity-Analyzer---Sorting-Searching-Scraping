from PyQt5.QtCore import (
    QAbstractTableModel,
    Qt,
    QVariant,
    QModelIndex
)


# for Pagination
from PyQt5.QtCore import QSize
import pandas as pd
import numpy as np

# ====================================================================
# 3. PYQT MODEL/VIEW WITH PAGINATION
# ====================================================================
 
class PaginatedPandasModel(QAbstractTableModel):
    """A data model for QTableView that wraps a Pandas DataFrame and implements pagination."""
              
    ROWS_PER_PAGE = 5000 # Max rows to display per page
    
    def __init__(self,data:pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self,parent)
        self.full_data = data
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]  
        self.total_pages = int(np.ceil(self.total_rows/self.ROWS_PER_PAGE)) 
        self.update_page_data()
        
    def update_data(self,data=pd.DataFrame):
        """Called when uderlying DataFrame is sorted or filtered."""    
        self.beginResetModel()
        self.page_number = 0
        self.total_rows = self.full_data.shape[0]  
        self.total_pages = int(np.ceil(self.total_rows/self.ROWS_PER_PAGE)) 
        if self.page_number >= self.total_pages:
            self.page_number = max(0,self.total_pages-1)
        self.update_page_data()
        self.endResetModel()
        
    def update_page_data(self):
        """Calcualate the slice of the DataFrame for the current page."""
        start_row=self.page_number*self.ROWS_PER_PAGE
        end_row = min((self.page_number+1)*self.ROWS_PER_PAGE,self.total_rows) 
        self._page_data = self.full_data.iloc [start_row:end_row]    
        
    def set_page(self,page_num):
        """Changes the current page."""
        if 0 <=page_num <self.total_pages:
            self.page_number = page_num    
            self.update_page_data()
            self.layoutChanged.emit() # Signal to the view to redraw
            return True
        return False
    
    def rowCount(self,parent=QModelIndex()):
        return self._page_data.shape([0])
    
    def columnCount(self, parent=QModelIndex()):
        return self._page_data.shape[1]
    
    def data(self,index,role=Qt.DisplayRole):
        if not index.isvalid():
            return QVariant()
        
        if role==Qt.DisplayRole:
            return str(self._page_data.iloc[index.row(),index.column()])
        return QVariant()
    
    def headerData(self,col,orientation,role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return str(self._page_data.columns[col])
        return QVariant()

