import sys
import time
import pandas as pd
import numpy as np
from typing import List, Tuple
from PyQt5.QtCore import (

    QObject,
    pyqtSignal,
    QRunnable
)
from PyQt5.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QComboBox,
    QLineEdit
)

# ====================================================================
# 4. SCRAPING THREADING & COMPOSITE FILTER UI
# ====================================================================

class ScraperWorkerSignals(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(int,list) 
    error = pyqtSignal(str) 
    
class ScraperWorker(QRunnable):
    """Worker thread to perform scraping logic (using dummy data)."""
    
    def __init__(self,start_url,target_count=25000):
        super().__init__()
        self.signals = ScraperWorkerSignals()
        self.url = start_url
        self.target_count = target_count
        self.is_running = True
        self.is_paused = False
        self.scraped_data=[]
        
        self.headers = []
        
    def run(self):
        pass
    def stop(self):
        self.is_running = False

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False
    
class FilterRow(QWidget):
    """A reusable widget for a single filter condition."""
    def __init__(self,column_names:List[str],parent=None):
        super.__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0,0,0,0)
        
        self.col_combo = QComboBox()
        self.col_combo.addItems(column_names)
        
        self.op_combo = QComboBox
        self.op_combo.addItems(['==', '>', '<', '!=', 'contains', 'starts with', 'ends with'])
        
        self.val_input = QLineEdit()
        self.val_input.setPlaceHolderText("Value")
        
        self.layout.addWidget(self.col_combo)
        self.layout.addWidget(self.op_combo)
        self.layout.addWidget(self.val_input)
    
    def get_filter_data(self)->Tuple[str,str,str]:
        return self.col_combo.currentText(),self.op_combo.currentText(),self.val_input.text().strip()  
       
            