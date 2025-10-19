# Entity Analyzer (Sorting, Searching & AI Scraping)

## Overview

The Entity Analyzer is a PyQt-based desktop application designed for data management, analysis, and intelligent web scraping. It provides a robust interface for loading tabular data (CSV/Pandas DataFrames), applying various sorting algorithms (including custom implementations and highly optimized native Pandas sorting), powerful filtering, and leveraging a Local Large Language Model (LLM) via **Ollama** and **Selenium** to scrape and parse structured data from websites.

## Features

* **Data Management**: Load, display, and paginate large datasets from CSV files.
* **Sorting Algorithms**: Compare performance of classic sorts (Bubble, Quick, Merge, Counting, Radix, Bucket) against optimized Pandas Timsort.
* **Dynamic Filtering**: Apply complex multi-condition filters (AND/OR) using a custom filter row component.
* **AI Web Scraping**: Use a modal configuration window to:
    * Scrape dynamic websites using **Selenium (Chrome/Chromedriver)**.
    * Extract unstructured text content.
    * Prompt a local LLM (e.g., **Mistral via Ollama**) to parse the text into a structured CSV/DataFrame.
* **Performance Tracking**: Measures and displays the execution time (in milliseconds) and theoretical complexity (Big O) for sorting operations.
* **Asynchronous Processing**: Uses `QThreadPool` to ensure the GUI remains responsive during the long-running web scraping and LLM parsing tasks.

## Prerequisites

Before running the application, you must have the following installed and configured:

1.  **Python 3.8+**
2.  **Ollama**: Install and run the Ollama server (e.g., `ollama run mistral`). The AI scraper is currently configured to use the `mistral` model.
3.  **Chromedriver**: The Selenium component requires the **Chrome browser** and its corresponding **Chromedriver** executable to be installed and accessible in your system's PATH.

## Installation

1.  **Clone the Repository (or ensure all files are in the same directory):**
    ```bash

     # Full Project progress at:
    
    git clone https://github.com/2024cs123/Entity-Analyzer---Sorting-Searching-Scraping.git
    
    cd  Entity-Analyzer---Sorting-Searching-Scraping\EntityAnalyzer_CompleteVersion

    ```

2.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure

| File | Description |
| :--- | :--- |
| `main_app.py` | Contains the main PyQt `QMainWindow` (`SortingApp`) and the application entry point. Ties all components together. |
| `sort_core.py` | Core logic for sorting. Contains `SortingLibrary` (algorithms) and `SortExecutor` (Pandas wrapper/timer). |
| `data_model.py` | PyQt model definitions. Contains `PaginatedPandasModel` and `FilterRow` widget. |
| `scraper_core.py` | Pure, non-GUI functions for AI scraping: `scrape_website`, `clean_body_content`, `parse_with_ollama`, etc. |
| `scraper_gui.py` | PyQt threading and UI components for scraping: `AIScraperWorker` (runs in background) and `AIScraperDialog` (user input). |
| `requirements.txt` | Lists all necessary Python dependencies. |

## Usage

1.  **Run the Main Application:**
    ```bash
    python main_app.py
    ```

2.  **Basic Data Operations:**
    * Use the **"Load CSV File"** button to load your data.
    * Use the **Filter** panel to add search conditions and click **"Apply Filter"**.
    * **Double-click a column header** in the table to select it for sorting.
    * Select an **Algorithm** from the dropdown and click **"Run Sort"**.

3.  **AI Web Scraping:**
    * Click the **"Start AI Web Scraper (LLM/Ollama)"** button.
    * Enter the **Target URL** (e.g., a Wikipedia page with a table).
    * Enter a detailed **Parse Description** prompt (e.g., "Extract the name and time complexity of all algorithms and output in CSV format with headers.").
    * Click **"Start AI Scraper"**. The task will run in the background, updating the progress bar.
    * Once complete, the scraped and parsed data will replace the current table view.

## License

This project is open-source. Please refer to the LICENSE file for details.