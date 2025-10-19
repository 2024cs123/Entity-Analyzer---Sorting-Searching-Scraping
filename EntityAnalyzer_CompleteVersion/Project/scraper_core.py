import time
import os
from typing import List
from urllib.parse import urlparse
from io import StringIO

import pandas as pd
import selenium.webdriver as webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup

# Tries to import LangChain/Ollama components. If they fail, the parsing function will raise an error.
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_community.llms import Ollama
except ImportError:
    class Ollama:
        def __init__(self, model): pass
    def parse_with_ollama(*args):
        raise ImportError("LangChain or Ollama not found. AI Scraper requires 'langchain_community' and a running Ollama server.")
    
# --- Core Scraper/Parser Functions ---

def is_valid_url(url: str) -> bool:
    try:
        result = urlparse(url.strip())
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def scrape_website(url: str) -> str:
    print("Launching chrome browser (headless)...")
    
    if not is_valid_url(url):
        raise ValueError(f"Invalid or malformed URL: {url!r}")
    
    # NOTE: ChromeDriver must be accessible from the system's PATH.
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    try:
        # Use Service if you need to explicitly point to a driver path, 
        # otherwise, rely on PATH lookup which is common practice.
        driver = webdriver.Chrome(options=options) 
    except Exception as e:
        raise Exception(f"Failed to initialize ChromeDriver. Ensure it is installed and in your PATH. Error: {e}")

    try:
        driver.get(url.strip())
        time.sleep(3)  # Waiting for page to load JS/AJAX
        html = driver.page_source
        return html
    finally:
        driver.quit()

def extract_body_content(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')    
    body = soup.body
    if body:
        return str(body)
    return ""

def clean_body_content(body_html: str) -> str:
    soup = BeautifulSoup(body_html, 'html.parser')
    
    for script_or_style in soup(["script", "style"]):
        script_or_style.extract()
    
    clean_body_content = soup.get_text(separator='\n')
    clean_body_content = '\n'.join([line.strip() for line in clean_body_content.splitlines() if line.strip()])
    return clean_body_content

def split_dom_content(text: str, chunk_size: int = 4000) -> List[str]:
    """Splits the clean text content into manageable chunks for the LLM."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def parse_with_ollama(dom_chunks: List[str], parse_description: str, model_name="mistral") -> str:
    """Parses chunks using Ollama/Mistral via LangChain, returning a merged CSV string."""
    model = Ollama(model=model_name)
    template = (
                "You are tasked with extracting specific information from the following text content: {dom_chunk}." 
                "Please follow these instructions carefully : \n\n"
                "1. **Extraction Task:** Extract only the information that directly matches the provided description: {parse_description}."
                "2. **Output Format:** Output the extracted data in a clean, consistent **CSV format** (comma-separated values) with headers included in the first line."
                "3. **No Extra Text:** Do not include any additional text, comments, or explanations in your response, only the CSV data."
                "4. **Empty Response:** If the content does not contain relevant information, respond only with 'No relevant information found.'"
                )
    
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    parsed_results = []
    
    for i, chunk in enumerate(dom_chunks, start=1):
            response = chain.invoke({
                "dom_chunk": chunk, "parse_description": parse_description})
            
            print(f"Processed chunk {i}/{len(dom_chunks)}")
            parsed_results.append(response.strip())
            
    # Combing results, removing any 'No relevant information found' messages
    combined_results = [res for res in parsed_results if res != 'No relevant information found.' and res]

    if not combined_results:
         return "No relevant information found."
         
    # Logic to merge CSV results: Take headers from the first chunk, append data from all
    first_chunk = combined_results[0].splitlines()
    if not first_chunk: return "Extraction failed."

    headers = first_chunk[0]
    data_lines = first_chunk[1:]
    
    for res in combined_results[1:]:
        # Skiping header line in subsequent chunks if they exist
        res_lines = res.splitlines()
        if len(res_lines) > 1 and res_lines[0].lower() == headers.lower():
             data_lines.extend(res_lines[1:])
        else:
             data_lines.extend(res_lines)


    return headers + "\n" + "\n".join(data_lines)