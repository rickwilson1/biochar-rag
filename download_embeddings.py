#!/usr/bin/env python3
"""
Download embeddings from Google Drive for Render deployment
"""

import os
import requests
import zipfile
import logging
from pathlib import Path
import time

logger = logging.getLogger(__name__)

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive using its file ID."""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    
    # Handle large file confirmation
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    # Save file
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)
    
    return True

def download_and_extract_embeddings():
    """Download embeddings zip from Google Drive and extract."""
    
    # Google Drive file ID (you'll need to set this)
    EMBEDDINGS_DRIVE_ID = os.getenv("EMBEDDINGS_DRIVE_ID")
    
    if not EMBEDDINGS_DRIVE_ID:
        logger.error("EMBEDDINGS_DRIVE_ID environment variable not set")
        return False
    
    # Create embeddings directory
    embeddings_dir = Path("embeddings_output")
    embeddings_dir.mkdir(exist_ok=True)
    
    # Check if files already exist
    required_files = [
        "biochar_embeddings.npy",
        "biochar_chunks_with_embeddings.csv", 
        "embedding_metadata.json"
    ]
    
    if all((embeddings_dir / f).exists() for f in required_files):
        logger.info("Embeddings already exist, skipping download")
        return True
    
    logger.info("Downloading embeddings from Google Drive...")
    
    try:
        # Download zip file
        zip_path = "embeddings.zip"
        download_file_from_google_drive(EMBEDDINGS_DRIVE_ID, zip_path)
        logger.info("Download completed")
        
        # Extract zip file
        logger.info("Extracting embeddings...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        
        # Clean up zip file
        os.remove(zip_path)
        
        # Verify required files exist
        missing_files = [f for f in required_files if not (embeddings_dir / f).exists()]
        if missing_files:
            logger.error(f"Missing files after extraction: {missing_files}")
            return False
        
        logger.info("Embeddings successfully downloaded and extracted")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download embeddings: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = download_and_extract_embeddings()
    if not success:
        exit(1)