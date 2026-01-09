# ===============================================================================
# File: regulatory/core/utils/Fda_data_insert_milvus.py
# Description: Utility to insert FDA standardized device names into Milvus vector store
# Author: Anmol Kushwaha
# ===============================================================================
import os
import json
import time
import hashlib
from dotenv import load_dotenv
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Iterator, Tuple, Set

# Load environment variables
load_dotenv()

# Read from environment
MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "fda_devices")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))  # Process in batches
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))  # Retry failed operations
DATA_FILE_PATH = os.getenv("FDA_DATA_FILE", os.path.join(os.path.dirname(__file__), "fda_devices_data.json"))

if not MILVUS_URI:
    raise ValueError(" MILVUS_URI is not set in .env file")


def create_device_id(device_name: str, category: str) -> str:
    """Create unique ID for device based on name and category"""
    unique_string = f"{device_name}|{category}".lower().strip()
    return hashlib.md5(unique_string.encode('utf-8')).hexdigest()


def load_fda_device_data() -> Dict[str, List[str]]:
    """Load FDA device data from JSON file"""
    try:
        with open(DATA_FILE_PATH, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print(f"✓ Loaded device data from: {DATA_FILE_PATH}")
            return data
    except FileNotFoundError:
        raise FileNotFoundError(f"FDA device data file not found: {DATA_FILE_PATH}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {DATA_FILE_PATH}: {e}")
    except Exception as e:
        raise Exception(f"Error loading FDA device data: {e}")



def get_existing_device_ids(vector_store: Milvus) -> Set[str]:
    """Get set of existing device IDs from Milvus collection"""
    try:
        # Try to get all documents to check for existing device IDs
        existing_docs = vector_store.similarity_search(
            query="device", 
            k=10000,  # Get many docs to check for duplicates
            param={"metric_type": "L2", "params": {"nprobe": 10}}
        )
        
        existing_ids = set()
        for doc in existing_docs:
            if hasattr(doc, 'metadata') and doc.metadata:
                device_name = doc.metadata.get('device_name', '')
                category = doc.metadata.get('category', '')
                if device_name and category:
                    device_id = create_device_id(device_name, category)
                    existing_ids.add(device_id)
        
        print(f" Found {len(existing_ids)} existing devices in collection")
        return existing_ids
        
    except Exception as e:
        print(f"  Could not check existing devices: {e}")
        print(" Assuming collection is empty or first-time insertion")
        return set()


def generate_device_data(fda_data: Dict[str, List[str]], existing_ids: Set[str]) -> Iterator[Tuple[str, Dict]]:
    """Generator to yield device data one at a time - memory efficient with duplicate filtering"""
    skipped_count = 0
    for category, devices in fda_data.items():
        for device in devices:
            device_id = create_device_id(device, category)
            
            # Skip if device already exists
            if device_id in existing_ids:
                skipped_count += 1
                continue
                
            enriched_text = (
                f"{device}. Category: {category}. "
                f"This is a medical device used in {category.lower()} procedures."
            )
            metadata = {
                "device_name": device,
                "category": category,
                "type": "FDA standardized name",
                "device_id": device_id  # Add unique ID to metadata
            }
            yield enriched_text, metadata
    
    if skipped_count > 0:
        print(f" Skipped {skipped_count} duplicate devices")


def batch_generator(data_generator: Iterator, batch_size: int) -> Iterator[Tuple[List[str], List[Dict]]]:
    """Group data into batches for efficient processing"""
    texts_batch = []
    metadatas_batch = []
    
    for text, metadata in data_generator:
        texts_batch.append(text)
        metadatas_batch.append(metadata)
        
        if len(texts_batch) >= batch_size:
            yield texts_batch, metadatas_batch
            texts_batch = []
            metadatas_batch = []
    
    # Yield remaining items if any
    if texts_batch:
        yield texts_batch, metadatas_batch


def insert_with_retry(vector_store: Milvus, texts: List[str], metadatas: List[Dict]) -> bool:
    """Insert data with retry logic for failed operations"""
    for attempt in range(MAX_RETRIES):
        try:
            vector_store.add_texts(texts=texts, metadatas=metadatas)
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to insert batch after {MAX_RETRIES} attempts")
                return False
    return False


def prepare_data():
    """Optimized data preparation with batch processing and error handling"""
    
    # Load FDA device data from JSON file
    try:
        fda_device_data = load_fda_device_data()
    except Exception as e:
        print(f"Error loading device data: {e}")
        return False
    
    # Count total devices for progress tracking
    total_devices = sum(len(devices) for devices in fda_device_data.values())
    print(f"Starting insertion of {total_devices} FDA device names...")
    print(f"Batch size: {BATCH_SIZE}, Collection: {MILVUS_COLLECTION}")
    
    # Initialize embeddings and vector store
    try:
        # Debug: Check if OpenAI API key is loaded
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print(" OPENAI_API_KEY not found in environment variables")
            return False
        else:
            print(f"✓ OpenAI API Key loaded (ends with: ...{openai_key[-4:]})")
        
        embeddings = OpenAIEmbeddings()
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=MILVUS_COLLECTION,
            connection_args={"uri": MILVUS_URI},
            auto_id=True
        )
    except Exception as e:
        print(f"Failed to initialize Milvus connection: {e}")
        return False

    # Check for existing devices to prevent duplicates
    print("🔍 Checking for existing devices to prevent duplicates...")
    existing_device_ids = get_existing_device_ids(vector_store)
    
    # Count new devices to be inserted (excluding duplicates)
    new_devices_count = 0
    for category, devices in fda_device_data.items():
        for device in devices:
            device_id = create_device_id(device, category)
            if device_id not in existing_device_ids:
                new_devices_count += 1
    
    if new_devices_count == 0:
        print(" All devices already exist in the collection. No new insertions needed.")
        return True
    
    print(f" Found {new_devices_count} new devices to insert (out of {total_devices} total)")

    # Process data in batches
    processed_count = 0
    failed_batches = 0
    start_time = time.time()
    
    try:
        data_gen = generate_device_data(fda_device_data, existing_device_ids)
        batch_gen = batch_generator(data_gen, BATCH_SIZE)
        
        for batch_num, (texts_batch, metadatas_batch) in enumerate(batch_gen, 1):
            print(f"Processing batch {batch_num} ({len(texts_batch)} devices)...")
            
            if insert_with_retry(vector_store, texts_batch, metadatas_batch):
                processed_count += len(texts_batch)
                print(f"✓ Batch {batch_num} successful. Progress: {processed_count}/{total_devices}")
            else:
                failed_batches += 1
                print(f"✗ Batch {batch_num} failed.")
            
            # Small delay between batches to avoid overwhelming the server
            time.sleep(0.1)
    
    except Exception as e:
        print(f"Unexpected error during processing: {e}")
        return False
    
    # Summary
    elapsed_time = time.time() - start_time
    duplicates_skipped = total_devices - new_devices_count
    print("\n=== INSERTION COMPLETE ===")
    print(f"✓ Successfully inserted: {processed_count}/{new_devices_count} new devices")
    print(f" Duplicates skipped: {duplicates_skipped}")
    print(f"✗ Failed batches: {failed_batches}")
    print(f" Total time: {elapsed_time:.2f} seconds")
    print(f" Categories: {list(fda_device_data.keys())}")
    print(f" Data source: {DATA_FILE_PATH}")
    print("  Duplicate prevention: ENABLED")
    
    return processed_count > 0


if __name__ == "__main__":
    success = prepare_data()
    if not success:
        exit(1)
