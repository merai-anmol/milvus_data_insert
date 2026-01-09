# FDA Device Names Milvus Insert

A utility to insert FDA standardized device names into a Milvus vector database with OpenAI embeddings. This tool processes a JSON file containing FDA device categorization data and stores it in a Milvus vector store for semantic search capabilities.

## Features

- **Batch Processing**: Efficiently processes device data in configurable batches
- **Duplicate Prevention**: Automatically detects and skips existing devices to prevent duplicates
- **Retry Logic**: Implements exponential backoff for failed insertions
- **Rich Metadata**: Stores device name, category, and unique device IDs alongside embeddings
- **Memory Efficient**: Uses generators for processing large datasets without loading everything into memory
- **Comprehensive Logging**: Detailed progress tracking and error reporting

## Project Structure

```
.
├── FDA_data_insert_milvus.py      # Main utility script
├── fda_devices_data.json           # FDA device names organized by category
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Prerequisites

- Python 3.8+
- Milvus server running and accessible
- OpenAI API key

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with the following configuration:

```env
MILVUS_URI=http://localhost:19530
MILVUS_COLLECTION=fda_devices
OPENAI_API_KEY=your_openai_api_key_here
BATCH_SIZE=50
MAX_RETRIES=3
FDA_DATA_FILE=./fda_devices_data.json
```

## Configuration

Environment variables can be set in a `.env` file:

| Variable            | Default                   | Description                                 |
| ------------------- | ------------------------- | ------------------------------------------- |
| `MILVUS_URI`        | Required                  | URI for Milvus server connection            |
| `MILVUS_COLLECTION` | `fda_devices`             | Name of the Milvus collection               |
| `OPENAI_API_KEY`    | Required                  | OpenAI API key for embeddings               |
| `BATCH_SIZE`        | `50`                      | Number of devices to process per batch      |
| `MAX_RETRIES`       | `3`                       | Number of retry attempts for failed batches |
| `FDA_DATA_FILE`     | `./fda_devices_data.json` | Path to the FDA device data JSON file       |

## Usage

Run the script to insert FDA device names into Milvus:

```bash
python FDA_data_insert_milvus.py
```

### Example Output

```
✓ Loaded device data from: ./fda_devices_data.json
Starting insertion of 1200 FDA device names...
Batch size: 50, Collection: fda_devices
✓ OpenAI API Key loaded (ends with: ...xxxx)
🔍 Checking for existing devices to prevent duplicates...
✓ Found 0 existing devices in collection
✓ Found 1200 new devices to insert (out of 1200 total)
Processing batch 1 (50 devices)...
✓ Batch 1 successful. Progress: 50/1200
...

=== INSERTION COMPLETE ===
✓ Successfully inserted: 1200/1200 new devices
✓ Duplicates skipped: 0
✗ Failed batches: 0
✓ Total time: 45.23 seconds
```

## Data Format

The `fda_devices_data.json` file contains FDA device names organized by category:

```json
{
  "Diagnostic Devices": [
    "Automated cell counter.",
    "Automated differential cell counter.",
    "Red cell indices device.",
    ...
  ],
  "Surgical Devices": [
    ...
  ],
  ...
}
```

## Functions

### `create_device_id(device_name, category)`

Creates a unique MD5 hash-based ID for a device based on its name and category.

### `load_fda_device_data()`

Loads FDA device data from the JSON file with error handling.

### `get_existing_device_ids(vector_store)`

Retrieves existing device IDs from the Milvus collection to prevent duplicates.

### `generate_device_data(fda_data, existing_ids)`

Generator function that yields enriched device data, filtering out duplicates.

### `batch_generator(data_generator, batch_size)`

Groups device data into batches for efficient batch processing.

### `insert_with_retry(vector_store, texts, metadatas)`

Inserts data with exponential backoff retry logic.

### `prepare_data()`

Main orchestration function that handles the complete insertion workflow.

## Error Handling

- **Missing Configuration**: Validates required environment variables before execution
- **File Errors**: Handles missing or malformed JSON data files gracefully
- **Connection Errors**: Implements retry logic with exponential backoff for transient failures
- **Duplicate Prevention**: Automatically skips devices already in the collection

## Performance Considerations

- Batch size affects memory usage and insertion speed. Adjust `BATCH_SIZE` based on your system resources
- Retry attempts can slow down processing. Adjust `MAX_RETRIES` based on network stability
- The script uses generators to minimize memory footprint for large datasets

## Troubleshooting

### MILVUS_URI is not set

Ensure your `.env` file contains `MILVUS_URI` environment variable.

### OPENAI_API_KEY not found

Check that `OPENAI_API_KEY` is properly set in your `.env` file.

### Failed to initialize Milvus connection

Verify that your Milvus server is running and accessible at the configured URI.

### FDA device data file not found

Ensure `fda_devices_data.json` exists in the specified path.

## Author

Anmol Kushwaha

## License

See LICENSE file for details.
