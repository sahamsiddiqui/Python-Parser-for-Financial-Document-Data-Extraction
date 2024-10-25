# Python-Parser-for-Financial-Document-Data-Extraction


### Project Overview
This project provides a Python-based parser to extract financial information from EDGAR HTML filings, specifically targeting Earnings Per Share (EPS) data. The parser processes raw HTML files, identifies relevant financial metrics, and outputs structured data for further analysis.

### Project Structure
- **`Data`**  
   Contains example HTML filings for parsing and testing.

- **`main2.py`**  
   The main script for parsing HTML filings and extracting targeted financial data.

- **`requirements.txt`**  
   Specifies necessary Python libraries and dependencies to run the project.

- **`output_csv/`**  
   Folder where parsed data (e.g., EPS values) is saved as CSV files.

---

## Getting Started

Follow these steps to set up and run the parser on your local machine:

### Prerequisites
- Python 3.x

### Setup and Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/edgar-financial-filings-parser.git
   cd edgar-financial-filings-parser
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Parser
Execute the main parsing script to process sample HTML files in the `Training_Filings` directory:
```bash
python main2.py
```

The results will be saved in `output_csv/` as structured CSV files.

