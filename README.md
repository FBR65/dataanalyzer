# 🚀 Enhanced Data Analyzer - Excel to DuckDB/DuckLake Transformation

A powerful Python application for analyzing data from multiple sources including SQL databases, MongoDB, and Excel files with automated transformation to DuckDB/DuckLake and Tableau-like data exploration capabilities.

## ✨ Enhanced Features

### 📊 Excel-to-DuckDB/DuckLake Pipeline
- **Automated Excel Processing**: Bulk import of Excel files
- **DuckDB Integration**: High-performance columnar database
- **DuckLake Support**: Parquet-based data lake with DuckDB integration
- **Intelligent Data Cleaning**: Automatic column name cleaning and data type detection
- **Metadata Tracking**: Complete tracking of processed files

### 🎯 PyWalker Integration
- **Tableau-like Exploration**: Drag-and-drop data exploration
- **Interactive Visualizations**: Advanced chart types and configuration options
- **Exploratory Data Analysis**: Intuitive user interface for business users

### 🔧 Enhanced Analysis Functions
- **Automatic Insights**: AI-powered data analysis and recommendations
- **Advanced Visualizations**: Plotly-based interactive charts
- **Flexible Data Exports**: Multiple formats (CSV, Excel, Parquet, JSON)
- **Performance-optimized Queries**: DuckDB-optimized SQL queries

### 🗄️ Multi-Database Support
- **SQL databases**: PostgreSQL, MySQL, MariaDB, Oracle, SQLite
- **MongoDB integration**: Document database support
- **DuckDB**: Columnar analytical database
- **Automatic database schema detection**

### 🤖 Natural Language Processing
- **Convert natural language queries** to SQL/MongoDB queries
- **Support for German and English queries**
- **Uses LLM models for query generation**
- **AI-powered data insights**

### 📈 Data Visualization
- **Automatic plot generation** with matplotlib and Plotly
- **Dynamic visualization** based on data structure
- **PyWalker integration** for interactive exploration
- **Base64 encoded image output**

### 🌐 Interactive Interface
- **Enhanced Gradio web interface**
- **Real-time connection testing**
- **Visual query results**
- **PyWalker embedded interface**

## 🏗️ Architecture

```
Enhanced Data Analyzer/
├── 📁 Excel Processing Layer
│   ├── ExcelProcessor (tools/excel_processor.py)
│   ├── Automatic data detection
│   └── Batch processing
├── 🗄️ DuckDB/DuckLake Layer
│   ├── DuckDBConnector (tools/duckdb_connector.py)
│   ├── Parquet-based storage
│   └── Advanced SQL functions
├── 🎨 Frontend Layer
│   ├── Enhanced Gradio Interface
│   ├── PyWalker Integration
│   └── Advanced Visualizations
└── 🐳 Containerization
    ├── Docker + Docker Compose
    ├── Ollama LLM Support
    └── Optional Services (MongoDB, MariaDB, Jupyter, Metabase)
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd dataanalyzer

# Set environment variables (optional)
cp .env.example .env

# Start services
docker-compose up -d

# Wait until all services are ready
docker-compose logs -f dataanalyzer
```

**Access Services:**
- **Data Analyzer**: http://localhost:7860
- **Jupyter Notebook**: http://localhost:8888 (Token: dataanalyzer123)
- **Metabase**: http://localhost:3000
- **Ollama**: http://localhost:11434

### Option 2: Local Installation
```bash
# Install dependencies
uv sync

# Create directories
mkdir -p excel_files databases ducklake generated_code logs

# Start enhanced frontend
uv run python enhanced_frontend.py
```

## Requirements

### Core Dependencies
```bash
uv sync
```

### Additional Dependencies for Enhanced Features
```bash
# DuckDB and data processing
uv add duckdb pygwalker plotly

# Excel processing
uv add openpyxl xlrd

# Enhanced visualizations
uv add seaborn plotly-express
```

### Database Drivers
```bash
# PostgreSQL
uv add psycopg2-binary

# MySQL/MariaDB
uv add mysql-connector-python

# Oracle
uv add oracledb

# For LLM Integration
uv add openai
```

## Configuration

### Environment Variables
```bash
# LLM Configuration
OPENAI_API_KEY=ollama                           # Default for local LLM
OPENAI_API_BASE_URL=http://localhost:11434/v1   # Ollama API endpoint
LLM_MODEL=qwen2.5:latest                        # Base LLM model
CODING_MODEL=qwen2.5-coder:latest               # Code-specific LLM model

# Database Configuration
DUCKDB_PATH=./databases/main.duckdb
DUCK_LAKE_PATH=./ducklake

# Interface Configuration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

### Excel Processing Configuration
```python
processor = ExcelProcessor(
    source_dir="./excel_files",
    duck_lake_path="./ducklake",
    use_parquet=True,        # Parquet for better performance
    auto_clean=True,         # Automatic data cleaning
    max_file_size_mb=100     # Maximum file size
)
```

## Project Structure

```
dataanalyzer/
├── main.py                    # Core application logic
├── gradio_frontend.py         # Basic web interface
├── enhanced_frontend.py       # Enhanced web interface with PyWalker
├── scheduler.py               # Automated Excel processing scheduler
├── tools/
│   ├── database.py            # SQL database connector
│   ├── duckdb_connector.py    # DuckDB connector and operations
│   ├── excel_processor.py     # Excel file processing and transformation
│   ├── mongodb.py             # MongoDB connector
│   └── pythontools.py         # Python REPL and visualization
├── excel_files/               # Source Excel files
├── databases/                 # DuckDB database files
├── ducklake/                  # Parquet-based data lake
├── generated_code/            # Auto-generated visualization code
├── logs/                      # Application logs
├── notebooks/                 # Jupyter notebooks
├── docker-compose.yml         # Docker composition
├── Dockerfile                 # Docker image definition
└── README.md
```

## 💡 Usage

### 1. Basic Web Interface
```bash
python gradio_frontend.py
```

### 2. Enhanced Interface with PyWalker
```bash
python enhanced_frontend.py
```

### 3. Process Excel Files
```python
from tools.excel_processor import ExcelProcessor

# Initialize processor
processor = ExcelProcessor(
    source_dir="./excel_files",
    duck_lake_path="./ducklake",
    duckdb_path="./databases/excel_data.duckdb"
)

# Process all Excel files
results = processor.process_directory(recursive=True)
print(f"Processed: {results['processed']} files")
```

### 4. Query Data with DuckDB
```python
from tools.duckdb_connector import DuckDBConnector

# Initialize connector
connector = DuckDBConnector(
    duckdb_path="./databases/excel_data.duckdb",
    duck_lake_path="./ducklake"
)

# Show available tables
tables = connector.list_tables()
print(f"Available tables: {[t['table_name'] for t in tables]}")

# Query data
df = connector.execute_query("SELECT * FROM my_table LIMIT 100")
print(df.head())
```

### 5. Connect to Databases
- Fill in connection details in the web interface
- Test connection
- Choose database type (SQL/MongoDB/DuckDB)

### 6. Natural Language Queries
```
Example queries:
- "Show me the average age of users by city"
- "What are the top 5 products by sales?"
- "Create a chart showing monthly revenue trends"
```

## 🎯 Workflow Examples

### Example 1: Automated Sales Data Analysis
```python
# 1. Process Excel sales data
processor = ExcelProcessor("./sales_data")
results = processor.process_directory()

# 2. Analyze data
connector = DuckDBConnector("databases/sales.duckdb")
sales_summary = connector.execute_query("""
    SELECT 
        region,
        SUM(sales_amount) as total_sales,
        COUNT(*) as transactions
    FROM sales_data 
    GROUP BY region
    ORDER BY total_sales DESC
""")

# 3. Generate insights
insights = connector.generate_insights("sales_data")
```

### Example 2: Multi-File Dashboard
```python
# Combine multiple Excel files into one dashboard
files = ["customers.xlsx", "orders.xlsx", "products.xlsx"]

for file in files:
    processor.process_excel_file(f"./data/{file}")

# Joined Analysis
combined_data = connector.execute_query("""
    SELECT 
        c.customer_name,
        o.order_date,
        p.product_name,
        o.quantity * p.price as revenue
    FROM customers c
    JOIN orders o ON c.customer_id = o.customer_id
    JOIN products p ON o.product_id = p.product_id
""")
```

## 🎨 PyWalker Features

### Drag-and-Drop Exploration
- **Dimensions**: Drag categorical fields to rows/columns
- **Measures**: Drag numeric fields to values
- **Filters**: Interactive data filtering
- **Chart Types**: Automatic visualization recommendations

### Advanced Visualizations
- **Heatmaps**: Correlation analyses
- **Geographic Maps**: Location-based analyses
- **Time Series**: Trend analyses
- **Multi-dimensional Analysis**: Complex relationships

## 📊 Performance Optimizations

### DuckDB Optimizations
```sql
-- Indexed columns for better performance
CREATE INDEX idx_date ON sales_data(order_date);
CREATE INDEX idx_customer ON sales_data(customer_id);

-- Partitioned tables
CREATE TABLE sales_partitioned AS 
SELECT * FROM sales_data 
PARTITION BY RANGE(order_date);
```

### Parquet Optimizations
```python
# Compressed Parquet files
df.to_parquet(
    "data.parquet",
    compression='snappy',
    row_group_size=10000
)

# Column-wise compression
df.to_parquet(
    "data.parquet",
    compression={'name': 'gzip', 'amount': 'snappy'}
)
```

## 🐳 Docker Services

### Complete Environment
```bash
# Start all services
docker-compose up -d

# Only Data Analyzer
docker-compose up -d dataanalyzer

# With additional services
docker-compose up -d dataanalyzer ollama mongodb jupyter metabase
```

### Service Overview
- **dataanalyzer**: Main application (Port 7860)
- **ollama**: Local LLM server (Port 11434)
- **mongodb**: Document database (Port 27017)
- **mariadb**: Relational database (Port 3306)
- **jupyter**: Notebook environment (Port 8888)
- **metabase**: BI dashboard (Port 3000)

## Database Setup

### DuckDB (Primary)
```python
# Automatic setup through ExcelProcessor
databases/excel_data.duckdb
```

### MongoDB
```python
mongodb://dataanalyzer:dataanalyzer_pwd@localhost:27017/my_test_db?authSource=test
```

### SQL Databases
```python
# MariaDB/MySQL
mariadb+mariadbconnector://dataanalyzer:dataanalyzer_pwd@localhost:3306/my_test_db

# PostgreSQL
postgresql://username:password@localhost:5432/database_name

# SQLite
sqlite:///path/to/database.db
```

## 🔍 Troubleshooting

### Common Issues

**Excel files cannot be read:**
```python
# Try different engines
df = pd.read_excel(file_path, engine='openpyxl')  # For .xlsx
df = pd.read_excel(file_path, engine='xlrd')      # For .xls
```

**DuckDB connection errors:**
```python
# Check connection
try:
    conn = duckdb.connect("test.duckdb")
    conn.execute("SELECT 1")
    print("✅ DuckDB available")
except Exception as e:
    print(f"❌ DuckDB error: {e}")
```

**PyWalker issues:**
```bash
# Reinstall PyWalker
pip install --upgrade pygwalker
```

**Docker issues:**
```bash
# Reset Docker environment
docker-compose down
docker-compose up -d --build
```

## Error Handling

- **Comprehensive logging system** with detailed error tracking
- **Automatic retry mechanism** for database connections
- **LLM-based SQL query validation** and fixing
- **Exception handling** for visualization generation
- **Data validation** for Excel processing
- **Performance monitoring** for large datasets

## Security Notes

- **Secure your database credentials**
- **Use environment variables** for sensitive data
- **Configure proper authentication** sources
- **Set appropriate database user permissions**
- **Validate input data** from Excel files
- **Monitor resource usage** for large datasets

## License

GNU AFFERO GENERAL PUBLIC LICENSE Version 3

---

*Powered by DuckDB 🦆, PyWalker 🚶, Gradio 🎨, and Docker 🐳*
