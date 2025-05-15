# Data Analyzer

A powerful Python application for analyzing data from both SQL and MongoDB databases with natural language queries and visualization capabilities.

## Features

- **Multi-Database Support**
  - SQL databases (PostgreSQL, MySQL, MariaDB, Oracle, SQLite)
  - MongoDB integration
  - Automatic database schema detection

- **Natural Language Processing**
  - Convert natural language queries to SQL/MongoDB queries
  - Support for both German and English queries
  - Uses LLM models for query generation

- **Data Visualization**
  - Automatic plot generation with matplotlib
  - Dynamic visualization based on data structure
  - Base64 encoded image output

- **Interactive Interface**
  - Gradio web interface
  - Real-time connection testing
  - Visual query results

## Requirements

### Core Dependencies
```bash
uv sync
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

Environment variables:
```bash
OPENAI_API_KEY=ollama                           # Default for local LLM
OPENAI_API_BASE_URL=http://localhost:11434/v1   # Ollama API endpoint
LLM_MODEL=qwen2.5:latest                        # Base LLM model
CODING_MODEL=qwen2.5-coder:latest               # Code-specific LLM model
```

## Project Structure

```
dataanalyzer/
├── main.py              # Core application logic
├── gradio_frontend.py   # Web interface
├── tools/
│   ├── database.py      # SQL database connector
│   ├── mongodb.py       # MongoDB connector
│   └── pythontools.py   # Python REPL and visualization
├── generated_code/      # Auto-generated visualization code
└── README.md
```

## Usage

1. Start the web interface:
```bash
python gradio_frontend.py
```

2. Connect to your database:
   - Fill in connection details
   - Test connection
   - Choose database type (SQL/MongoDB)

3. Enter natural language query:
```
Example queries:
- "Show me the average age of users by city"
- "Zeige mir die Durchschnittsalter der Benutzer nach Stadt"
```

## Database Setup

### MongoDB
```python
mongodb://dataanalyzer:dataanalyzer_pwd@localhost:27017/my_test_db?authSource=test
```

### SQL
```python
mariadb+mariadbconnector://dataanalyzer:dataanalyzer_pwd@localhost:3306/my_test_db
```

## Error Handling

- Comprehensive logging system
- Automatic retry mechanism for database connections
- LLM-based SQL query validation and fixing
- Exception handling for visualization generation

## Security Notes

- Secure your database credentials
- Use environment variables for sensitive data
- Configure proper authentication sources
- Set appropriate database user permissions

## License

GNU AFFERO GENERAL PUBLIC LICENSE Version 3
