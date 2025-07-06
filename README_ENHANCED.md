# 🚀 Enhanced Data Analyzer - Excel zu DuckDB/DuckLake Transformation

Eine erweiterte Version des Data Analyzers mit automatisierter Excel-zu-DuckDB/DuckLake-Transformation und PyWalker-Integration für Tableau-ähnliche Datenexploration.

## ✨ Neue Features

### 📊 Excel-zu-DuckDB/DuckLake Pipeline
- **Automatisierte Excel-Verarbeitung**: Bulk-Import von Excel-Dateien
- **DuckDB Integration**: Hochperformante spaltenorientierte Datenbank
- **DuckLake Support**: Parquet-basierter Data Lake mit DuckDB-Integration
- **Intelligente Datenbereinigung**: Automatische Spaltennamensbereinigung und Datentyperkennung
- **Metadaten-Tracking**: Vollständige Nachverfolgung verarbeiteter Dateien

### 🎯 PyWalker Integration
- **Tableau-ähnliche Exploration**: Drag-and-Drop Datenexploration
- **Interaktive Visualisierungen**: Erweiterte Chart-Typen und Konfigurationsmöglichkeiten
- **Explorative Datenanalyse**: Intuitive Benutzeroberfläche für Geschäftsanwender

### 🔧 Erweiterte Analysefunktionen
- **Automatische Insights**: KI-gestützte Datenanalyse und Empfehlungen
- **Erweiterte Visualisierungen**: Plotly-basierte interaktive Charts
- **Flexible Datenexporte**: Multiple Formate (CSV, Excel, Parquet, JSON)
- **Performance-optimierte Abfragen**: DuckDB-optimierte SQL-Queries

## 🏗️ Architektur

```
Enhanced Data Analyzer/
├── 📁 Excel Processing Layer
│   ├── ExcelProcessor (tools/excel_processor.py)
│   ├── Automatische Datenerkennung
│   └── Batch-Verarbeitung
├── 🗄️ DuckDB/DuckLake Layer
│   ├── DuckDBConnector (tools/duckdb_connector.py)
│   ├── Parquet-basierte Speicherung
│   └── Erweiterte SQL-Funktionen
├── 🎨 Frontend Layer
│   ├── Enhanced Gradio Interface
│   ├── PyWalker Integration
│   └── Erweiterte Visualisierungen
└── 🐳 Containerization
    ├── Docker + Docker Compose
    ├── Ollama LLM Support
    └── Optionale Services (MongoDB, MariaDB, Jupyter, Metabase)
```

## 🚀 Schnellstart

### Option 1: Docker (Empfohlen)
```bash
# Repository klonen
git clone <repository-url>
cd dataanalyzer

# Umgebungsvariablen setzen (optional)
cp .env.example .env

# Services starten
docker-compose up -d

# Warten bis alle Services bereit sind
docker-compose logs -f dataanalyzer
```

**Zugriff auf Services:**
- **Data Analyzer**: http://localhost:7860
- **Jupyter Notebook**: http://localhost:8888 (Token: dataanalyzer123)
- **Metabase**: http://localhost:3000
- **Ollama**: http://localhost:11434

### Option 2: Lokale Installation
```bash
# Abhängigkeiten installieren
uv sync

# Verzeichnisse erstellen
mkdir -p excel_files databases ducklake generated_code logs

# Enhanced Frontend starten
uv run python enhanced_frontend.py
```

## 💡 Verwendung

### 1. Excel-Dateien verarbeiten
```python
from tools.excel_processor import ExcelProcessor

# Processor initialisieren
processor = ExcelProcessor(
    source_dir="./excel_files",
    duck_lake_path="./ducklake",
    duckdb_path="./databases/excel_data.duckdb"
)

# Alle Excel-Dateien verarbeiten
results = processor.process_directory(recursive=True)
print(f"Verarbeitet: {results['processed']} Dateien")
```

### 2. Daten mit DuckDB abfragen
```python
from tools.duckdb_connector import DuckDBConnector

# Connector initialisieren
connector = DuckDBConnector(
    duckdb_path="./databases/excel_data.duckdb",
    duck_lake_path="./ducklake"
)

# Verfügbare Tabellen anzeigen
tables = connector.list_tables()
print(f"Verfügbare Tabellen: {[t['table_name'] for t in tables]}")

# Daten abfragen
df = connector.execute_query("SELECT * FROM my_table LIMIT 100")
print(df.head())
```

### 3. PyWalker für Exploration verwenden
```python
import pygwalker as pyg

# PyWalker Interface erstellen
pyg_html = pyg.to_html(df)
# HTML wird in Gradio Interface eingebettet
```

## 🎯 Workflow-Beispiele

### Beispiel 1: Automatisierte Verkaufsdaten-Analyse
```python
# 1. Excel-Verkaufsdaten verarbeiten
processor = ExcelProcessor("./sales_data")
results = processor.process_directory()

# 2. Daten analysieren
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

# 3. Insights generieren
insights = connector.generate_insights("sales_data")
```

### Beispiel 2: Multi-File Dashboard
```python
# Mehrere Excel-Dateien zu einem Dashboard kombinieren
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

## 🔧 Konfiguration

### Umgebungsvariablen
```bash
# LLM-Konfiguration
OPENAI_API_KEY=ollama
OPENAI_API_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:latest
CODING_MODEL=qwen2.5-coder:latest

# Datenbank-Konfiguration
DUCKDB_PATH=./databases/main.duckdb
DUCK_LAKE_PATH=./ducklake

# Interface-Konfiguration
GRADIO_SERVER_NAME=0.0.0.0
GRADIO_SERVER_PORT=7860
```

### Excel-Verarbeitung konfigurieren
```python
processor = ExcelProcessor(
    source_dir="./excel_files",
    duck_lake_path="./ducklake",
    use_parquet=True,        # Parquet für bessere Performance
    auto_clean=True,         # Automatische Datenbereinigung
    max_file_size_mb=100     # Maximale Dateigröße
)
```

## 🎨 PyWalker Features

### Drag-and-Drop Exploration
- **Dimensionen**: Ziehen Sie kategorische Felder in Zeilen/Spalten
- **Kennzahlen**: Ziehen Sie numerische Felder in Werte
- **Filter**: Interaktive Datenfilterung
- **Chart-Typen**: Automatische Visualisierungsempfehlungen

### Erweiterte Visualisierungen
- **Heatmaps**: Korrelationsanalysen
- **Geographische Karten**: Standortbasierte Analysen
- **Zeitreihen**: Trend-Analysen
- **Multi-dimensionale Analyse**: Komplexe Zusammenhänge

## 📊 Performance-Optimierungen

### DuckDB-Optimierungen
```sql
-- Indizierte Spalten für bessere Performance
CREATE INDEX idx_date ON sales_data(order_date);
CREATE INDEX idx_customer ON sales_data(customer_id);

-- Partitionierte Tabellen
CREATE TABLE sales_partitioned AS 
SELECT * FROM sales_data 
PARTITION BY RANGE(order_date);
```

### Parquet-Optimierungen
```python
# Komprimierte Parquet-Dateien
df.to_parquet(
    "data.parquet",
    compression='snappy',
    row_group_size=10000
)

# Spaltenweise Komprimierung
df.to_parquet(
    "data.parquet",
    compression={'name': 'gzip', 'amount': 'snappy'}
)
```

## 🐳 Docker Services

### Vollständige Umgebung
```bash
# Alle Services starten
docker-compose up -d

# Nur Data Analyzer
docker-compose up -d dataanalyzer

# Mit zusätzlichen Services
docker-compose up -d dataanalyzer ollama mongodb jupyter metabase
```

### Service-Übersicht
- **dataanalyzer**: Hauptanwendung (Port 7860)
- **ollama**: Lokaler LLM-Server (Port 11434)
- **mongodb**: Dokumentendatenbank (Port 27017)
- **mariadb**: Relationale Datenbank (Port 3306)
- **jupyter**: Notebook-Umgebung (Port 8888)
- **metabase**: BI-Dashboard (Port 3000)

## 🔍 Troubleshooting

### Häufige Probleme

**Excel-Dateien können nicht gelesen werden:**
```python
# Verschiedene Engines probieren
df = pd.read_excel(file_path, engine='openpyxl')  # Für .xlsx
df = pd.read_excel(file_path, engine='xlrd')      # Für .xls
```

**DuckDB-Verbindungsfehler:**
```python
# Verbindung prüfen
try:
    conn = duckdb.connect("test.duckdb")
    conn.execute("SELECT 1")
    print("✅ DuckDB verfügbar")
except Exception as e:
    print(f"❌ DuckDB-Fehler: {e}")
```

**PyWalker-Probleme:**
```bash
# PyWalker neu installieren
pip install --upgrade pygwalker
```

## 🚀 Roadmap

### Geplante Features
- [ ] **Automatisierte Scheduler**: Cron-basierte Excel-Verarbeitung
- [ ] **Erweiterte BI-Integration**: Tableau/Power BI Connectors
- [ ] **Machine Learning Pipeline**: Automated ML auf DuckDB-Daten
- [ ] **Real-time Processing**: Streaming Excel-Updates
- [ ] **Cloud-Integration**: AWS S3/Azure Blob Storage
- [ ] **REST API**: Programmatischer Zugriff auf alle Funktionen

### Verbesserungen
- [ ] **Performance**: Parallel Excel-Verarbeitung
- [ ] **Skalierbarkeit**: Distributed DuckDB
- [ ] **Monitoring**: Detaillierte Metriken und Alerts
- [ ] **Security**: Verschlüsselung und Zugriffskontrolle

## 📄 Lizenz

GNU AFFERO GENERAL PUBLIC LICENSE Version 3

## 🤝 Beitragen

Beiträge sind willkommen! Bitte lesen Sie die CONTRIBUTING.md für Details.

## 📞 Support

Bei Fragen oder Problemen öffnen Sie bitte ein Issue auf GitHub.

---

*Powered by DuckDB 🦆, PyWalker 🚶, Gradio 🎨, und Docker 🐳*
