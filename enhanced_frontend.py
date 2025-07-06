import os
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.io as pio
import logging
import json
from datetime import datetime
from typing import Dict, List

# Configure logging
logger = logging.getLogger(__name__)

# Try to import PyWalker for Tableau-like interface
try:
    import pygwalker as pyg

    PYGWALKER_AVAILABLE = True
except ImportError:
    PYGWALKER_AVAILABLE = False
    logger.warning("PyWalker not available - advanced exploration features disabled")

# Try to import Dash components (for future use)
try:
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    logger.warning("Dash not available - dashboard features limited")

# Import custom tools
try:
    from tools.excel_processor import ExcelProcessor
    from tools.duckdb_connector import DuckDBConnector
except ImportError as e:
    logger.error(f"Error importing custom tools: {e}")

# Global state
current_data = None
current_analyzer = None
current_config = None
excel_processor = None
duckdb_connector = None

# Configuration files
CONFIG_FILE = "database_configs.json"
EXCEL_CONFIG_FILE = "excel_configs.json"


def load_excel_configs():
    """Load Excel processing configurations"""
    if os.path.exists(EXCEL_CONFIG_FILE):
        with open(EXCEL_CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}


def save_excel_config(config_name: str, config_data: Dict):
    """Save Excel processing configuration"""
    configs = load_excel_configs()
    configs[config_name] = config_data

    with open(EXCEL_CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=2)

    return f"✅ Excel configuration '{config_name}' saved successfully"


def setup_excel_processor(source_dir: str, duck_lake_path: str = "ducklake") -> tuple:
    """Setup Excel processor and DuckDB connector"""
    global excel_processor, duckdb_connector

    try:
        # Validate source directory
        if not os.path.exists(source_dir):
            return False, f"❌ Source directory does not exist: {source_dir}"

        # Initialize Excel processor
        excel_processor = ExcelProcessor(
            source_dir=source_dir,
            duck_lake_path=duck_lake_path,
            duckdb_path="databases/excel_data.duckdb",
            use_parquet=True,
            auto_clean=True,
        )

        # Initialize DuckDB connector
        duckdb_connector = DuckDBConnector(
            duckdb_path="databases/excel_data.duckdb",
            duck_lake_path=duck_lake_path,
            enable_extensions=True,
        )

        return True, f"✅ Excel processor initialized for directory: {source_dir}"

    except Exception as e:
        logger.error(f"Error setting up Excel processor: {e}")
        return False, f"❌ Error: {str(e)}"


def process_excel_files(recursive: bool = True) -> tuple:
    """Process Excel files in the configured directory"""
    global excel_processor

    if not excel_processor:
        return None, "❌ Excel processor not initialized. Please setup first."

    try:
        results = excel_processor.process_directory(recursive=recursive)

        summary = f"""
        📊 Processing Complete:
        - ✅ Successfully processed: {results["processed"]} files
        - ❌ Errors: {results["errors"]} files
        - ⏭️ Skipped (unchanged): {results["skipped"]} files
        """

        # Create summary DataFrame
        if results["details"]:
            summary_df = pd.DataFrame(results["details"])
            return summary_df, summary
        else:
            return None, summary

    except Exception as e:
        logger.error(f"Error processing Excel files: {e}")
        return None, f"❌ Error processing files: {str(e)}"


def get_available_tables() -> List[str]:
    """Get list of available tables from DuckDB"""
    global duckdb_connector

    if not duckdb_connector:
        return []

    try:
        tables = duckdb_connector.list_tables()
        return [table["table_name"] for table in tables]
    except Exception as e:
        logger.error(f"Error getting tables: {e}")
        return []


def query_table(table_name: str, custom_query: str = None) -> tuple:
    """Query a table and return results"""
    global duckdb_connector, current_data

    if not duckdb_connector:
        return None, "❌ DuckDB connector not initialized"

    try:
        if custom_query and custom_query.strip():
            # Use custom query
            df = duckdb_connector.execute_query(custom_query)
        else:
            # Default query
            df = duckdb_connector.execute_query(
                f"SELECT * FROM {table_name} LIMIT 1000"
            )

        current_data = df

        if df.empty:
            return None, "ℹ️ Query returned no results"

        return df, f"✅ Retrieved {len(df)} rows from {table_name}"

    except Exception as e:
        logger.error(f"Error querying table: {e}")
        return None, f"❌ Query error: {str(e)}"


def analyze_table(table_name: str) -> tuple:
    """Analyze a table and return insights"""
    global duckdb_connector

    if not duckdb_connector:
        return None, "❌ DuckDB connector not initialized"

    try:
        analysis = duckdb_connector.analyze_table(table_name)

        # Create summary text
        summary = f"""
        📊 Analysis for {table_name}:
        
        📈 Basic Stats:
        - Rows: {analysis["row_count"]:,}
        - Columns: {analysis["column_count"]}
        - Numeric columns: {analysis["numeric_columns"]}
        - Categorical columns: {analysis["categorical_columns"]}
        
        🔍 Data Quality:
        - Missing values: {sum(analysis["missing_values"].values())}
        - Completeness: {((analysis["row_count"] * analysis["column_count"] - sum(analysis["missing_values"].values())) / (analysis["row_count"] * analysis["column_count"]) * 100):.1f}%
        """

        # Create analysis DataFrame
        analysis_data = []

        # Add numeric column stats
        for col, stats in analysis["numeric_summary"].items():
            if isinstance(stats, dict):
                analysis_data.append(
                    {
                        "Column": col,
                        "Type": "Numeric",
                        "Mean": stats.get("mean", 0),
                        "Std": stats.get("std", 0),
                        "Min": stats.get("min", 0),
                        "Max": stats.get("max", 0),
                        "Missing": analysis["missing_values"].get(col, 0),
                    }
                )

        # Add categorical column stats
        for col, values in analysis["categorical_summary"].items():
            analysis_data.append(
                {
                    "Column": col,
                    "Type": "Categorical",
                    "Unique Values": len(values),
                    "Most Common": max(values, key=values.get) if values else "N/A",
                    "Missing": analysis["missing_values"].get(col, 0),
                }
            )

        analysis_df = pd.DataFrame(analysis_data)

        return analysis_df, summary

    except Exception as e:
        logger.error(f"Error analyzing table: {e}")
        return None, f"❌ Analysis error: {str(e)}"


def create_pygwalker_interface(df: pd.DataFrame) -> str:
    """Create PyWalker interface for Tableau-like exploration"""
    if not PYGWALKER_AVAILABLE or df is None or df.empty:
        return "<p>PyWalker not available or no data to display</p>"

    try:
        # Create PyWalker HTML
        html_content = pyg.to_html(df, spec="./pygwalker_config.json")
        return html_content

    except Exception as e:
        logger.error(f"Error creating PyWalker interface: {e}")
        return f"<p>Error creating exploration interface: {str(e)}</p>"


def generate_visualization(vis_type: str, table_name: str = None) -> tuple:
    """Generate visualization using DuckDB data"""
    global duckdb_connector, current_data

    if not duckdb_connector and not current_data:
        return None, "❌ No data available for visualization"

    try:
        # Use current data or query table
        if current_data is not None:
            df = current_data
        elif table_name:
            df = duckdb_connector.execute_query(
                f"SELECT * FROM {table_name} LIMIT 1000"
            )
        else:
            return None, "❌ No data available"

        if df.empty:
            return None, "❌ No data to visualize"

        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        fig = None

        if vis_type == "auto":
            if len(categorical_cols) >= 1:
                value_counts = df[categorical_cols[0]].value_counts().head(10)
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"Distribution of {categorical_cols[0]}",
                )
            elif len(numeric_cols) >= 1:
                fig = px.histogram(
                    df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}"
                )

        elif vis_type == "bar" and len(categorical_cols) >= 1:
            value_counts = df[categorical_cols[0]].value_counts().head(10)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f"Bar Chart: {categorical_cols[0]}",
            )

        elif vis_type == "pie" and len(categorical_cols) >= 1:
            value_counts = df[categorical_cols[0]].value_counts().head(10)
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f"Pie Chart: {categorical_cols[0]}",
            )

        elif vis_type == "scatter" and len(numeric_cols) >= 2:
            fig = px.scatter(
                df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                title=f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}",
            )

        elif vis_type == "line" and len(numeric_cols) >= 1:
            # Create line chart with index
            fig = px.line(
                df.reset_index(),
                x="index",
                y=numeric_cols[0],
                title=f"Line Chart: {numeric_cols[0]}",
            )

        if fig:
            html_plot = pio.to_html(fig, include_plotlyjs="cdn", div_id="plotly-div")
            return html_plot, "✅ Visualization created successfully"
        else:
            return None, "❌ Could not create visualization with selected parameters"

    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return None, f"❌ Visualization error: {str(e)}"


def export_data(format_type: str, table_name: str = None) -> str:
    """Export data in various formats"""
    global current_data, duckdb_connector

    try:
        # Get data
        if current_data is not None:
            df = current_data
            filename_base = "exported_data"
        elif table_name and duckdb_connector:
            df = duckdb_connector.execute_query(f"SELECT * FROM {table_name}")
            filename_base = f"{table_name}_export"
        else:
            return None

        if df.empty:
            return None

        # Export based on format
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format_type == "csv":
            filename = f"{filename_base}_{timestamp}.csv"
            df.to_csv(filename, index=False)
        elif format_type == "excel":
            filename = f"{filename_base}_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
        elif format_type == "parquet":
            filename = f"{filename_base}_{timestamp}.parquet"
            df.to_parquet(filename, index=False)
        elif format_type == "json":
            filename = f"{filename_base}_{timestamp}.json"
            df.to_json(filename, orient="records", indent=2)
        else:
            return None

        return filename

    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        return None


def create_enhanced_interface():
    """Create enhanced Gradio interface with Excel processing and PyWalker integration"""

    with gr.Blocks(
        title="Advanced Data Analyzer - Excel to DuckDB/DuckLake",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown("# 📊 Advanced Data Analyzer")
        gr.Markdown(
            "*Excel zu DuckDB/DuckLake Transformation mit interaktiver Datenanalyse*"
        )

        with gr.Tabs():
            # Tab 1: Excel Processing
            with gr.Tab("📁 Excel Processing"):
                gr.Markdown("## Excel-Dateien verarbeiten")

                with gr.Row():
                    source_dir = gr.Textbox(
                        label="📂 Quellverzeichnis",
                        placeholder="Pfad zu Excel-Dateien...",
                        value="./excel_files",
                    )
                    duck_lake_path = gr.Textbox(
                        label="🏞️ DuckLake Pfad",
                        placeholder="Pfad für Parquet-Dateien...",
                        value="./ducklake",
                    )

                with gr.Row():
                    setup_btn = gr.Button("🔧 Setup Excel Processor", variant="primary")
                    process_btn = gr.Button(
                        "▶️ Process Excel Files", variant="secondary"
                    )
                    recursive_check = gr.Checkbox(
                        label="Rekursiv verarbeiten", value=True
                    )

                setup_status = gr.Textbox(label="Setup Status", interactive=False)

                with gr.Row():
                    process_results = gr.Dataframe(label="Processing Results")
                    process_status = gr.Textbox(
                        label="Processing Status", interactive=False
                    )

                # Event handlers
                setup_btn.click(
                    setup_excel_processor,
                    inputs=[source_dir, duck_lake_path],
                    outputs=[setup_status],
                )

                process_btn.click(
                    process_excel_files,
                    inputs=[recursive_check],
                    outputs=[process_results, process_status],
                )

            # Tab 2: Data Explorer
            with gr.Tab("🔍 Data Explorer"):
                gr.Markdown("## Daten erkunden und analysieren")

                with gr.Row():
                    table_dropdown = gr.Dropdown(
                        label="📋 Verfügbare Tabellen", choices=[], interactive=True
                    )
                    refresh_tables_btn = gr.Button("🔄 Refresh Tables")

                with gr.Row():
                    with gr.Column():
                        custom_query = gr.Textbox(
                            label="🔍 Custom SQL Query (optional)",
                            placeholder="SELECT * FROM table_name WHERE condition...",
                            lines=3,
                        )
                        query_btn = gr.Button("▶️ Execute Query", variant="primary")

                    with gr.Column():
                        analyze_btn = gr.Button("📊 Analyze Table", variant="secondary")
                        export_format = gr.Dropdown(
                            label="Export Format",
                            choices=["csv", "excel", "parquet", "json"],
                            value="csv",
                        )
                        export_btn = gr.Button("💾 Export Data")

                query_status = gr.Textbox(label="Query Status", interactive=False)

                with gr.Row():
                    query_results = gr.Dataframe(label="Query Results", max_rows=20)
                    analysis_results = gr.Dataframe(label="Analysis Results")

                analysis_status = gr.Textbox(label="Analysis Status", interactive=False)

                # Event handlers
                refresh_tables_btn.click(
                    lambda: gr.update(choices=get_available_tables()),
                    outputs=[table_dropdown],
                )

                query_btn.click(
                    query_table,
                    inputs=[table_dropdown, custom_query],
                    outputs=[query_results, query_status],
                )

                analyze_btn.click(
                    analyze_table,
                    inputs=[table_dropdown],
                    outputs=[analysis_results, analysis_status],
                )

                export_btn.click(
                    export_data,
                    inputs=[export_format, table_dropdown],
                    outputs=[gr.File(label="Downloaded File")],
                )

            # Tab 3: PyWalker Explorer (Tableau-like)
            with gr.Tab("📈 Advanced Explorer"):
                gr.Markdown("## PyWalker - Tableau-ähnliche Datenexploration")

                if PYGWALKER_AVAILABLE:
                    with gr.Row():
                        explorer_table = gr.Dropdown(
                            label="📋 Tabelle für Exploration",
                            choices=[],
                            interactive=True,
                        )
                        load_explorer_btn = gr.Button(
                            "📊 Load in Explorer", variant="primary"
                        )

                    pygwalker_html = gr.HTML(label="PyWalker Interface")

                    # Event handlers
                    load_explorer_btn.click(
                        lambda table_name: create_pygwalker_interface(
                            duckdb_connector.execute_query(
                                f"SELECT * FROM {table_name} LIMIT 10000"
                            )
                            if duckdb_connector and table_name
                            else None
                        ),
                        inputs=[explorer_table],
                        outputs=[pygwalker_html],
                    )

                    # Sync table choices
                    refresh_tables_btn.click(
                        lambda: gr.update(choices=get_available_tables()),
                        outputs=[explorer_table],
                    )
                else:
                    gr.Markdown(
                        "⚠️ PyWalker not available. Install with: `pip install pygwalker`"
                    )

            # Tab 4: Visualizations
            with gr.Tab("📊 Visualizations"):
                gr.Markdown("## Datenvisualisierung")

                with gr.Row():
                    vis_table = gr.Dropdown(
                        label="📋 Tabelle", choices=[], interactive=True
                    )
                    vis_type = gr.Dropdown(
                        label="📊 Visualisierungstyp",
                        choices=["auto", "bar", "pie", "scatter", "line"],
                        value="auto",
                    )
                    create_vis_btn = gr.Button(
                        "📊 Create Visualization", variant="primary"
                    )

                vis_status = gr.Textbox(label="Visualization Status", interactive=False)
                vis_output = gr.HTML(label="Visualization")

                # Event handlers
                create_vis_btn.click(
                    generate_visualization,
                    inputs=[vis_type, vis_table],
                    outputs=[vis_output, vis_status],
                )

                # Sync table choices
                refresh_tables_btn.click(
                    lambda: gr.update(choices=get_available_tables()),
                    outputs=[vis_table],
                )

        # Footer
        gr.Markdown("---")
        gr.Markdown("*Powered by DuckDB, PyWalker, and Gradio*")

    return interface


def main():
    """Main function to launch the enhanced interface"""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Create directories
    os.makedirs("databases", exist_ok=True)
    os.makedirs("ducklake", exist_ok=True)
    os.makedirs("excel_files", exist_ok=True)

    # Create and launch interface
    interface = create_enhanced_interface()

    # Launch with custom settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
