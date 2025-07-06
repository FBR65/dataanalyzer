import logging
import duckdb
import pandas as pd
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class DuckDBConnector:
    """
    Enhanced DuckDB connector with support for DuckLake (Parquet) files
    and advanced analytical capabilities.
    """

    def __init__(
        self,
        duckdb_path: str = ":memory:",
        duck_lake_path: str = "ducklake",
        enable_extensions: bool = True,
    ):
        """
        Initialize DuckDB connector

        Args:
            duckdb_path: Path to DuckDB database file (":memory:" for in-memory)
            duck_lake_path: Path to DuckLake storage directory
            enable_extensions: Whether to load DuckDB extensions
        """
        self.duckdb_path = duckdb_path
        self.duck_lake_path = Path(duck_lake_path)
        self.duck_lake_path.mkdir(parents=True, exist_ok=True)

        # Initialize connection
        self.conn = duckdb.connect(duckdb_path)

        # Load extensions if enabled
        if enable_extensions:
            self._load_extensions()

        # Register DuckLake path for Parquet access
        self._register_duck_lake()

        logger.info(f"DuckDB connector initialized with database: {duckdb_path}")

    def _load_extensions(self):
        """Load useful DuckDB extensions"""
        try:
            # Load common extensions
            extensions = [
                "parquet",  # Parquet support
                "json",  # JSON support
                "httpfs",  # HTTP file system support
                "spatial",  # Spatial data support
                "icu",  # International components
                "fts",  # Full-text search
            ]

            for ext in extensions:
                try:
                    self.conn.execute(f"INSTALL {ext}")
                    self.conn.execute(f"LOAD {ext}")
                    logger.debug(f"Loaded DuckDB extension: {ext}")
                except Exception as e:
                    logger.debug(f"Could not load extension {ext}: {e}")

        except Exception as e:
            logger.warning(f"Error loading DuckDB extensions: {e}")

    def _register_duck_lake(self):
        """Register DuckLake path for easy Parquet access"""
        try:
            # Create a view that shows all available Parquet files
            parquet_files = list(self.duck_lake_path.glob("*.parquet"))

            if parquet_files:
                # Create a catalog view for all Parquet files
                self.conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS duck_lake_catalog AS
                    SELECT 
                        filename,
                        file_size,
                        file_last_modified
                    FROM glob('"""
                    + str(self.duck_lake_path / "*.parquet")
                    + """')
                """
                )

                logger.info(
                    f"Registered DuckLake with {len(parquet_files)} Parquet files"
                )

        except Exception as e:
            logger.warning(f"Error registering DuckLake: {e}")

    def execute_query(self, query: str, params: List = None) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame

        Args:
            query: SQL query string
            params: Optional parameters for parameterized queries

        Returns:
            Pandas DataFrame with results
        """
        try:
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)

            return result.df()

        except Exception as e:
            logger.error(f"Error executing query: {e}")
            logger.error(f"Query: {query}")
            raise

    def load_parquet_file(self, file_path: str, table_name: str = None) -> str:
        """
        Load a Parquet file into DuckDB

        Args:
            file_path: Path to the Parquet file
            table_name: Optional table name (if None, derives from filename)

        Returns:
            Name of the created table
        """
        if table_name is None:
            table_name = Path(file_path).stem.lower()

        try:
            # Load Parquet file directly
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE {table_name} AS 
                SELECT * FROM read_parquet('{file_path}')
            """)

            logger.info(f"Loaded Parquet file {file_path} as table {table_name}")
            return table_name

        except Exception as e:
            logger.error(f"Error loading Parquet file {file_path}: {e}")
            raise

    def load_all_parquet_files(self) -> List[str]:
        """
        Load all Parquet files from DuckLake into DuckDB

        Returns:
            List of created table names
        """
        parquet_files = list(self.duck_lake_path.glob("*.parquet"))
        table_names = []

        for file_path in parquet_files:
            try:
                table_name = self.load_parquet_file(str(file_path))
                table_names.append(table_name)
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(table_names)} Parquet files into DuckDB")
        return table_names

    def list_tables(self) -> List[Dict[str, Any]]:
        """
        List all tables in the database

        Returns:
            List of table information dictionaries
        """
        try:
            tables = self.conn.execute("""
                SELECT 
                    table_name,
                    table_type,
                    estimated_size
                FROM information_schema.tables
                WHERE table_schema = 'main'
                ORDER BY table_name
            """).fetchall()

            return [
                {"table_name": row[0], "table_type": row[1], "estimated_size": row[2]}
                for row in tables
            ]

        except Exception as e:
            logger.error(f"Error listing tables: {e}")
            return []

    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a table

        Args:
            table_name: Name of the table

        Returns:
            Dictionary with table information
        """
        try:
            # Get basic table info
            info = self.conn.execute(f"""
                SELECT COUNT(*) as row_count
                FROM {table_name}
            """).fetchone()

            # Get column info
            columns = self.conn.execute(f"""
                SELECT 
                    column_name,
                    data_type,
                    is_nullable
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
            """).fetchall()

            # Get sample data
            sample = self.conn.execute(f"""
                SELECT * FROM {table_name} LIMIT 5
            """).df()

            return {
                "table_name": table_name,
                "row_count": info[0] if info else 0,
                "columns": [
                    {"name": col[0], "type": col[1], "nullable": col[2]}
                    for col in columns
                ],
                "sample_data": sample.to_dict("records"),
            }

        except Exception as e:
            logger.error(f"Error getting table info for {table_name}: {e}")
            raise

    def analyze_table(self, table_name: str) -> Dict[str, Any]:
        """
        Perform statistical analysis on a table

        Args:
            table_name: Name of the table to analyze

        Returns:
            Dictionary with analysis results
        """
        try:
            # Get data for analysis
            df = self.execute_query(f"SELECT * FROM {table_name}")

            # Perform analysis
            numeric_cols = df.select_dtypes(include=["number"]).columns
            categorical_cols = df.select_dtypes(include=["object"]).columns

            analysis = {
                "table_name": table_name,
                "row_count": len(df),
                "column_count": len(df.columns),
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "missing_values": df.isnull().sum().to_dict(),
                "numeric_summary": df[numeric_cols].describe().to_dict()
                if len(numeric_cols) > 0
                else {},
                "categorical_summary": {
                    col: df[col].value_counts().head(10).to_dict()
                    for col in categorical_cols
                }
                if len(categorical_cols) > 0
                else {},
            }

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {e}")
            raise

    def create_aggregated_view(
        self,
        table_name: str,
        group_by: List[str],
        aggregations: Dict[str, str],
        view_name: str = None,
    ) -> str:
        """
        Create an aggregated view from a table

        Args:
            table_name: Source table name
            group_by: List of columns to group by
            aggregations: Dict of column -> aggregation function
            view_name: Optional name for the view

        Returns:
            Name of the created view
        """
        if view_name is None:
            view_name = f"{table_name}_aggregated"

        try:
            # Build aggregation clauses
            agg_clauses = []
            for col, func in aggregations.items():
                agg_clauses.append(f"{func}({col}) as {col}_{func}")

            group_by_clause = ", ".join(group_by)
            agg_clause = ", ".join(agg_clauses)

            query = f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT 
                    {group_by_clause},
                    {agg_clause}
                FROM {table_name}
                GROUP BY {group_by_clause}
            """

            self.conn.execute(query)
            logger.info(f"Created aggregated view {view_name}")
            return view_name

        except Exception as e:
            logger.error(f"Error creating aggregated view: {e}")
            raise

    def generate_visualization_data(
        self,
        table_name: str,
        chart_type: str = "auto",
        x_column: str = None,
        y_column: str = None,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Generate data suitable for visualization

        Args:
            table_name: Name of the table
            chart_type: Type of chart (bar, line, pie, scatter, auto)
            x_column: X-axis column name
            y_column: Y-axis column name
            limit: Maximum number of rows to return

        Returns:
            Dictionary with visualization data and metadata
        """
        try:
            # Get data
            df = self.execute_query(f"SELECT * FROM {table_name} LIMIT {limit}")

            if df.empty:
                return {"error": "Table is empty"}

            # Determine columns if not specified
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

            if x_column is None and categorical_cols:
                x_column = categorical_cols[0]
            elif x_column is None and numeric_cols:
                x_column = numeric_cols[0]

            if y_column is None and numeric_cols:
                y_column = numeric_cols[0]

            # Auto-detect chart type
            if chart_type == "auto":
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    chart_type = "bar"
                elif len(numeric_cols) >= 2:
                    chart_type = "scatter"
                elif len(categorical_cols) > 0:
                    chart_type = "pie"
                else:
                    chart_type = "line"

            # Prepare data based on chart type
            if chart_type == "bar" and x_column and y_column:
                # Group by x_column and aggregate y_column
                agg_data = df.groupby(x_column)[y_column].mean().reset_index()
                data = {
                    "x": agg_data[x_column].tolist(),
                    "y": agg_data[y_column].tolist(),
                    "type": "bar",
                }
            elif chart_type == "pie" and x_column:
                # Count values for pie chart
                value_counts = df[x_column].value_counts().head(10)
                data = {
                    "labels": value_counts.index.tolist(),
                    "values": value_counts.values.tolist(),
                    "type": "pie",
                }
            elif chart_type == "scatter" and x_column and y_column:
                data = {
                    "x": df[x_column].tolist(),
                    "y": df[y_column].tolist(),
                    "type": "scatter",
                }
            elif chart_type == "line" and x_column and y_column:
                data = {
                    "x": df[x_column].tolist(),
                    "y": df[y_column].tolist(),
                    "type": "line",
                }
            else:
                # Fallback to basic bar chart
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    value_counts = df[col].value_counts().head(10)
                    data = {
                        "x": value_counts.index.tolist(),
                        "y": value_counts.values.tolist(),
                        "type": "bar",
                    }
                else:
                    return {"error": "Cannot determine appropriate visualization"}

            return {
                "data": data,
                "metadata": {
                    "table_name": table_name,
                    "chart_type": chart_type,
                    "x_column": x_column,
                    "y_column": y_column,
                    "row_count": len(df),
                    "available_columns": {
                        "numeric": numeric_cols,
                        "categorical": categorical_cols,
                    },
                },
            }

        except Exception as e:
            logger.error(f"Error generating visualization data: {e}")
            return {"error": str(e)}

    async def generate_insights(self, table_name: str) -> Dict[str, Any]:
        """
        Generate automated insights about a table

        Args:
            table_name: Name of the table to analyze

        Returns:
            Dictionary with insights
        """
        try:
            analysis = self.analyze_table(table_name)

            insights = {
                "table_name": table_name,
                "summary": {
                    "total_rows": analysis["row_count"],
                    "total_columns": analysis["column_count"],
                    "data_quality": {
                        "missing_values_count": sum(
                            analysis["missing_values"].values()
                        ),
                        "completeness_score": 1
                        - (
                            sum(analysis["missing_values"].values())
                            / (analysis["row_count"] * analysis["column_count"])
                        ),
                    },
                },
                "column_insights": [],
                "recommendations": [],
            }

            # Analyze numeric columns
            for col, stats in analysis["numeric_summary"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    insights["column_insights"].append(
                        {
                            "column": col,
                            "type": "numeric",
                            "mean": stats["mean"],
                            "std": stats["std"],
                            "min": stats["min"],
                            "max": stats["max"],
                            "outliers_potential": stats["std"] > stats["mean"]
                            if stats["mean"] > 0
                            else False,
                        }
                    )

            # Analyze categorical columns
            for col, values in analysis["categorical_summary"].items():
                unique_count = len(values)
                most_common = max(values, key=values.get) if values else None

                insights["column_insights"].append(
                    {
                        "column": col,
                        "type": "categorical",
                        "unique_values": unique_count,
                        "most_common": most_common,
                        "diversity_score": unique_count / analysis["row_count"]
                        if analysis["row_count"] > 0
                        else 0,
                    }
                )

            # Generate recommendations
            if insights["summary"]["data_quality"]["completeness_score"] < 0.9:
                insights["recommendations"].append(
                    "Consider data cleaning - high missing values detected"
                )

            if analysis["row_count"] > 100000:
                insights["recommendations"].append(
                    "Large dataset - consider creating indexed views for better performance"
                )

            return insights

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {"error": str(e)}

    def export_to_parquet(self, table_name: str, file_path: str = None) -> str:
        """
        Export table to Parquet format

        Args:
            table_name: Name of the table to export
            file_path: Optional file path (if None, uses duck_lake_path)

        Returns:
            Path to the exported file
        """
        if file_path is None:
            file_path = (
                self.duck_lake_path
                / f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )

        try:
            self.conn.execute(f"""
                COPY {table_name} TO '{file_path}' (FORMAT PARQUET)
            """)

            logger.info(f"Exported table {table_name} to {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Error exporting table {table_name}: {e}")
            raise

    def backup_database(self, backup_path: str = None) -> str:
        """
        Create a backup of the database

        Args:
            backup_path: Optional backup file path

        Returns:
            Path to the backup file
        """
        if backup_path is None:
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.duckdb"

        try:
            self.conn.execute(f"EXPORT DATABASE '{backup_path}'")
            logger.info(f"Database backed up to {backup_path}")
            return backup_path

        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            raise

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_duckdb_connector(
    duckdb_path: str = "databases/main.duckdb",
    duck_lake_path: str = "ducklake",
    enable_extensions: bool = True,
) -> DuckDBConnector:
    """Factory function to create a DuckDB connector"""
    return DuckDBConnector(
        duckdb_path=duckdb_path,
        duck_lake_path=duck_lake_path,
        enable_extensions=enable_extensions,
    )
