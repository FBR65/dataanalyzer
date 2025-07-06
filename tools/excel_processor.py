import pandas as pd
import duckdb
import logging
from typing import Dict, List
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


class ExcelProcessor:
    """
    Handles Excel file processing, transformation, and storage in DuckDB/DuckLake format.
    """

    def __init__(
        self,
        source_dir: str,
        duck_lake_path: str = "ducklake",
        duckdb_path: str = "databases/main.duckdb",
        use_parquet: bool = True,
        auto_clean: bool = True,
    ):
        """
        Initialize Excel processor

        Args:
            source_dir: Directory containing Excel files to process
            duck_lake_path: Path for DuckLake storage (Parquet files)
            duckdb_path: Path for DuckDB database file
            use_parquet: Whether to use Parquet format for DuckLake
            auto_clean: Whether to automatically clean data during processing
        """
        self.source_dir = Path(source_dir)
        self.duck_lake_path = Path(duck_lake_path)
        self.duckdb_path = Path(duckdb_path)
        self.use_parquet = use_parquet
        self.auto_clean = auto_clean

        # Create necessary directories
        self.duck_lake_path.mkdir(parents=True, exist_ok=True)
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize DuckDB connection
        self.conn = duckdb.connect(str(self.duckdb_path))

        # Create metadata table for tracking processed files
        self._create_metadata_table()

        logger.info(f"Excel processor initialized with source: {source_dir}")

    def _create_metadata_table(self):
        """Create metadata table for tracking processed files"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS file_metadata (
                file_path TEXT PRIMARY KEY,
                file_name TEXT,
                file_hash TEXT,
                table_name TEXT,
                parquet_path TEXT,
                processed_at TIMESTAMP,
                row_count INTEGER,
                column_count INTEGER,
                file_size_bytes INTEGER,
                status TEXT
            )
        """)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file for change detection"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare DataFrame for storage"""
        if not self.auto_clean:
            return df

        # Clean column names
        df.columns = df.columns.str.strip()
        df.columns = df.columns.str.replace(r"[^\w\s]", "_", regex=True)
        df.columns = df.columns.str.replace(r"\s+", "_", regex=True)

        # Remove completely empty rows and columns
        df = df.dropna(how="all", axis=0)  # Remove empty rows
        df = df.dropna(how="all", axis=1)  # Remove empty columns

        # Handle missing values intelligently
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna("Unknown")
            elif df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(0)

        return df

    def _generate_table_name(self, file_name: str) -> str:
        """Generate a valid table name from file name"""
        # Remove extension and clean name
        name = Path(file_name).stem
        name = name.lower()
        name = name.replace(" ", "_")
        name = "".join(c for c in name if c.isalnum() or c == "_")

        # Ensure it starts with a letter
        if not name[0].isalpha():
            name = f"table_{name}"

        return name

    def process_excel_file(self, file_path: Path) -> Dict:
        """
        Process a single Excel file and store in DuckDB/DuckLake

        Returns:
            Dictionary with processing results
        """
        try:
            file_hash = self._calculate_file_hash(file_path)
            file_name = file_path.name

            # Check if file already processed and unchanged
            existing = self.conn.execute(
                "SELECT file_hash, status FROM file_metadata WHERE file_path = ?",
                [str(file_path)],
            ).fetchone()

            if existing and existing[0] == file_hash and existing[1] == "success":
                logger.info(f"File {file_name} already processed and unchanged")
                return {"status": "skipped", "reason": "unchanged"}

            # Read Excel file
            logger.info(f"Processing Excel file: {file_name}")

            # Try to read with different engines for compatibility
            try:
                df = pd.read_excel(file_path, engine="openpyxl")
            except Exception:
                try:
                    df = pd.read_excel(file_path, engine="xlrd")
                except Exception:
                    df = pd.read_excel(file_path)

            # Clean the dataframe
            df = self._clean_dataframe(df)

            if df.empty:
                logger.warning(f"File {file_name} is empty after cleaning")
                return {"status": "error", "reason": "empty_after_cleaning"}

            # Generate table name
            table_name = self._generate_table_name(file_name)

            # Store in DuckLake (Parquet) if enabled
            parquet_path = None
            if self.use_parquet:
                parquet_path = self.duck_lake_path / f"{table_name}.parquet"
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved to DuckLake: {parquet_path}")

            # Store in DuckDB
            self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            self.conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM df")

            # Update metadata
            self.conn.execute(
                """
                INSERT OR REPLACE INTO file_metadata 
                (file_path, file_name, file_hash, table_name, parquet_path, 
                 processed_at, row_count, column_count, file_size_bytes, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    str(file_path),
                    file_name,
                    file_hash,
                    table_name,
                    str(parquet_path) if parquet_path else None,
                    datetime.now(),
                    len(df),
                    len(df.columns),
                    file_path.stat().st_size,
                    "success",
                ],
            )

            logger.info(
                f"Successfully processed {file_name}: {len(df)} rows, {len(df.columns)} columns"
            )

            return {
                "status": "success",
                "table_name": table_name,
                "rows": len(df),
                "columns": len(df.columns),
                "parquet_path": str(parquet_path) if parquet_path else None,
            }

        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")

            # Update metadata with error
            self.conn.execute(
                """
                INSERT OR REPLACE INTO file_metadata 
                (file_path, file_name, file_hash, processed_at, status)
                VALUES (?, ?, ?, ?, ?)
            """,
                [
                    str(file_path),
                    file_path.name,
                    "",
                    datetime.now(),
                    f"error: {str(e)}",
                ],
            )

            return {"status": "error", "reason": str(e)}

    def process_directory(self, recursive: bool = True) -> Dict:
        """
        Process all Excel files in the source directory

        Args:
            recursive: Whether to search subdirectories

        Returns:
            Dictionary with processing summary
        """
        excel_extensions = [".xlsx", ".xls", ".xlsm", ".xlsb"]

        if recursive:
            excel_files = []
            for ext in excel_extensions:
                excel_files.extend(self.source_dir.rglob(f"*{ext}"))
        else:
            excel_files = []
            for ext in excel_extensions:
                excel_files.extend(self.source_dir.glob(f"*{ext}"))

        if not excel_files:
            logger.warning(f"No Excel files found in {self.source_dir}")
            return {"status": "no_files", "processed": 0, "errors": 0, "skipped": 0}

        logger.info(f"Found {len(excel_files)} Excel files to process")

        results = {"processed": 0, "errors": 0, "skipped": 0, "details": []}

        for file_path in excel_files:
            result = self.process_excel_file(file_path)
            results["details"].append({"file": file_path.name, **result})

            if result["status"] == "success":
                results["processed"] += 1
            elif result["status"] == "error":
                results["errors"] += 1
            elif result["status"] == "skipped":
                results["skipped"] += 1

        logger.info(
            f"Processing complete: {results['processed']} processed, "
            f"{results['errors']} errors, {results['skipped']} skipped"
        )

        return results

    def get_available_tables(self) -> List[Dict]:
        """Get list of available tables from processed files"""
        tables = self.conn.execute("""
            SELECT table_name, file_name, processed_at, row_count, column_count, status
            FROM file_metadata 
            WHERE status = 'success'
            ORDER BY processed_at DESC
        """).fetchall()

        return [
            {
                "table_name": row[0],
                "file_name": row[1],
                "processed_at": row[2],
                "row_count": row[3],
                "column_count": row[4],
                "status": row[5],
            }
            for row in tables
        ]

    def query_table(self, table_name: str, query: str = None) -> pd.DataFrame:
        """
        Query a table from DuckDB

        Args:
            table_name: Name of the table to query
            query: Optional custom SQL query (if None, returns all data)

        Returns:
            Pandas DataFrame with query results
        """
        if query is None:
            query = f"SELECT * FROM {table_name}"

        try:
            return self.conn.execute(query).df()
        except Exception as e:
            logger.error(f"Error querying table {table_name}: {e}")
            raise

    def get_table_schema(self, table_name: str) -> List[Dict]:
        """Get schema information for a table"""
        try:
            schema = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            return [
                {
                    "column_name": row[0],
                    "column_type": row[1],
                    "null": row[2],
                    "key": row[3],
                    "default": row[4],
                    "extra": row[5],
                }
                for row in schema
            ]
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {e}")
            raise

    def cleanup_old_files(self, days_old: int = 30):
        """Clean up old processed files from metadata and storage"""
        cutoff_date = datetime.now() - pd.Timedelta(days=days_old)

        # Get old files
        old_files = self.conn.execute(
            """
            SELECT file_path, table_name, parquet_path 
            FROM file_metadata 
            WHERE processed_at < ?
        """,
            [cutoff_date],
        ).fetchall()

        cleaned_count = 0
        for file_path, table_name, parquet_path in old_files:
            try:
                # Drop table if exists
                if table_name:
                    self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")

                # Remove parquet file if exists
                if parquet_path and Path(parquet_path).exists():
                    Path(parquet_path).unlink()

                # Remove from metadata
                self.conn.execute(
                    "DELETE FROM file_metadata WHERE file_path = ?", [file_path]
                )

                cleaned_count += 1

            except Exception as e:
                logger.error(f"Error cleaning up {file_path}: {e}")

        logger.info(f"Cleaned up {cleaned_count} old files")
        return cleaned_count

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB connection closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_excel_processor(
    source_dir: str,
    duck_lake_path: str = "ducklake",
    duckdb_path: str = "databases/main.duckdb",
    use_parquet: bool = True,
    auto_clean: bool = True,
) -> ExcelProcessor:
    """Factory function to create an ExcelProcessor instance"""
    return ExcelProcessor(
        source_dir=source_dir,
        duck_lake_path=duck_lake_path,
        duckdb_path=duckdb_path,
        use_parquet=use_parquet,
        auto_clean=auto_clean,
    )


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(
        description="Process Excel files to DuckDB/DuckLake"
    )
    parser.add_argument("source_dir", help="Directory containing Excel files")
    parser.add_argument(
        "--duck-lake-path", default="ducklake", help="DuckLake storage path"
    )
    parser.add_argument(
        "--duckdb-path", default="databases/main.duckdb", help="DuckDB file path"
    )
    parser.add_argument(
        "--no-parquet", action="store_true", help="Disable Parquet storage"
    )
    parser.add_argument(
        "--no-clean", action="store_true", help="Disable automatic data cleaning"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Process files
    with create_excel_processor(
        source_dir=args.source_dir,
        duck_lake_path=args.duck_lake_path,
        duckdb_path=args.duckdb_path,
        use_parquet=not args.no_parquet,
        auto_clean=not args.no_clean,
    ) as processor:
        results = processor.process_directory()
        print(f"Processing complete: {results}")
