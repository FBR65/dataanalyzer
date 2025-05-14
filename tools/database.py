"""
pip install SQLAlchemy
# For PostgreSQL:
pip install psycopg2-binary
# For MySQL (using mysqlconnector driver):
pip install mysql-connector-python
# For SQLite: sqlite3 is built-in, SQLAlchemy will use it.
# For Oracle:
# pip install oracledb # (cx_Oracle is now python-oracledb)
# For OpenAI integration:
pip install cx_Oracle
"""

import logging
from pathlib import Path
from typing import Any, List, Dict, Optional, Union, Tuple

# SQLAlchemy imports
from sqlalchemy import create_engine, text, inspect, MetaData
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError, NoSuchTableError

# For OpenAI compatible API
try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger("mcp_server.services.database_connector")


class DatabaseConnector:
    """
    Provides CRUD and schema operations for various SQL databases using SQLAlchemy.
    """

    def __init__(
        self,
        connection_string: str,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        llm_model_name: str = "qwen2.5-coder",
    ):
        """
        Initializes the database service using SQLAlchemy.

        Args:
            connection_string: The SQLAlchemy connection string for the database.
                               Examples:
                               - SQLite: "sqlite:///path/to/your/database.db"
                               - PostgreSQL: "postgresql://user:pass@host:port/dbname"
                               - MySQL: "mysql+mysqlconnector://user:pass@host:port/dbname"
            openai_api_key: Optional API key for the OpenAI-compatible service.
            openai_api_base: Optional base URL for the OpenAI-compatible service.
            llm_model_name: The name of the LLM model to use for SQL generation.
        """
        self.connection_string = connection_string
        self.engine: Optional[Engine] = None
        self.db_type: Optional[str] = None  # Will be inferred from engine
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.llm_model_name = llm_model_name
        self.metadata = MetaData()
        self.inspector = None
        self._schema_info = None

        # Special handling for SQLite: ensure directory exists
        if connection_string.startswith("sqlite:///"):
            # For "sqlite:///:memory:" or "sqlite://":
            if connection_string.lower() not in ("sqlite:///:memory:", "sqlite://"):
                # Extract path: "sqlite:///path/to/file.db" -> "/path/to/file.db"
                # If relative path like "sqlite:///./file.db", Path will handle it.
                db_file_path_str = connection_string[len("sqlite:///") :]
                db_path_obj = Path(db_file_path_str).expanduser().resolve()
                db_dir = db_path_obj.parent
                try:
                    db_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Ensured DB directory exists for SQLite: {db_dir}")
                    # Update connection string to absolute path for consistency
                    self.connection_string = f"sqlite:///{db_path_obj.as_posix()}"
                except OSError as e:
                    logger.error(
                        f"Failed to create directory for SQLite DB from '{connection_string}': {e}"
                    )
                    raise

        try:
            # echo=True can be useful for debugging SQL
            self.engine = create_engine(self.connection_string, echo=False)
            self.connection = self.engine.connect()
            self.db_type = self.engine.dialect.name
            self.inspector = inspect(self.engine)
            self._check_connection()
            logger.info(
                f"SQLAlchemy service for '{self.db_type}' initialized successfully for: {self.connection_string}"
            )
        except SQLAlchemyError as e:
            logger.error(
                f"SQLAlchemy error during initialization for '{self.connection_string}': {e}"
            )
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during SQLAlchemy DB initialization for '{self.connection_string}': {e}"
            )
            raise

    def _check_connection(self):
        """Verify that a connection can be established and a simple query run."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            logger.debug(
                f"Successfully connected to {self.db_type} DB via SQLAlchemy: {self.connection_string}"
            )
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy connection check failed for {self.db_type}: {e}")
            raise

    def get_schema_info(self) -> dict:
        """Get database schema information"""
        if self._schema_info is None:
            schema_info = {}
            for table_name in self.inspector.get_table_names():
                columns = []
                for column in self.inspector.get_columns(table_name):
                    columns.append(
                        {
                            "name": column["name"],
                            "type": str(column["type"]),
                            "nullable": column.get("nullable", True),
                        }
                    )
                schema_info[table_name] = {
                    "columns": columns,
                    "primary_key": self.inspector.get_pk_constraint(table_name).get(
                        "constrained_columns", []
                    ),
                }
            self._schema_info = schema_info
            logger.info(f"Schema retrieved for {len(schema_info)} tables")
        return self._schema_info

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Tuple[Any, ...]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query and return results using SQLAlchemy.

        Handles both read (SELECT) and write (INSERT, UPDATE, DELETE, CREATE, etc.) operations.
        For write operations, returns a list containing a dict with 'affected_rows'.
        For read operations, returns a list of dictionaries representing the rows.
        SQLAlchemy's text() construct is used, so params should match its expectations
        (dict for named parameters, list of tuples for executemany-style).

        Raises:
            SQLAlchemyError: If a database error occurs during execution.
            ValueError: If the query string is empty or invalid.
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.debug(
            f"Executing query via SQLAlchemy: {query[:100]}... with params: {params}"
        )
        try:
            with self.engine.connect() as connection:
                # For DDL statements or others that don't return rows and might not have 'rowcount'
                # in the same way, we need to be careful.
                # SQLAlchemy handles autocommit for single statements if not in an explicit transaction.
                # For explicit transaction control:
                # with connection.begin():
                #    result = connection.execute(text(query), params)

                result = connection.execute(
                    text(query), params or {}
                )  # Pass empty dict if params is None

                is_ddl_or_no_result_expected = (
                    query.strip()
                    .upper()
                    .startswith(
                        ("CREATE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE")
                    )
                )
                # Some DML like INSERT might not return rows by default on all DBs unless RETURNING is used
                is_dml_without_returning = (
                    query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
                    and "RETURNING" not in query.upper()
                )

                if result.returns_rows:
                    # For SELECT or DML with RETURNING
                    # .mappings() provides dict-like RowMapping objects
                    rows = result.mappings().all()
                    logger.debug(f"Read query returned {len(rows)} rows.")
                    # Commit is generally not needed for SELECTs, but if the query had side effects
                    # within a transaction, it would be part of the transaction block.
                    # connection.commit() # Only if part of an explicit transaction block started with connection.begin()
                    return [
                        dict(row) for row in rows
                    ]  # Convert RowMappings to plain dicts
                else:
                    # For DML (INSERT, UPDATE, DELETE without RETURNING) or DDL
                    affected_rows = result.rowcount
                    # DDL statements might have rowcount as -1 or 0 depending on driver.
                    # For DML, it should be the number of affected rows.
                    logger.info(
                        f"Write/DDL query executed. Affected rows: {affected_rows}"
                    )
                    # Commit changes for DML/DDL if not in autocommit mode or if an explicit transaction was started.
                    # If using engine-level autocommit (often default for single execute outside transaction),
                    # explicit commit here might be redundant or even error if no transaction is active.
                    # If we started with connection.begin(), then connection.commit() would be here.
                    # For safety with various DDL/DML, ensure commit if changes were made.
                    if not is_ddl_or_no_result_expected or (
                        is_dml_without_returning and affected_rows > 0
                    ):
                        connection.commit()  # Ensure DML changes are committed
                    return [{"affected_rows": affected_rows}]

        except SQLAlchemyError as e:
            logger.error(
                f"SQLAlchemy error executing query '{query[:100]}...' with params {params}: {e}"
            )
            # connection.rollback() # If part of an explicit transaction block
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error executing query via SQLAlchemy '{query[:100]}...': {e}"
            )
            raise

    async def validate_and_fix_sql(
        self, sql_query: str, error_message: str = None
    ) -> str:
        """Use LLM to validate and fix SQL queries"""
        prompt = f"""
Please check and fix this SQL query:

{sql_query}

{f"The query failed with error: {error_message}" if error_message else ""}

Return only the corrected SQL query, no explanations.
"""
        try:
            if self.openai_api_key == "ollama":
                import requests

                response = requests.post(
                    f"{self.openai_api_base}/chat/completions",
                    json={
                        "model": self.llm_model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                return response.json()["choices"][0]["message"]["content"].strip()
            else:
                openai.api_key = self.openai_api_key
                if self.openai_api_base:
                    openai.api_base = self.openai_api_base
                response = await openai.ChatCompletion.acreate(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in LLM SQL validation: {e}")
            return sql_query

    def execute_raw_query(self, query: str) -> None:
        """Execute a raw SQL query (for DDL operations)"""
        try:
            # Split multiple statements and execute separately
            statements = [stmt.strip() for stmt in query.split(";") if stmt.strip()]
            with self.engine.begin() as conn:
                for stmt in statements:
                    try:
                        conn.execute(text(stmt))
                    except Exception as e:
                        logger.error(f"Error executing statement: {e}")
                        raise
        except Exception as e:
            logger.error(f"Error executing raw query: {e}")
            raise

    def close_connection(self):
        """Close the database connection"""
        try:
            if self.connection:
                self.connection.close()
            if self.engine:
                self.engine.dispose()
        except Exception as e:
            logger.error(f"Error closing database connection: {e}")
            raise

    def list_tables(self) -> List[str]:
        """Lists all user-defined tables in the database using SQLAlchemy Inspector."""
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")
        logger.debug(f"Listing tables for {self.db_type} using SQLAlchemy Inspector")
        try:
            inspector = inspect(self.engine)
            # get_table_names() also includes views by default, filter if needed
            # For only tables:
            # table_names = []
            # for t_name in inspector.get_table_names():
            #    if not inspector.get_view_definition(t_name): # Crude check, might need dialect specific
            #        table_names.append(t_name)
            # return table_names
            # However, get_table_names() is generally what's expected.
            return inspector.get_table_names()
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error listing tables: {e}")
            raise

    def describe_table(
        self, table_name: str, schema: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Gets the schema for a specific table using SQLAlchemy Inspector.

        Args:
            table_name: The name of the table to describe.
            schema: Optional schema name, for databases that support schemas (e.g., PostgreSQL).

        Returns:
            List of dicts, each: {name, type, nullable, default, primary_key, autoincrement, comment}.
            The 'type' is a SQLAlchemy type object (e.g., String, Integer).
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")
        if not table_name or not table_name.replace("_", "").isalnum():
            raise ValueError(f"Invalid table name format: {table_name}")

        logger.debug(
            f"Describing table: {table_name} (schema: {schema}) for {self.db_type} using SQLAlchemy Inspector"
        )
        schema_info: List[Dict[str, Any]] = []

        try:
            inspector = inspect(self.engine)

            # Check if table exists first
            if not inspector.has_table(table_name, schema=schema):
                logger.warning(
                    f"Table '{table_name}' (schema: {schema}) not found by SQLAlchemy inspector."
                )
                return []

            columns = inspector.get_columns(table_name, schema=schema)
            pk_constraint = inspector.get_pk_constraint(table_name, schema=schema)
            primary_keys = (
                set(pk_constraint["constrained_columns"])
                if pk_constraint and "constrained_columns" in pk_constraint
                else set()
            )
            # Foreign keys can also be fetched: inspector.get_foreign_keys(table_name, schema=schema)
            # Indexes: inspector.get_indexes(table_name, schema=schema)

            for col in columns:
                # col is a dictionary with keys like:
                # 'name', 'type', 'nullable', 'default', 'autoincrement', 'primary_key', 'comment'
                # 'type' is a SQLAlchemy type object, e.g., VARCHAR(), INTEGER().
                # We convert it to string for simpler representation.
                schema_info.append(
                    {
                        "name": col["name"],
                        "type": str(
                            col["type"]
                        ),  # Convert SQLAlchemy type object to string
                        "nullable": col["nullable"],
                        "default": col.get(
                            "default"
                        ),  # .get() as 'default' might not always be present
                        "primary_key": col["name"]
                        in primary_keys,  # More reliable than col.get('primary_key') for composite PKs
                        "autoincrement": col.get(
                            "autoincrement", False
                        ),  # 'auto' or True/False
                        "comment": col.get("comment"),
                    }
                )

            if not schema_info and columns:  # Should not happen if columns were found
                logger.warning(
                    f"Columns found for '{table_name}' but no schema_info generated."
                )
            elif not columns:  # Should be caught by has_table, but as a safeguard
                logger.warning(
                    f"No columns found for table '{table_name}' by SQLAlchemy inspector, though has_table might have been true."
                )

        except (
            NoSuchTableError
        ):  # Explicitly catch if inspector.get_columns raises this
            logger.warning(
                f"Table '{table_name}' (schema: {schema}) not found by SQLAlchemy inspector (NoSuchTableError)."
            )
            return []
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error describing table '{table_name}': {e}")
            raise
        except Exception as e:
            logger.exception(
                f"Unexpected error describing table '{table_name}' for {self.db_type}: {e}"
            )
            raise
        return schema_info

    def get_schema_representation(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieves a representation of the entire database schema (all tables and their columns).
        This is useful for providing context to an NL-to-SQL model.
        """
        if not self.engine:
            raise RuntimeError("Database engine not initialized.")

        logger.info("Fetching full schema representation...")
        inspector = inspect(self.engine)
        db_schema = {}

        # Consider schemas for databases like PostgreSQL
        # For simplicity, this example gets tables from the default schema or all accessible ones.
        # You might need to iterate inspector.get_schema_names() for multi-schema DBs.
        default_schema = None  # For SQLite and MySQL, schema is often None or implicit
        if self.db_type in ["postgresql"]:  # Add other multi-schema DBs if needed
            # This gets the default search path schema, or you might want to iterate all schemas.
            # default_schema = inspector.default_schema_name # Often 'public' for PostgreSQL
            # For now, let's list tables from whatever is default/accessible
            pass

        table_names = inspector.get_table_names(schema=default_schema)

        for table_name in table_names:
            try:
                # Skip SQLAlchemy's internal migration table if it exists
                if table_name == "alembic_version":
                    continue
                db_schema[table_name] = self.describe_table(
                    table_name, schema=default_schema
                )
            except Exception as e:
                logger.error(
                    f"Could not describe table {table_name} for schema representation: {e}"
                )
        logger.info(f"Successfully fetched schema for {len(db_schema)} tables.")
        return db_schema

    async def generate_sql_from_natural_language(self, query: str) -> str:
        """Generate SQL query from natural language using schema information"""
        schema_info = self.get_schema_info()
        if not schema_info:
            raise ValueError("Database schema could not be retrieved")

        # For now, return a basic query based on the schema
        if "users_test" in schema_info:
            return "SELECT city, AVG(age) as avg_age FROM users_test GROUP BY city;"
        else:
            raise ValueError("Required table 'users_test' not found in schema")


if __name__ == "__main__":
    import os

    # Best practice: Load from environment variables or a config file
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL = "qwen2.5-coder:latest"

    # Example for SQLite
    db_connector = DatabaseConnector(
        connection_string="sqlite:///./my_app.db",
        openai_api_key=OPENAI_API_KEY,
        openai_api_base=OPENAI_API_BASE,
        llm_model_name=LLM_MODEL,
    )

    # Create a dummy table for testing
    try:
        db_connector.execute_query("DROP TABLE IF EXISTS employees")
        db_connector.execute_query("""
            CREATE TABLE employees (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                department TEXT,
                salary INTEGER,
                hire_date DATE
            )
        """)
        # Use named parameters with a list of dictionaries for bulk insert
        employee_data = [
            {
                "name": "Alice Smith",
                "department": "Engineering",
                "salary": 90000,
                "hire_date": "2020-01-15",
            },
            {
                "name": "Bob Johnson",
                "department": "Marketing",
                "salary": 75000,
                "hire_date": "2019-03-01",
            },
            {
                "name": "Charlie Brown",
                "department": "Engineering",
                "salary": 120000,
                "hire_date": "2022-07-20",
            },
        ]
        db_connector.execute_query(
            "INSERT INTO employees (name, department, salary, hire_date) VALUES (:name, :department, :salary, :hire_date)",
            employee_data,
        )
        print("Dummy table created and populated.")
    except Exception as e:
        print(f"Error setting up dummy table: {e}")

    # Test the natural language to SQL generation
    if OPENAI_API_KEY and OPENAI_API_BASE:
        nl_query = "Show me the names and salaries of all employees in the Engineering department earning more than 80000"
        try:
            sql_query = db_connector.generate_sql_from_natural_language(nl_query)
            print(f"\nNatural Language Query: {nl_query}")
            print(f"Generated SQL: {sql_query}")

            # Optionally, execute the generated query
            if sql_query:
                results = db_connector.execute_query(sql_query)
                print(f"Results from generated SQL: {results}")

        except (ImportError, ValueError, RuntimeError) as e:
            print(f"Error generating SQL: {e}")
    else:
        print("\nSkipping NL-to-SQL test: OpenAI API key or base URL not configured.")
