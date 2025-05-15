import os
import logging
import datetime
import time
import asyncio  # Add missing import for asyncio

from typing import Optional, Union, Dict, List, Any
from tools.database import DatabaseConnector
from tools.mongodb import MongoDBConnector
from tools.pythontools import PythonREPL, data_visualization
from pymongo import MongoClient

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(
        self,
        sql_connection_string: Optional[str] = None,
        mongo_connection_string: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        llm_model_name: str = "qwen2.5",
        coding_model_name: str = "qwen2.5-coder",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        auth_source: Optional[str] = "admin",
    ):
        self.sql_db = None
        self.mongo_db = None
        self.python_repl = PythonREPL()
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.llm_model_name = llm_model_name
        self.coding_model_name = coding_model_name
        self.auth_source = auth_source

        if sql_connection_string:
            retry_count = 0
            last_error = None
            while retry_count < max_retries:
                try:
                    self.sql_db = DatabaseConnector(
                        sql_connection_string,
                        openai_api_key=openai_api_key,
                        openai_api_base=openai_api_base,
                        llm_model_name=coding_model_name,
                    )
                    logger.info("SQL database connection established")
                    break
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(
                            f"Retry {retry_count}/{max_retries} connecting to SQL database..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to connect to SQL database after {max_retries} attempts: {e}"
                        )
                        raise last_error

        if mongo_connection_string:
            retry_count = 0
            last_error = None
            while retry_count < max_retries:
                try:
                    from urllib.parse import urlparse, parse_qs, urlencode

                    parsed = urlparse(mongo_connection_string)
                    query_params = parse_qs(parsed.query)

                    if "authSource" not in query_params:
                        query_params["authSource"] = [self.auth_source]
                        new_query = urlencode(query_params, doseq=True)
                        mongo_connection_string = parsed._replace(
                            query=new_query
                        ).geturl()

                    safe_conn_str = mongo_connection_string.replace(
                        "//" + mongo_connection_string.split("//")[1].split("@")[0],
                        "//<credentials>",
                    )
                    logger.info(f"Attempting MongoDB connection: {safe_conn_str}")

                    logger.debug(
                        f"Connection string passed to MongoDBConnector: {mongo_connection_string}"
                    )

                    logger.info("Initializing MongoDBConnector...")

                    logger.info(
                        f"MongoDBConnector using connection string: {mongo_connection_string}"
                    )

                    self.mongo_db = MongoDBConnector(
                        mongo_connection_string,
                        openai_api_key=openai_api_key,
                        openai_api_base=openai_api_base,
                        llm_model_name=coding_model_name,
                        auth_source=query_params["authSource"][0],
                    )

                    break
                except Exception as e:
                    last_error = e
                    logger.error(
                        f"Error during MongoDB connection attempt {retry_count + 1}/{max_retries}: {e}"
                    )
                    if retry_count < max_retries:
                        logger.warning(
                            f"Retrying MongoDB connection ({retry_count + 1}/{max_retries})..."
                        )
                        time.sleep(retry_delay)
                    else:
                        logger.error(
                            f"Failed to connect to MongoDB after {max_retries} attempts: {e}"
                        )
                        raise last_error

    async def analyze_data(
        self,
        query: str,
        db_type: str,
        target_db: str,
        visualization_required: bool = False,
        report_format: str = "text",
    ) -> Dict[str, Any]:
        """
        Analyze data from specified database based on natural language query.

        Args:
            query: Natural language query describing the analysis needed
            db_type: Type of database ("sql" or "mongodb")
            target_db: Name of the target database
            visualization_required: Whether to generate visualization
            report_format: Output format ("text", "markdown", "html")

        Returns:
            Dictionary containing analysis results and metadata
        """
        try:
            if not query or not query.strip():
                return {"error": "Query cannot be empty", "status": "error"}

            if db_type not in ["sql", "mongodb"]:
                return {
                    "error": f"Unsupported database type: {db_type}",
                    "status": "error",
                }

            if db_type == "sql" and not self.sql_db:
                return {"error": "SQL database not connected", "status": "error"}
            elif db_type == "mongodb" and not self.mongo_db:
                return {"error": "MongoDB not connected", "status": "error"}

            try:
                raw_data = await asyncio.wait_for(
                    self._fetch_data(query, db_type, target_db),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                return {"error": "Database query timed out", "status": "error"}

            if not raw_data:
                return {"error": "No data returned from query", "status": "error"}

            report = await self._generate_report(
                query, raw_data, report_format, visualization_required
            )

            return {
                "status": "success",
                "query": query,
                "data": raw_data,
                "report": report,
                "metadata": {
                    "db_type": db_type,
                    "target_db": target_db,
                    "report_format": report_format,
                },
            }

        except Exception as e:
            logger.error(f"Error in analyze_data: {e}")
            return {"error": str(e), "status": "error"}

    async def _fetch_data(
        self, query: str, db_type: str, target_db: str
    ) -> Union[List[Dict[str, Any]], None]:
        """Fetch data from specified database using natural language query"""
        try:
            if db_type == "sql":
                if not self.sql_db:
                    raise ValueError("SQL database connection not initialized")

                sql_query = await self.sql_db.generate_sql_from_natural_language(query)
                results = self.sql_db.execute_query(sql_query)
                return results

            elif db_type == "mongodb":
                if not self.mongo_db:
                    raise ValueError("MongoDB connection not initialized")

                if hasattr(
                    self.mongo_db, "generate_mongodb_query_from_natural_language"
                ):
                    logger.info(f"MongoDB: target_db={target_db}, query='{query}'")
                    mongo_query = (
                        self.mongo_db.generate_mongodb_query_from_natural_language(
                            query,
                            target_db_name="my_test_db",
                        )
                    )
                    logger.info(f"Query to MongoDB: {mongo_query}")

                    query_type = (
                        "aggregate" if isinstance(mongo_query, list) else "find"
                    )

                    if "." in target_db:
                        db_name, collection_name = target_db.split(".", 1)
                    else:
                        db_name = "my_test_db"
                        collection_name = target_db

                    logger.info(
                        f"Accessing MongoDB database: {db_name}, collection: {collection_name}"
                    )

                    collection = self.mongo_db.get_collection(db_name, collection_name)

                    if query_type == "find":
                        results = list(collection.find(mongo_query))
                    elif query_type == "aggregate":
                        results = list(
                            collection.aggregate(mongo_query, allowDiskUse=True)
                        )
                    else:
                        raise ValueError(f"Unsupported query type: {query_type}")

                    return results
                else:
                    logger.error(
                        "MongoDBConnector is not being used. Natural language queries for MongoDB are not supported with direct MongoClient. Please use MongoDBConnector."
                    )
                    return {
                        "error": "MongoDBConnector is not being used. Natural language queries for MongoDB are not supported with direct MongoClient.",
                        "status": "error",
                    }

        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise

    async def _generate_llm_report(self, prompt: str) -> str:
        """Generate a report using the configured LLM"""
        try:
            import openai

            if self.openai_api_key == "ollama":
                import requests

                response = requests.post(
                    f"{self.openai_api_base}/chat/completions",
                    json={
                        "model": self.llm_model_name,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                )
                return response.json()["choices"][0]["message"]["content"]
            else:
                openai.api_key = self.openai_api_key
                if self.openai_api_base:
                    openai.api_base = self.openai_api_base

                response = await openai.ChatCompletion.acreate(
                    model=self.llm_model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error generating LLM report: {e}")
            return f"Error generating report: {str(e)}"

    async def _generate_report(
        self,
        query: str,
        data: List[Dict[str, Any]],
        report_format: str,
        visualization_required: bool,
    ) -> Dict[str, Any]:
        """Generate analysis report with optional visualization"""
        try:
            visualization = None
            if visualization_required:
                visualization_code = f"""
import pandas as pd
import matplotlib.pyplot as plt

# Convert data to DataFrame
df = pd.DataFrame({data})

# Create appropriate visualization based on data
plt.figure(figsize=(10, 6))
{self._generate_visualization_code(data)}
plt.title('Data Visualization')
plt.tight_layout()
"""
                visualization = await data_visualization(
                    visualization_code, self.python_repl
                )

            report_prompt = f"""
Based on the following data, create a detailed {report_format} format report.
The report should include:
1. A summary of the data
2. Key findings and insights
3. Any notable patterns or trends
4. Recommendations if applicable
5. The Answer should be in German.

Original Query: {query}
Data: {data}

Please format the response in {report_format} format.
"""
            report_content = await self._generate_llm_report(report_prompt)

            report = {
                "summary": report_content,
                "visualization": visualization if visualization_required else None,
                "format": report_format,
                "timestamp": datetime.datetime.now().isoformat(),
            }

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def _generate_visualization_code(self, data: List[Dict[str, Any]]) -> str:
        """Generate appropriate visualization code based on data structure"""

        columns = list(data[0].keys()) if data else []
        if len(columns) < 2:
            raise ValueError("Insufficient columns for visualization.")

        index_column = columns[0]
        y_axis_column = columns[1]

        return f"""
# Set the index dynamically
df.set_index('{index_column}', inplace=True)

# Create bar plot
df.plot(kind='bar', y='{y_axis_column}', legend=False)
plt.xlabel('{index_column.replace("_", " ").capitalize()}')
plt.ylabel('{y_axis_column.replace("_", " ").capitalize()}')
plt.xticks(rotation=45)
"""

    def close(self):
        """Close all database connections"""
        if self.sql_db:
            try:
                self.sql_db.close_connection()
                logger.info("SQL database connection closed")
            except Exception as e:
                logger.error(f"Error closing SQL connection: {e}")

        if self.mongo_db:
            try:
                self.mongo_db.close_connection()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")


async def initialize_mongodb(conn_str: str) -> bool:
    """Initialize MongoDB with required user and collections"""
    try:
        logger.info("Verifying MongoDB user credentials...")
        test_client = MongoClient(conn_str)
        test_client["my_test_db"].list_collection_names()
        logger.info("MongoDB user credentials verified successfully")

        client = MongoClient(conn_str)
        test_db = client.my_test_db
        if "users_test" not in test_db.list_collection_names():
            test_db.users_test.insert_many(
                [
                    {"name": "John Doe", "age": 25, "city": "Berlin"},
                    {"name": "Jane Smith", "age": 30, "city": "Hamburg"},
                    {"name": "Max Mustermann", "age": 35, "city": "Berlin"},
                    {"name": "Anna Müller", "age": 28, "city": "Munich"},
                ]
            )
            logger.info("MongoDB test data initialized")

        return True
    except Exception as e:
        logger.error(f"Error initializing MongoDB: {e}")
        return False


async def main():
    sql_conn_str = os.getenv(
        "SQL_CONNECTION_STRING",
        "mariadb+mariadbconnector://dataanalyzer:dataanalyzer_pwd@localhost:3306/my_test_db",
    )
    mongo_conn_str = os.getenv(
        "MONGO_CONNECTION_STRING",
        "mongodb://dataanalyzer:dataanalyzer_pwd@localhost:27017/my_test_db?authSource=test",
    )

    def validate_connection_string(conn_str: str, db_type: str) -> bool:
        if not conn_str:
            return False
        if db_type == "mongodb":
            if ":3306" in conn_str:
                logger.error(
                    "Invalid MongoDB port (3306) detected. Using default port 27017"
                )
                return False
            if "authSource=" not in conn_str:
                logger.error("MongoDB connection string missing authSource parameter")
                return False
        return True

    if not validate_connection_string(mongo_conn_str, "mongodb"):
        mongo_conn_str = mongo_conn_str.replace(":3306", ":27017")

    try:
        if await initialize_mongodb(mongo_conn_str):
            logger.info("MongoDB initialization successful")
        else:
            logger.warning("MongoDB initialization failed, but continuing...")
    except Exception as e:
        logger.error(f"Error during MongoDB setup: {e}")

    openai_api_key = os.getenv("OPENAI_API_KEY", "ollama")
    openai_api_base = os.getenv("OPENAI_API_BASE_URL", "http://localhost:11434/v1")
    llm_model = os.getenv("LLM_MODEL", "qwen2.5:latest")
    coding_model = os.getenv("CODING_MODEL", "qwen2.5-coder:latest")

    create_table = """
    CREATE TABLE IF NOT EXISTS users_test (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(100),
        age INT,
        city VARCHAR(100)
    )"""

    insert_data = """
    INSERT INTO users_test (name, age, city)
    SELECT name, age, city FROM (
        SELECT 'John Doe' as name, 25 as age, 'Berlin' as city UNION ALL
        SELECT 'Jane Smith', 30, 'Hamburg' UNION ALL
        SELECT 'Max Mustermann', 35, 'Berlin' UNION ALL
        SELECT 'Anna Müller', 28, 'Munich'
    ) as tmp
    WHERE NOT EXISTS (SELECT 1 FROM users_test LIMIT 1)"""

    analyzer = DataAnalyzer(
        sql_connection_string=sql_conn_str,
        mongo_connection_string=mongo_conn_str,
        openai_api_key=openai_api_key,
        openai_api_base=openai_api_base,
        llm_model_name=llm_model,
        coding_model_name=coding_model,
    )

    if analyzer.sql_db:
        try:
            for sql in [create_table, insert_data]:
                try:
                    analyzer.sql_db.execute_raw_query(sql)
                except Exception as e:
                    fixed_sql = await analyzer.sql_db.validate_and_fix_sql(sql, str(e))
                    analyzer.sql_db.execute_raw_query(fixed_sql)

            logger.info("Database schema initialized successfully")
            schema_info = analyzer.sql_db.get_schema_info()
            logger.info(f"Schema verified with {len(schema_info)} tables")
        except Exception as e:
            logger.error(f"Error initializing/verifying database schema: {e}")

    if analyzer.mongo_db:
        try:
            db_name = "my_test_db"
            collection_name = "users_test"
            db = analyzer.mongo_db.get_collection(db_name, collection_name)

            logger.info(
                f"Checking if collection '{db_name}.{collection_name}' contains data..."
            )
            document_count = db.count_documents({})
            logger.info(
                f"Document count in '{db_name}.{collection_name}': {document_count}"
            )

            if document_count == 0:
                logger.info(
                    f"Initializing MongoDB collection '{db_name}.{collection_name}' with test data..."
                )
                try:
                    db.insert_many(
                        [
                            {"name": "John Doe", "age": 25, "city": "Berlin"},
                            {"name": "Jane Smith", "age": 30, "city": "Hamburg"},
                            {"name": "Max Mustermann", "age": 35, "city": "Berlin"},
                            {"name": "Anna Müller", "age": 28, "city": "Munich"},
                        ]
                    )

                    inserted_count = db.count_documents({})
                    logger.info(
                        f"Data successfully inserted into '{db_name}.{collection_name}'. Document count: {inserted_count}"
                    )
                except Exception as insert_error:
                    logger.error(
                        f"Error inserting data into collection '{db_name}.{collection_name}': {insert_error}"
                    )
            else:
                logger.info(
                    f"MongoDB collection '{db_name}.{collection_name}' already contains data."
                )
        except Exception as e:
            logger.error(f"Error initializing MongoDB schema: {e}")

    return analyzer


if __name__ == "__main__":
    import asyncio

    async def run_example():
        try:
            analyzer = await main()

            queries = [
                "Zeige mir die Durchschnittsalter der Benutzer nach Stadt",
                "Show me the average age of users by city",
            ]

            for query in queries:
                result = await analyzer.analyze_data(
                    query=query,
                    db_type="sql",
                    target_db="my_test_db.users_test",
                    visualization_required=True,
                    report_format="markdown",
                )

                print(f"\nAnalysis Results for query: {query}")
                print(f"Status: {result.get('status', 'error')}")
                if result.get("status") == "error":
                    print(f"Error: {result.get('error')}")
                else:
                    print("\nReport:")
                    print(result["report"].get("summary", "No summary available"))

        except Exception as e:
            logger.error(f"Error in example: {e}")
        finally:
            if "analyzer" in locals():
                analyzer.close()

    asyncio.run(run_example())
