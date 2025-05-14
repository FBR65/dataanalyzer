import os
import gradio as gr
from main import DataAnalyzer
import base64
from PIL import Image
import io
from urllib.parse import quote_plus  # Add this import

# Database connection templates
DB_TEMPLATES = {
    "postgresql": "postgresql://{user}:{password}@{host}:{port}/{database}",
    "mysql": "mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}",
    "mariadb": "mariadb+mariadbconnector://{user}:{password}@{host}:{port}/{database}",
    "oracle": "oracle+oracledb://{user}:{password}@{host}:{port}/{database}",
    "sqlite": "sqlite:///{database}",
}

# Add default ports for different database types
DB_DEFAULT_PORTS = {
    "postgresql": "5432",
    "mysql": "3306",
    "mariadb": "3306",
    "oracle": "1521",
    "sqlite": "",  # No port needed for SQLite
    "mongodb": "27017",
}

CSS = """
.connection-status {
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
}
.success {
    background-color: #d4edda;
    color: #155724;
    border: 1px solid #c3e6cb;
}
.error {
    background-color: #f8d7da;
    color: #721c24;
    border: 1px solid #f5c6cb;
}
"""


def build_mongodb_connection_string(
    username, password, host, port, target_db="my_test_db", auth_db="test"
):
    """Helper function to build MongoDB connection string consistently"""
    if not port or port == "3306":
        port = DB_DEFAULT_PORTS["mongodb"]
    # URL encode username and password, include target_db in connection string
    username = quote_plus(username)
    password = quote_plus(password)
    return f"mongodb://{username}:{password}@{host}:{port}/{target_db}?authSource={auth_db}"


async def test_connection(
    db_type: str,
    sql_flavor: str,
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
    auth_source: str,  # Add auth_source parameter
):
    try:
        if db_type == "mongodb":
            conn_str = build_mongodb_connection_string(
                username=username,
                password=password,
                host=host,
                port=port,
                auth_db=auth_source,
            )
            from pymongo import MongoClient

            client = MongoClient(
                conn_str,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
            )
            db = client.my_test_db
            collections = db.list_collection_names()
            print(f"Available collections: {collections}")  # Debug output
            client.close()
        else:
            if not port:
                port = DB_DEFAULT_PORTS.get(sql_flavor, "3306")
            # Use the *target* database for SQL, not the auth database textbox
            # Use target_collection as the database name for SQL
            db_name = database if db_type == "sql" else None
            # If the user left the database field empty, fallback to target_collection
            if not db_name:
                db_name = "my_test_db"
            conn_template = DB_TEMPLATES[sql_flavor]
            conn_str = conn_template.format(
                user=username,
                password=password,
                host=host,
                port=port,
                database=db_name,
            )
            from sqlalchemy import create_engine, text

            engine = create_engine(conn_str)
            with engine.connect() as connection:
                # Use SQLAlchemy's text() for raw SQL queries
                connection.execute(text("SELECT 1"))

        return "<div class='connection-status success'>✅ Connection successful!</div>"
    except Exception as e:
        return (
            f"<div class='connection-status error'>❌ Connection failed: {str(e)}</div>"
        )


async def process_query(
    query: str,
    db_type: str,
    sql_flavor: str,
    host: str,
    port: str,
    database: str,
    username: str,
    password: str,
    target_collection: str,
    auth_source: str,  # Add auth_source parameter
):
    try:
        if db_type == "mongodb":
            # Use exact same connection string as test_connection
            mongo_conn = build_mongodb_connection_string(
                username=username,
                password=password,
                host=host,
                port=port,
                auth_db=auth_source,
            )
            sql_conn = None

            # Initialize MongoDBConnector with authSource
            analyzer = DataAnalyzer(
                sql_connection_string=sql_conn,
                mongo_connection_string=mongo_conn,
                openai_api_key=os.getenv("OPENAI_API_KEY", "ollama"),
                openai_api_base=os.getenv(
                    "OPENAI_API_BASE_URL", "http://localhost:11434/v1"
                ),
                llm_model_name=os.getenv("LLM_MODEL", "qwen2.5:latest"),
                coding_model_name=os.getenv("CODING_MODEL", "qwen2.5-coder:latest"),
                auth_source=auth_source,  # Use the auth_source textbox value
            )
        else:
            if not port:
                port = DB_DEFAULT_PORTS.get(sql_flavor, "3306")
            conn_template = DB_TEMPLATES[sql_flavor]
            conn_str = conn_template.format(
                user=username,
                password=password,
                host=host,
                port=port,
                database=database,
            )
            sql_conn = conn_str
            mongo_conn = None

            # Initialize analyzer
            analyzer = DataAnalyzer(
                sql_connection_string=sql_conn,
                mongo_connection_string=mongo_conn,
                openai_api_key=os.getenv("OPENAI_API_KEY", "ollama"),
                openai_api_base=os.getenv(
                    "OPENAI_API_BASE_URL", "http://localhost:11434/v1"
                ),
                llm_model_name=os.getenv("LLM_MODEL", "qwen2.5:latest"),
                coding_model_name=os.getenv("CODING_MODEL", "qwen2.5-coder:latest"),
            )

        # Process query with correct target collection
        target_db = target_collection if db_type == "mongodb" else target_collection
        result = await analyzer.analyze_data(
            query=query,
            db_type=db_type,
            target_db=target_db,
            visualization_required=True,
            report_format="markdown",
        )

        # Process visualization if available
        image = None
        if result.get("report", {}).get("visualization"):
            vis_data = result["report"]["visualization"]
            if vis_data.startswith("data:image/png;base64,"):
                img_data = base64.b64decode(vis_data.split(",")[1])
                image = Image.open(io.BytesIO(img_data))

        # Get report summary
        summary = result.get("report", {}).get("summary", "No summary available")

        # Close connections
        analyzer.close()

        return image, summary

    except Exception as e:
        return None, f"**Error**\n\n❌ {str(e)}"


# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Data Analyzer Interface", css=CSS) as interface:
        gr.HTML("<h1><center>Data Analyzer</center></h1>")

        with gr.Row():
            query = gr.Textbox(
                label="Anfrage",
                placeholder="Geben Sie hier Ihre Anfrage ein...",
                lines=3,
            )
        with gr.Row():
            submit_btn = gr.Button("Analyse")

        with gr.Row():
            with gr.Accordion("Datenbankverbindung", open=False):
                with gr.Column():
                    db_type = gr.Radio(
                        choices=["sql", "mongodb"],
                        label="Datenbank Typ",
                        value="mongodb",
                    )

                    with gr.Column(visible=True) as sql_options:
                        sql_flavor = gr.Dropdown(
                            choices=list(DB_TEMPLATES.keys()),
                            label="SQL Datenbank Typ",
                            value="mariadb",
                        )

                    host = gr.Textbox(label="Host", value="localhost")
                    port = gr.Textbox(
                        label="Port",
                        value="3306",
                        placeholder="Wird automatisch basierend auf DB-Typ gesetzt",
                    )
                    # Add a textbox for the SQL/MongoDB database to use (target database)
                    target_database = gr.Textbox(
                        label="Datenbank (SQL: Datenbankname, MongoDB: Ziel-Datenbank)",
                        value="my_test_db",
                        placeholder="Name der Datenbank, z.B. my_test_db",
                    )
                    # Add a textbox for MongoDB authSource
                    auth_source = gr.Textbox(
                        label="Auth Database (authSource für MongoDB)",
                        value="test",
                        placeholder="Database where user credentials are stored (default: 'test')",
                    )
                    username = gr.Textbox(label="Username", value="dataanalyzer")
                    password = gr.Textbox(
                        label="Password", type="password", value="dataanalyzer_pwd"
                    )
                    target_collection = gr.Textbox(
                        label="Collection Name",
                        value="users_test",
                        placeholder="Collection name in my_test_db database",
                    )
                with gr.Row():
                    test_conn_btn = gr.Button("Test Connection")
                connection_status = gr.HTML(
                    value="",
                    label="Connection Status",
                    elem_classes=["connection-status"],
                )

        with gr.Row():
            with gr.Column():
                output_image = gr.Image(label="Visualisierung")
            with gr.Column():
                output_text = gr.Markdown(label="Analyse Bericht")

        # Handle SQL/MongoDB option visibility
        def toggle_sql_options(db_type: str):
            """Toggle visibility of SQL-specific options"""
            return gr.update(visible=(db_type == "sql"))

        db_type.change(toggle_sql_options, db_type, sql_options)

        # Update port and auth db visibility when database type changes
        def update_port_and_visibility(db_type: str, current_port: str):
            if db_type == "mongodb":
                new_port = DB_DEFAULT_PORTS["mongodb"]
                auth_db_visible = True
            else:
                new_port = current_port if current_port != "27017" else "3306"
                auth_db_visible = False

            return {
                port: gr.update(value=new_port),
                sql_options: gr.update(visible=(db_type == "sql")),
                auth_source: gr.update(visible=auth_db_visible),  # Fix: Use auth_source
                connection_status: gr.update(value=""),
            }

        db_type.change(
            fn=update_port_and_visibility,
            inputs=[db_type, port],
            outputs=[
                port,
                sql_options,
                auth_source,
                connection_status,
            ],  # Fix: Use auth_source
        )

        # Add test connection handler
        test_conn_btn.click(
            fn=test_connection,
            inputs=[
                db_type,
                sql_flavor,
                host,
                port,
                target_database,  # Use the new target_database textbox for SQL/MongoDB
                username,
                password,
                auth_source,  # Pass the auth_source textbox value
            ],
            outputs=[connection_status],
        )

        # Submit handler
        submit_btn.click(
            fn=process_query,
            inputs=[
                query,
                db_type,
                sql_flavor,
                host,
                port,
                target_database,  # Use the new target_database textbox for SQL/MongoDB
                username,
                password,
                target_collection,
                auth_source,  # Pass the auth_source textbox value
            ],
            outputs=[output_image, output_text],
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0")
