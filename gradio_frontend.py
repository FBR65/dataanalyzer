import os
import gradio as gr
import pandas as pd
import asyncio
import plotly.express as px
import plotly.io as pio
import logging
import sys
import json

# Configure logging
logger = logging.getLogger(__name__)

# Global state
current_data = None
current_analyzer = None
current_config = None

# Configuration file for storing database settings
CONFIG_FILE = "database_configs.json"


# Try to import DataAnalyzer
try:
    from main import DataAnalyzer
except ImportError:
    DataAnalyzer = None


def build_mongodb_connection_string(
    username, password, host, port, target_db="my_test_db", auth_db="test"
):
    """Helper function to build MongoDB connection string consistently"""
    return f"mongodb://{username}:{password}@{host}:{port}/{target_db}?authSource={auth_db}"


async def execute_query(query, target_collection):
    """Execute query and return data"""
    global current_data, current_analyzer, current_config

    if not current_config:
        return (
            None,
            "❌ Keine Datenbankverbindung konfiguriert. Bitte testen Sie zuerst die Verbindung.",
        )

    try:
        # Simple direct MongoDB query for testing
        from pymongo import MongoClient

        mongo_conn = build_mongodb_connection_string(
            username=current_config["username"],
            password=current_config["password"],
            host=current_config["host"],
            port=current_config["port"],
            auth_db=current_config["auth_source"],
        )

        client = MongoClient(mongo_conn, serverSelectionTimeoutMS=5000)
        db = client.my_test_db
        collection = db[target_collection]

        # Simple find query for testing
        results = list(collection.find({}))
        client.close()

        current_data = results
        if current_data:
            df = pd.DataFrame(current_data)
            return df, f"✅ {len(current_data)} Datensätze erfolgreich abgerufen"
        else:
            return None, "ℹ️ Keine Daten gefunden"

    except Exception as e:
        logger.error(f"Error in execute_query: {e}")
        return None, f"❌ Fehler beim Ausführen der Abfrage: {str(e)}"


def execute_query_sync(query, target_collection):
    """Synchronous wrapper for execute_query"""
    return asyncio.run(execute_query(query, target_collection))


def download_csv():
    """Create CSV download for current data"""
    global current_data
    if current_data:
        df = pd.DataFrame(current_data)
        csv_path = "export_data.csv"
        df.to_csv(csv_path, index=False)
        return csv_path
    return None


def generate_visualization(vis_type):
    """Generate visualization based on selected type"""
    global current_data

    if not current_data:
        return None, "❌ Keine Daten für Visualisierung verfügbar"

    try:
        df = pd.DataFrame(current_data)
        numeric_cols = df.select_dtypes(include=["number"]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        if vis_type == "auto":
            if len(categorical_cols) >= 1:
                value_counts = df[categorical_cols[0]].value_counts().reset_index()
                value_counts.columns = [categorical_cols[0], "count"]
                fig = px.pie(
                    value_counts,
                    values="count",
                    names=categorical_cols[0],
                    title=f"Distribution of {categorical_cols[0]}",
                )
            elif len(numeric_cols) >= 1:
                fig = px.histogram(
                    df, x=numeric_cols[0], title=f"Distribution of {numeric_cols[0]}"
                )
            else:
                return None, "❌ Keine geeignete Visualisierung möglich"
        elif vis_type == "bar" and len(categorical_cols) >= 1:
            value_counts = df[categorical_cols[0]].value_counts().reset_index()
            value_counts.columns = [categorical_cols[0], "count"]
            fig = px.bar(
                value_counts,
                x=categorical_cols[0],
                y="count",
                title=f"Bar Chart: {categorical_cols[0]}",
            )
        elif vis_type == "pie" and len(categorical_cols) >= 1:
            value_counts = df[categorical_cols[0]].value_counts().reset_index()
            value_counts.columns = [categorical_cols[0], "count"]
            fig = px.pie(
                value_counts,
                values="count",
                names=categorical_cols[0],
                title=f"Pie Chart: {categorical_cols[0]}",
            )
        else:
            return None, "❌ Keine geeignete Visualisierung für diese Daten möglich"

        html_plot = pio.to_html(fig, include_plotlyjs="cdn", div_id="plotly-div")
        return html_plot, "✅ Visualisierung erstellt"

    except Exception as e:
        return None, f"❌ Fehler bei der Visualisierung: {str(e)}"


def generate_report_sync(query):
    """Generate analysis report"""
    return "📋 Report-Funktion steht zur Verfügung (LLM erforderlich)"


def load_database_configs():
    """Load saved database configurations"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_database_config(config_name, config_data):
    """Save a database configuration"""
    configs = load_database_configs()
    configs[config_name] = config_data

    with open(CONFIG_FILE, "w") as f:
        json.dump(configs, f, indent=2)

    return f"✅ Configuration '{config_name}' saved successfully"


def get_config_names():
    """Get list of saved configuration names"""
    configs = load_database_configs()
    return list(configs.keys())


def load_config_by_name(config_name):
    """Load specific configuration by name"""
    configs = load_database_configs()
    return configs.get(config_name, {})


def create_interface():
    """Create the main Gradio interface"""
    with gr.Blocks(title="Data Analyzer") as interface:
        gr.HTML("<h1><center>🔍 Data Analyzer</center></h1>")

        with gr.Tabs():
            # Tab 1: Database Connection with dropdown and save/load
            with gr.TabItem("🔧 Database Connection"):
                gr.HTML("<h3>Database Configuration</h3>")

                with gr.Row():
                    with gr.Column():
                        # Configuration management section
                        gr.HTML("<h4>📁 Configuration Management</h4>")

                        with gr.Row():
                            config_dropdown = gr.Dropdown(
                                choices=get_config_names(),
                                label="Saved Configurations",
                                interactive=True,
                            )
                            load_config_btn = gr.Button("📂 Load", size="sm")
                            refresh_configs_btn = gr.Button("🔄 Refresh", size="sm")

                        with gr.Row():
                            config_name_input = gr.Textbox(
                                label="Configuration Name",
                                placeholder="Enter name to save current settings",
                            )
                            save_config_btn = gr.Button("💾 Save Config", size="sm")

                        config_save_status = gr.HTML()

                        gr.HTML("<hr><h4>🗄️ Database Settings</h4>")

                        # Database type selection as dropdown
                        db_type = gr.Dropdown(
                            choices=[
                                ("MongoDB", "mongodb"),
                                ("MySQL", "mysql"),
                                ("MariaDB", "mariadb"),
                                ("PostgreSQL", "postgresql"),
                                ("SQLite", "sqlite"),
                            ],
                            label="Database Type",
                            value="mongodb",
                        )

                        # Connection settings that change based on database type
                        with gr.Group() as mongodb_settings:
                            gr.HTML("""
                            <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <h4>⚠️ MongoDB Setup Required</h4>
                                <ul>
                                    <li>Install MongoDB Community Server</li>
                                    <li>Start MongoDB service</li>
                                    <li>Create user 'dataanalyzer' with password 'dataanalyzer_pwd'</li>
                                    <li>Create database 'my_test_db' with collection 'users_test'</li>
                                </ul>
                            </div>
                            """)

                            mongo_host = gr.Textbox(label="Host", value="localhost")
                            mongo_port = gr.Textbox(label="Port", value="27017")
                            mongo_database = gr.Textbox(
                                label="Database", value="my_test_db"
                            )
                            mongo_username = gr.Textbox(
                                label="Username", value="dataanalyzer"
                            )
                            mongo_password = gr.Textbox(
                                label="Password",
                                value="dataanalyzer_pwd",
                                type="password",
                            )
                            mongo_auth_source = gr.Textbox(
                                label="Auth Source", value="test"
                            )
                            mongo_collection = gr.Textbox(
                                label="Collection", value="users_test"
                            )

                        with gr.Group(visible=False) as sql_settings:
                            gr.HTML("""
                            <div style='background-color: #d1ecf1; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <h4>🗄️ SQL Database Setup</h4>
                                <p>Configure your SQL database connection:</p>
                            </div>
                            """)

                            sql_host = gr.Textbox(label="Host", value="localhost")
                            sql_port = gr.Textbox(label="Port", value="3306")
                            sql_database = gr.Textbox(label="Database", value="test_db")
                            sql_username = gr.Textbox(label="Username", value="root")
                            sql_password = gr.Textbox(
                                label="Password", value="", type="password"
                            )
                            sql_table = gr.Textbox(label="Table", value="users")

                        with gr.Group(visible=False) as sqlite_settings:
                            gr.HTML("""
                            <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                                <h4>📁 SQLite Setup</h4>
                                <p>SQLite uses a local file - no server required!</p>
                            </div>
                            """)

                            sqlite_file = gr.Textbox(
                                label="Database File", value="test.db"
                            )
                            sqlite_table = gr.Textbox(label="Table", value="users")

                        # Test connection button
                        test_btn = gr.Button(
                            "✅ Test Database Connection", variant="primary", size="lg"
                        )
                        connection_result = gr.HTML()

                        def update_database_settings(db_type_choice):
                            """Show/hide settings based on database type"""
                            if db_type_choice == "mongodb":
                                return (
                                    gr.update(visible=True),  # mongodb_settings
                                    gr.update(visible=False),  # sql_settings
                                    gr.update(visible=False),  # sqlite_settings
                                    gr.update(value="27017"),  # port update
                                )
                            elif db_type_choice in ["mysql", "mariadb", "postgresql"]:
                                port_val = (
                                    "3306"
                                    if db_type_choice in ["mysql", "mariadb"]
                                    else "5432"
                                )
                                return (
                                    gr.update(visible=False),  # mongodb_settings
                                    gr.update(visible=True),  # sql_settings
                                    gr.update(visible=False),  # sqlite_settings
                                    gr.update(value=port_val),  # port update
                                )
                            elif db_type_choice == "sqlite":
                                return (
                                    gr.update(visible=False),  # mongodb_settings
                                    gr.update(visible=False),  # sql_settings
                                    gr.update(visible=True),  # sqlite_settings
                                    gr.update(value=""),  # port update
                                )

                        def load_selected_config(config_name):
                            """Load the selected configuration"""
                            if not config_name:
                                return [
                                    gr.update() for _ in range(15)
                                ]  # Return empty updates

                            config = load_config_by_name(config_name)
                            if not config:
                                return [gr.update() for _ in range(15)]

                            # Update all fields based on loaded config
                            db_type_val = config.get("db_type", "mongodb")

                            updates = [
                                gr.update(value=db_type_val),  # db_type
                                gr.update(
                                    value=config.get("host", "localhost")
                                ),  # mongo_host / sql_host
                                gr.update(
                                    value=config.get("port", "27017")
                                ),  # mongo_port / sql_port
                                gr.update(
                                    value=config.get("database", "my_test_db")
                                ),  # database
                                gr.update(
                                    value=config.get("username", "dataanalyzer")
                                ),  # username
                                gr.update(value=config.get("password", "")),  # password
                                gr.update(
                                    value=config.get("auth_source", "test")
                                ),  # mongo_auth_source
                                gr.update(
                                    value=config.get("collection", "users_test")
                                ),  # mongo_collection / sql_table
                                gr.update(
                                    value=config.get("database", "test.db")
                                ),  # sqlite_file
                            ]

                            return updates

                        def save_current_config(
                            config_name,
                            db_type_val,
                            m_host,
                            m_port,
                            m_db,
                            m_user,
                            m_pass,
                            m_auth,
                            m_coll,
                            s_host,
                            s_port,
                            s_db,
                            s_user,
                            s_pass,
                            s_table,
                            sqlite_file_path,
                            sqlite_table_name,
                        ):
                            """Save current configuration"""
                            if not config_name.strip():
                                return "❌ Please enter a configuration name"

                            if db_type_val == "mongodb":
                                config_data = {
                                    "db_type": db_type_val,
                                    "host": m_host,
                                    "port": m_port,
                                    "database": m_db,
                                    "username": m_user,
                                    "password": m_pass,
                                    "auth_source": m_auth,
                                    "collection": m_coll,
                                }
                            elif db_type_val in ["mysql", "mariadb", "postgresql"]:
                                config_data = {
                                    "db_type": db_type_val,
                                    "host": s_host,
                                    "port": s_port,
                                    "database": s_db,
                                    "username": s_user,
                                    "password": s_pass,
                                    "table": s_table,
                                }
                            elif db_type_val == "sqlite":
                                config_data = {
                                    "db_type": db_type_val,
                                    "database": sqlite_file_path,
                                    "table": sqlite_table_name,
                                }

                            return save_database_config(config_name, config_data)

                        def refresh_config_dropdown():
                            """Refresh the configuration dropdown"""
                            return gr.update(choices=get_config_names())

                        def test_database_connection(
                            db_type_choice,
                            # MongoDB params
                            m_host,
                            m_port,
                            m_db,
                            m_user,
                            m_pass,
                            m_auth,
                            m_coll,
                            # SQL params
                            s_host,
                            s_port,
                            s_db,
                            s_user,
                            s_pass,
                            s_table,
                            # SQLite params
                            sqlite_file_path,
                            sqlite_table_name,
                        ):
                            try:
                                global current_config

                                if db_type_choice == "mongodb":
                                    from pymongo import MongoClient

                                    conn_str = f"mongodb://{m_user}:{m_pass}@{m_host}:{m_port}/{m_db}?authSource={m_auth}"
                                    client = MongoClient(
                                        conn_str, serverSelectionTimeoutMS=5000
                                    )
                                    db = client[m_db]
                                    collection = db[m_coll]
                                    count = collection.count_documents({})
                                    client.close()

                                    current_config = {
                                        "db_type": "mongodb",
                                        "host": m_host,
                                        "port": m_port,
                                        "database": m_db,
                                        "username": m_user,
                                        "password": m_pass,
                                        "auth_source": m_auth,
                                        "collection": m_coll,
                                    }

                                    return f"""
                                    <div style='background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px;'>
                                        <h4>✅ MongoDB Connection Successful!</h4>
                                        <p>Found {count} documents in collection '{m_coll}'</p>
                                        <p><strong>You can now use the other tabs!</strong></p>
                                    </div>
                                    """

                                elif db_type_choice in [
                                    "mysql",
                                    "mariadb",
                                    "postgresql",
                                ]:
                                    # SQL connection test
                                    if db_type_choice in ["mysql", "mariadb"]:
                                        import mysql.connector

                                        conn = mysql.connector.connect(
                                            host=s_host,
                                            port=int(s_port),
                                            database=s_db,
                                            user=s_user,
                                            password=s_pass,
                                        )
                                    else:  # postgresql
                                        import psycopg2

                                        conn = psycopg2.connect(
                                            host=s_host,
                                            port=int(s_port),
                                            database=s_db,
                                            user=s_user,
                                            password=s_pass,
                                        )

                                    cursor = conn.cursor()
                                    cursor.execute(f"SELECT COUNT(*) FROM {s_table}")
                                    count = cursor.fetchone()[0]
                                    conn.close()

                                    current_config = {
                                        "db_type": db_type_choice,
                                        "host": s_host,
                                        "port": s_port,
                                        "database": s_db,
                                        "username": s_user,
                                        "password": s_pass,
                                        "table": s_table,
                                    }

                                    return f"""
                                    <div style='background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px;'>
                                        <h4>✅ {db_type_choice.upper()} Connection Successful!</h4>
                                        <p>Found {count} rows in table '{s_table}'</p>
                                        <p><strong>You can now use the other tabs!</strong></p>
                                    </div>
                                    """

                                elif db_type_choice == "sqlite":
                                    import sqlite3

                                    conn = sqlite3.connect(sqlite_file_path)
                                    cursor = conn.cursor()
                                    cursor.execute(
                                        f"SELECT COUNT(*) FROM {sqlite_table_name}"
                                    )
                                    count = cursor.fetchone()[0]
                                    conn.close()

                                    current_config = {
                                        "db_type": "sqlite",
                                        "database": sqlite_file_path,
                                        "table": sqlite_table_name,
                                    }

                                    return f"""
                                    <div style='background-color: #d4edda; color: #155724; padding: 15px; border-radius: 8px;'>
                                        <h4>✅ SQLite Connection Successful!</h4>
                                        <p>Found {count} rows in table '{sqlite_table_name}'</p>
                                        <p><strong>You can now use the other tabs!</strong></p>
                                    </div>
                                    """

                            except Exception as e:
                                return f"""
                                <div style='background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 8px;'>
                                    <h4>❌ Connection Failed</h4>
                                    <p>Error: {str(e)}</p>
                                    <p><strong>Please check your {db_type_choice} configuration!</strong></p>
                                </div>
                                """

                        # Event handlers
                        db_type.change(
                            update_database_settings,
                            inputs=[db_type],
                            outputs=[
                                mongodb_settings,
                                sql_settings,
                                sqlite_settings,
                                sql_port,
                            ],
                        )

                        load_config_btn.click(
                            load_selected_config,
                            inputs=[config_dropdown],
                            outputs=[
                                db_type,
                                mongo_host,
                                mongo_port,
                                mongo_database,
                                mongo_username,
                                mongo_password,
                                mongo_auth_source,
                                mongo_collection,
                                sqlite_file,
                            ],
                        )

                        refresh_configs_btn.click(
                            refresh_config_dropdown, outputs=[config_dropdown]
                        )

                        save_config_btn.click(
                            save_current_config,
                            inputs=[
                                config_name_input,
                                db_type,
                                mongo_host,
                                mongo_port,
                                mongo_database,
                                mongo_username,
                                mongo_password,
                                mongo_auth_source,
                                mongo_collection,
                                sql_host,
                                sql_port,
                                sql_database,
                                sql_username,
                                sql_password,
                                sql_table,
                                sqlite_file,
                                sqlite_table,
                            ],
                            outputs=[config_save_status],
                        )

                        test_btn.click(
                            test_database_connection,
                            inputs=[
                                db_type,
                                mongo_host,
                                mongo_port,
                                mongo_database,
                                mongo_username,
                                mongo_password,
                                mongo_auth_source,
                                mongo_collection,
                                sql_host,
                                sql_port,
                                sql_database,
                                sql_username,
                                sql_password,
                                sql_table,
                                sqlite_file,
                                sqlite_table,
                            ],
                            outputs=[connection_result],
                        )

            # Tab 2: Query Data
            with gr.TabItem("📊 Query Data"):
                gr.HTML(
                    """
                <div style='background-color: #d1ecf1; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4>🔍 Data Queries</h4>
                    <p>Query your database (test connection first)</p>
                </div>
                """
                )

                query_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter a query or just leave default to get all data",
                    lines=2,
                    value="Show all data",
                )

                with gr.Row():
                    execute_btn = gr.Button(
                        "🔍 Run Query", variant="primary", size="lg"
                    )
                    download_csv_btn = gr.Button("📥 Download CSV")

                query_status = gr.HTML()
                data_display = gr.Dataframe(label="Results")
                csv_download = gr.File(label="CSV Download", visible=False)

            # Tab 3: Charts
            with gr.TabItem("📊 Charts"):
                gr.HTML(
                    """
                <div style='background-color: #d4edda; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4>📈 Create Charts</h4>
                    <p>Generate charts from your query results</p>
                </div>
                """
                )

                with gr.Row():
                    vis_type = gr.Dropdown(
                        choices=["auto", "bar", "pie", "scatter"],
                        label="Chart Type",
                        value="auto",
                    )
                    generate_vis_btn = gr.Button("📊 Create Chart", variant="primary")

                vis_status = gr.HTML()
                visualization_display = gr.HTML(label="Chart")

            # Tab 4: Report
            with gr.TabItem("📄 Report"):
                gr.HTML(
                    """
                <div style='background-color: #fff3cd; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
                    <h4>📋 AI Analysis Report</h4>
                    <p>Generate reports (requires LLM setup)</p>
                </div>
                """
                )

                generate_report_btn = gr.Button(
                    "🤖 Generate AI Report", variant="primary", size="lg"
                )
                report_display = gr.Markdown(label="Analysis Report")

        # Event handlers
        execute_btn.click(
            execute_query_sync,
            [query_input, gr.State("users_test")],
            [data_display, query_status],
        )

        download_csv_btn.click(download_csv, outputs=csv_download)

        generate_vis_btn.click(
            generate_visualization, vis_type, [visualization_display, vis_status]
        )

        generate_report_btn.click(generate_report_sync, query_input, report_display)

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(server_name="0.0.0.0", server_port=7860)
