"""
Erforderliche Installationen:
pip install pymongo openai
"""

import logging
import time
from typing import Any, List, Dict, Optional, Union
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure, ConfigurationError
from bson import ObjectId  # Für potenzielle Typendarstellung und Konvertierung
from pprint import pprint  # Für das schöne Drucken von Strukturen

# Für OpenAI-kompatible API
try:
    import openai
except ImportError:
    openai = None
    # print("OpenAI-Bibliothek nicht gefunden. NL zu MongoDB-Abfragegenerierung wird nicht verfügbar sein.")

logger = logging.getLogger("mcp_server.services.mongodb_connector")


class MongoDBConnector:
    """
    Stellt Operationen für MongoDB bereit, einschließlich Auflisten von Datenbanken,
    Collections, Collection-Strukturen und Generieren von MongoDB-Abfragen
    aus natürlicher Sprache.
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        openai_api_key: Optional[str] = None,
        openai_api_base: Optional[str] = None,
        llm_model_name: str = "qwen2.5",  # Or another suitable model
        auth_source: Optional[str] = "admin",  # Allow specifying authSource
    ):
        """Initialize MongoDB connector with improved authentication handling."""
        self.connection_string = connection_string
        self.client: Optional[MongoClient] = None
        self.openai_api_key = openai_api_key
        self.openai_api_base = openai_api_base
        self.llm_model_name = llm_model_name
        self.auth_source = auth_source  # Store the authSource

        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"Initializing MongoDBConnector with connection string: {connection_string}, authSource: {auth_source}"
        )

        try:
            # Parse connection string to extract authentication details
            from urllib.parse import urlparse

            parsed_uri = urlparse(connection_string)

            # If no auth in connection string, try without authentication
            if not parsed_uri.username and not parsed_uri.password:
                self.client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                )
            else:
                # Try with provided authentication and authSource
                self.client = MongoClient(
                    connection_string,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    authSource=self.auth_source,  # Use the specified authSource
                )

            # Test connection
            self.client.admin.command("ping")
            logger.info(
                f"MongoDB client successfully initialized for: {self._mask_uri_credentials(connection_string)}"
            )
        except ConnectionFailure as e:
            logger.error(
                f"MongoDB connection error for '{self._mask_uri_credentials(connection_string)}': {e}"
            )
            self.client = None
            raise
        except ConfigurationError as e:
            logger.error(
                f"MongoDB configuration error for '{self._mask_uri_credentials(connection_string)}': {e}"
            )
            self.client = None
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error during MongoDB client initialization for '{self._mask_uri_credentials(connection_string)}': {e}"
            )
            self.client = None
            raise

    def _mask_uri_credentials(self, uri: str) -> str:
        """Maskiert Anmeldeinformationen in einer MongoDB-URI für das Logging."""
        try:
            from urllib.parse import urlparse, urlunparse

            parsed = urlparse(uri)
            if (parsed.username or parsed.password) and parsed.hostname:
                masked_netloc = f"****:****@{parsed.hostname}"
                if parsed.port:
                    masked_netloc += f":{parsed.port}"
                parsed = parsed._replace(netloc=masked_netloc)
                return urlunparse(parsed)
            return uri
        except Exception:  # Fallback, falls das Parsen fehlschlägt
            return "mongodb://<credentials_masked>@<host>"

    def close_connection(self):
        """Schließt die MongoDB-Verbindung."""
        if self.client:
            self.client.close()
            logger.info("MongoDB-Verbindung geschlossen.")

    def list_databases(self) -> List[str]:
        """
        Listet alle Datenbanken auf, auf die der verbundene Benutzer Zugriff hat.
        """
        if not self.client:
            logger.error("MongoDB-Client nicht initialisiert.")
            raise RuntimeError("MongoDB-Client nicht initialisiert.")
        try:
            db_names = self.client.list_database_names()
            logger.debug(f"Verfügbare Datenbanken: {db_names}")
            return db_names
        except OperationFailure as e:
            logger.error(f"Fehler beim Auflisten der Datenbanken: {e}")
            if e.code == 13:  # Unauthorized
                logger.warning(
                    "Benutzer hat möglicherweise keine Berechtigung, alle Datenbanken aufzulisten."
                )
            raise
        except Exception as e:
            logger.error(f"Unerwarteter Fehler beim Auflisten der Datenbanken: {e}")
            raise

    def list_collections(self, db_name: str) -> List[str]:
        """
        List all collections in a specified database.
        """
        if not self.client:
            logger.error("MongoDB client is not initialized.")
            raise RuntimeError("MongoDB client is not initialized.")
        if not db_name:
            raise ValueError("Database name cannot be empty.")

        try:
            # Ensure the database exists before listing collections
            if db_name not in self.list_databases():
                raise ValueError(f"Database '{db_name}' does not exist.")

            db = self.client[db_name]
            collection_names = db.list_collection_names()
            logger.debug(f"Collections in database '{db_name}': {collection_names}")
            return collection_names
        except OperationFailure as e:
            logger.error(f"Error listing collections for database '{db_name}': {e}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error while listing collections for database '{db_name}': {e}"
            )
            raise

    def get_collection_structure(
        self, db_name: str, collection_name: str, sample_size: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Ermittelt die Struktur einer Collection durch Inspektion von Beispieldokumenten.
        Gibt eine Liste von Felddefinitionen zurück, wobei jede Definition
        den Feldnamen und seinen abgeleiteten Typ vom ersten angetroffenen Nicht-Null-Wert enthält.

        Args:
            db_name: Der Name der Datenbank.
            collection_name: Der Name der Collection.
            sample_size: Anzahl der zu untersuchenden Dokumente zur Ableitung der Struktur.
                         Eine größere Zahl gibt eine umfassendere Sicht, ist aber langsamer.

        Returns:
            Eine Liste von Dictionaries, wobei jedes Dictionary ein Feld repräsentiert
            und 'name' sowie 'type' (als String) enthält.
            Gibt eine leere Liste zurück, wenn die Collection leer ist oder ein Fehler auftritt.
        """
        if not self.client:
            logger.error("MongoDB-Client nicht initialisiert.")
            raise RuntimeError("MongoDB-Client nicht initialisiert.")
        if not db_name or not collection_name:
            raise ValueError("Datenbank- und Collection-Namen dürfen nicht leer sein.")

        logger.debug(
            f"Ermittle Struktur für Collection '{collection_name}' in Datenbank '{db_name}' mit sample_size={sample_size}"
        )
        # field_name: type_string. Verwendet, um Typen zu sammeln.
        # Nimmt den Typ des ersten Nicht-Null-Wertes, der für ein Feld gefunden wird.
        structure_map: Dict[str, str] = {}
        field_order: List[
            str
        ] = []  # Um eine gewisse Reihenfolge der Felder beizubehalten

        try:
            db = self.client[db_name]
            collection = db[collection_name]

            documents_to_inspect = list(collection.find().limit(sample_size))

            if not documents_to_inspect:
                logger.info(
                    f"Collection '{db_name}.{collection_name}' ist leer oder es wurden keine Dokumente für das Sampling gefunden."
                )
                return []

            for doc in documents_to_inspect:
                if not doc:
                    continue
                for key, value in doc.items():
                    if key not in structure_map:  # Neues Feld hinzufügen
                        field_order.append(key)
                        structure_map[key] = type(value).__name__
                    # Wenn der bisherige Typ 'NoneType' war und ein neuer Wert nicht None ist, aktualisiere den Typ
                    elif structure_map[key] == "NoneType" and value is not None:
                        structure_map[key] = type(value).__name__

            # Formatieren für die Ausgabe
            schema_representation = []
            for field_name in field_order:
                schema_representation.append(
                    {"name": field_name, "type": structure_map[field_name]}
                )

            logger.debug(
                f"Abgeleitete Struktur für '{db_name}.{collection_name}': {schema_representation}"
            )
            return schema_representation

        except OperationFailure as e:
            logger.error(
                f"Operationsfehler beim Ermitteln der Struktur für '{db_name}.{collection_name}': {e}"
            )
            return []
        except Exception as e:
            logger.error(
                f"Unerwarteter Fehler beim Ermitteln der Struktur für '{db_name}.{collection_name}': {e}"
            )
            raise

    def _get_db_schema_for_llm(
        self,
        target_db_name: Optional[str] = None,
        max_collections_per_db: int = 10,
        sample_docs_for_struct: int = 1,
    ) -> str:
        """
        Erstellt eine String-Repräsentation des Datenbankschemas für das LLM.
        Wenn target_db_name angegeben ist, wird nur das Schema dieser DB abgerufen.
        Andernfalls wird versucht, das Schema für einige Datenbanken abzurufen.
        """
        if not self.client:
            raise RuntimeError("MongoDB-Client nicht initialisiert.")

        schema_parts = []
        databases_to_inspect = []

        if target_db_name:
            databases_to_inspect.append(target_db_name)
        else:
            try:
                all_dbs = self.list_databases()
                system_dbs = [
                    "admin",
                    "local",
                    "config",
                ]  # Übliche System-DBs ausschließen
                databases_to_inspect = [db for db in all_dbs if db not in system_dbs][
                    :3
                ]  # Max. 3 DBs für den Überblick
                if (
                    not databases_to_inspect and all_dbs
                ):  # Falls nur System-DBs vorhanden sind, eine auswählen
                    databases_to_inspect = [all_dbs[0]]
            except Exception as e:
                logger.warning(
                    f"Konnte nicht alle Datenbanken für LLM-Schema auflisten: {e}."
                )
                return "Fehler: Konnte Datenbankliste für Schemaerstellung nicht ermitteln."

        if not databases_to_inspect:
            return "Keine Datenbanken zur Inspektion für das LLM-Schema gefunden."

        for db_name in databases_to_inspect:
            schema_parts.append(f"Datenbank: {db_name}")
            try:
                collections = self.list_collections(db_name)
                for i, coll_name in enumerate(collections):
                    if i >= max_collections_per_db:
                        schema_parts.append("  ... (und weitere Collections)")
                        break
                    schema_parts.append(f"  Collection: {coll_name}")
                    try:
                        coll_struct = self.get_collection_structure(
                            db_name, coll_name, sample_size=sample_docs_for_struct
                        )
                        if coll_struct:
                            for field_info in coll_struct:
                                schema_parts.append(
                                    f"    - {field_info['name']} (Typ: {field_info['type']})"
                                )
                        else:
                            schema_parts.append(
                                "    (Collection ist leer oder Struktur konnte nicht ermittelt werden)"
                            )
                    except Exception as e_coll:
                        schema_parts.append(
                            f"    (Fehler beim Inspizieren der Collection {coll_name}: {e_coll})"
                        )
            except Exception as e_db:
                schema_parts.append(
                    f"  (Fehler beim Auflisten der Collections für DB {db_name}: {e_db})"
                )
            schema_parts.append("")  # Leerzeile zwischen Datenbanken

        if not schema_parts:
            return "Es konnten keine Datenbankschema-Informationen abgerufen werden."
        return "\n".join(schema_parts)

    def generate_mongodb_query_from_natural_language(
        self, natural_language_input: str, target_db_name: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Generiert eine MongoDB-Abfrage (als Python-Dictionary für pymongo)
        aus einer natürlichsprachlichen Eingabe.

        Args:
            natural_language_input: Die Benutzeranfrage in natürlicher Sprache.
            target_db_name: Optional. Die spezifische Datenbank, auf die sich die Abfrage bezieht.
                            Wenn None, wird das Schema aus mehreren DBs bereitgestellt.

        Returns:
            Ein Python-Dictionary, das den MongoDB-Filter für `find()` darstellt,
            oder eine Liste von Dictionaries für eine Aggregationspipeline.

        Raises:
            ImportError: Wenn die 'openai'-Bibliothek nicht installiert ist.
            ValueError: Wenn API-Schlüssel/Basis-URL nicht konfiguriert sind.
            RuntimeError: Bei API-Fehlern oder anderen Problemen während der Generierung.
        """
        logger.info(
            f'Versuche, MongoDB-Abfrage zu generieren aus: "{natural_language_input}" für DB: {target_db_name or "beliebig"}'
        )
        if openai is None:
            raise ImportError(
                "Die 'openai'-Bibliothek wird für die Generierung von natürlichsprachlichen MongoDB-Abfragen benötigt. "
                "Bitte installieren Sie sie: pip install openai"
            )

        if not self.openai_api_key or not self.openai_api_base:
            raise ValueError(
                "OpenAI API-Schlüssel und Basis-URL müssen für die Abfragegenerierung konfiguriert sein."
            )

        try:
            schema_representation_str = self._get_db_schema_for_llm(
                target_db_name=target_db_name
            )
            if (
                "Fehler:" in schema_representation_str
                or "Keine Datenbank" in schema_representation_str
            ):
                logger.error(
                    f"Konnte Datenbankschema für LLM nicht abrufen. NL-zu-MongoDB-Generierung kann nicht fortgesetzt werden. Schema-Info: {schema_representation_str}"
                )
                raise ValueError(
                    f"Datenbankschema konnte nicht abgerufen werden: {schema_representation_str}"
                )
            logger.debug(f"Schema für NL-zu-MongoDB: \n{schema_representation_str}")
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Schemas für NL-zu-MongoDB: {e}")
            raise RuntimeError(
                "Fehler beim Abrufen des Schemas, das für die NL-zu-MongoDB-Generierung erforderlich ist."
            ) from e

        system_prompt = f"""Sie sind ein Experte für die Generierung von MongoDB-Abfragen.
Gegeben ist das folgende MongoDB-Datenbankschema (Datenbanken, Collections und Beispielfelder mit Typen) und eine Benutzerfrage in natürlicher Sprache. Generieren Sie eine syntaktisch korrekte MongoDB-Abfrage.
Die Abfrage sollte ein Python-Dictionary sein, das für die Verwendung mit der `find()`-Methode von PyMongo (als Filterargument) geeignet ist, oder eine Aggregationspipeline (Liste von Dictionaries), wenn die Abfrage komplex ist.
Wenn Sie eine Aggregationspipeline generieren, sollte es eine Liste von Stufen sein.
Wenn Sie einen `find`-Filter generieren, sollte es ein einzelnes Dictionary sein.
Geben Sie NUR das Python-Dictionary oder die Liste von Dictionaries für die Abfrage aus. Fügen Sie keine Erklärungen, Variablenzuweisungen (z.B. `query = ...`) oder Markdown-Backticks um den Code ein.

Datenbankschema:
{schema_representation_str}

Beispiel `find`-Abfrageformat: {{"feld_name": "wert", "numerisches_feld": {{"$gt": 10}}}}
Beispiel Aggregationspipeline-Format: [{{"$match": {{"status": "A"}}}}, {{"$group": {{"_id": "$cust_id", "total": {{"$sum": "$amount"}}}}}}]
"""
        user_prompt_content = f"Benutzerfrage: {natural_language_input}\nMongoDB-Abfrage (Python dict/list):"

        max_retries = 3  # Number of retries for OpenAI API
        retry_delay = 2  # Initial delay in seconds

        for attempt in range(max_retries):
            try:
                client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_api_base,
                )
                logger.info(
                    f"Sende Anfrage an LLM ({self.llm_model_name}) unter {self.openai_api_base} für MongoDB-Abfrage"
                )

                completion = client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt_content},
                    ],
                    temperature=0.1,  # Niedrigere Temperatur für deterministischere Abfragen
                )

                generated_query_str = completion.choices[0].message.content.strip()
                logger.info(
                    f"LLM ({self.llm_model_name}) Rohantwort: {generated_query_str}"
                )

                import ast

                try:
                    # Bereinige häufige LLM-Artefakte
                    for prefix in ["```python", "```json", "```"]:
                        if generated_query_str.startswith(prefix):
                            generated_query_str = generated_query_str[
                                len(prefix) :
                            ].strip()
                    if generated_query_str.endswith("```"):
                        generated_query_str = generated_query_str[: -len("```")].strip()

                    parsed_query = ast.literal_eval(generated_query_str)
                    if not isinstance(parsed_query, (dict, list)):
                        raise ValueError(
                            "LLM hat kein gültiges Dictionary oder keine Liste für die Abfrage zurückgegeben."
                        )

                    logger.info(
                        f"LLM ({self.llm_model_name}) generierte MongoDB-Abfrage (geparst): {parsed_query}"
                    )
                    return parsed_query
                except (SyntaxError, ValueError) as e_parse:
                    logger.error(
                        f"Fehler beim Parsen der LLM-Antwort in eine MongoDB-Abfrage: {generated_query_str}. Fehler: {e_parse}"
                    )
                    raise RuntimeError(
                        f"LLM generierte ein ungültiges Abfrageformat: {generated_query_str}"
                    ) from e_parse

            except openai.APIError as e:
                logger.error(
                    f"OpenAI API-Fehler während der MongoDB-Abfragegenerierung: {e}"
                )
                if attempt < max_retries - 1:
                    logger.info(
                        f"Retrying OpenAI API request ({attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise RuntimeError(
                        f"Fehler bei der Generierung der MongoDB-Abfrage aufgrund eines API-Fehlers: {e}"
                    ) from e
            except Exception as e:
                logger.error(
                    f"Unerwarteter Fehler während der MongoDB-Abfragegenerierung mit LLM: {e}"
                )
                raise RuntimeError(
                    f"Ein unerwarteter Fehler trat während der MongoDB-Abfragegenerierung auf: {e}"
                ) from e

    def execute_raw_query(
        self,
        db_name: str,
        collection_name: str,
        query_type: str,
        query_details: Union[Dict, List],
        find_options: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Führt eine rohe MongoDB-Abfrage aus (find oder aggregate).

        Args:
            db_name: Der Name der Datenbank.
            collection_name: Der Name der Collection.
            query_type: "find" oder "aggregate".
            query_details: Der Abfragefilter (für find) oder die Pipeline (für aggregate).
            find_options: Optionales Dictionary für find-Operationen (z.B. Projektion, Sortierung, Limit).

        Returns:
            Eine Liste von Dokumenten.
        """
        if not self.client:
            raise RuntimeError("MongoDB-Client nicht initialisiert.")

        db = self.client[db_name]
        collection = db[collection_name]
        results = []

        logger.info(
            f"Führe {query_type} auf {db_name}.{collection_name} aus mit Details: {query_details}"
        )

        try:
            if query_type == "find":
                cursor = collection.find(query_details, **(find_options or {}))
                results = list(cursor)
            elif query_type == "aggregate":
                cursor = collection.aggregate(query_details)  # type: ignore
                results = list(cursor)
            else:
                raise ValueError(
                    f"Nicht unterstützter query_type: {query_type}. Muss 'find' oder 'aggregate' sein."
                )

            logger.debug(f"Abfrageausführung lieferte {len(results)} Dokumente.")

            # Konvertiere ObjectIds in Strings für einfachere Handhabung (z.B. JSON-Serialisierung)
            def convert_objectids_in_doc(doc):
                if isinstance(doc, list):
                    return [convert_objectids_in_doc(item) for item in doc]
                if isinstance(doc, dict):
                    return {
                        key: convert_objectids_in_doc(value)
                        for key, value in doc.items()
                    }
                if isinstance(doc, ObjectId):
                    return str(doc)
                return doc

            return [convert_objectids_in_doc(doc.copy()) for doc in results]

        except OperationFailure as e:
            logger.error(
                f"MongoDB-Operationsfehler während der Rohabfrageausführung: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Unerwarteter Fehler während der Rohabfrageausführung: {e}")
            raise

    def get_collection(self, db_name: str, collection_name: str):
        """
        Retrieve a collection from the specified database.
        """
        if not self.client:
            raise RuntimeError("MongoDB client is not initialized.")
        db = self.client[db_name]
        return db[collection_name]


if __name__ == "__main__":
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # --- Konfiguration ---
    MONGO_CONNECTION_STRING = os.getenv(
        "MONGO_CONNECTION_STRING",
        "mongodb://localhost:27017/",  # Default to unauthenticated local connection
    )
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "ollama")
    OPENAI_API_BASE = os.getenv(
        "OPENAI_API_BASE_URL", "http://localhost:11434/v1"
    )  # Beispiel für lokales Ollama
    LLM_MODEL = os.getenv(
        "LLM_MODEL", "qwen2.5-coder:latest"
    )  # Stellen Sie sicher, dass dieses Modell in Ollama verfügbar ist

    mongo_connector = None  # Vorinitialisieren für finally-Block
    try:
        mongo_connector = MongoDBConnector(
            connection_string=MONGO_CONNECTION_STRING,
            openai_api_key=OPENAI_API_KEY,
            openai_api_base=OPENAI_API_BASE,
            llm_model_name=LLM_MODEL,
        )
    except Exception as e:
        logger.error(f"Fehler bei der Initialisierung des MongoDBConnectors: {e}")
        exit(1)

    # Create database and user if needed
    async def setup_test_db(connector):
        if connector.client:
            try:
                admin_db = connector.client.admin
                test_db = connector.client["my_test_db"]

                # Create user if it doesn't exist
                try:
                    admin_db.command(
                        {
                            "createUser": "dataanalyzer",
                            "pwd": "dataanalyzer_pwd",
                            "roles": [{"role": "readWrite", "db": "my_test_db"}],
                        }
                    )
                    logger.info("Test user created successfully")
                except Exception as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Could not create test user: {e}")

                return test_db
            except Exception as e:
                logger.error(f"Error setting up test database: {e}")
                return None

    # --- Basisoperationen ---
    try:
        print("\n--- Auflisten der Datenbanken ---")
        databases = mongo_connector.list_databases()
        pprint(databases)

        test_db_name = "my_test_db"  # Eindeutiger Name für Tests
        test_coll_name = "users_test"

        # Testdaten vorbereiten (erstellen/auffüllen, falls nicht vorhanden)
        logger.info(
            f"Vorbereiten der Testdatenbank '{test_db_name}' und Collection '{test_coll_name}'..."
        )
        if mongo_connector.client:
            db = mongo_connector.client[test_db_name]
            collection = db[test_coll_name]
            if collection.count_documents({}) == 0:
                logger.info(
                    f"Fülle '{test_db_name}.{test_coll_name}' mit Beispieldaten..."
                )
                collection.insert_many(
                    [
                        {
                            "name": "Alice",
                            "age": 30,
                            "city": "New York",
                            "hobbies": ["lesen", "wandern"],
                            "status": "aktiv",
                        },
                        {
                            "name": "Bob",
                            "age": 24,
                            "city": "London",
                            "hobbies": ["coden", "musik"],
                            "occupation": "Ingenieur",
                            "status": "aktiv",
                        },
                        {
                            "name": "Charlie",
                            "age": 35,
                            "city": "Paris",
                            "occupation": "Künstler",
                            "status": "inaktiv",
                        },
                        {
                            "name": "Diana",
                            "age": 28,
                            "city": "New York",
                            "hobbies": ["gaming"],
                            "occupation": "Ingenieur",
                            "status": "aktiv",
                        },
                    ]
                )
                logger.info("Beispieldaten eingefügt.")
            else:
                logger.info(
                    f"Collection '{test_db_name}.{test_coll_name}' enthält bereits Daten."
                )

        print(f"\n--- Auflisten der Collections in '{test_db_name}' ---")
        collections = mongo_connector.list_collections(test_db_name)
        pprint(collections)

        if test_coll_name in collections:
            print(
                f"\n--- Struktur der Collection '{test_db_name}.{test_coll_name}' (sample_size=2) ---"
            )
            structure = mongo_connector.get_collection_structure(
                test_db_name, test_coll_name, sample_size=2
            )
            pprint(structure)
        else:
            print(
                f"Test-Collection '{test_coll_name}' nicht in '{test_db_name}' gefunden, um Struktur zu inspizieren."
            )

    except RuntimeError as e:
        logger.error(f"Laufzeitfehler während der MongoDB-Basisoperationen: {e}")
    except ValueError as e:
        logger.error(f"Wertfehler während der MongoDB-Basisoperationen: {e}")
    except OperationFailure as e:
        logger.error(
            f"MongoDB-Operationsfehler: {e}. Überprüfen Sie Berechtigungen und Serverstatus."
        )
    except Exception as e:
        logger.error(f"Ein unerwarteter Fehler ist aufgetreten: {e}", exc_info=True)

    # --- Natürlichsprachliche Abfragen zu MongoDB ---
    if (
        mongo_connector
        and mongo_connector.openai_api_key
        and mongo_connector.openai_api_base
        and openai
    ):
        print("\n--- Natürlichsprachliche Abfragen zu MongoDB ---")

        nl_queries_for_test_db = [
            ("Finde alle Benutzer älter als 30", test_db_name, test_coll_name),
            (
                "Zeige mir Benutzer aus New York, die gerne wandern",
                test_db_name,
                test_coll_name,
            ),
            ("Liste Namen und Berufe aller Ingenieure", test_db_name, test_coll_name),
            (
                "Wie viele aktive Benutzer gibt es in jeder Stadt?",
                test_db_name,
                test_coll_name,
            ),  # Erfordert Aggregation
        ]

        for nl_query, db_name_for_query, coll_name_for_exec in nl_queries_for_test_db:
            print(
                f"\nNatürlichsprachliche Anfrage: {nl_query} (für DB: {db_name_for_query})"
            )
            try:
                mongo_db_query = (
                    mongo_connector.generate_mongodb_query_from_natural_language(
                        nl_query, target_db_name=db_name_for_query
                    )
                )
                print("Generierte MongoDB-Abfrage (Python dict/list):")
                pprint(mongo_db_query)

                if mongo_db_query:
                    query_type_to_exec = (
                        "aggregate" if isinstance(mongo_db_query, list) else "find"
                    )

                    print(
                        f"Führe generierte Abfrage ({query_type_to_exec}) auf '{db_name_for_query}.{coll_name_for_exec}' aus..."
                    )
                    results = mongo_connector.execute_raw_query(
                        db_name=db_name_for_query,
                        collection_name=coll_name_for_exec,
                        query_type=query_type_to_exec,
                        query_details=mongo_db_query,
                    )
                    print(f"Ergebnisse ({len(results)} Dokumente):")
                    pprint(results[:5])  # Zeige die ersten 5 Ergebnisse
                    if len(results) > 5:
                        print("...")
            except (ImportError, ValueError, RuntimeError) as e:
                logger.error(
                    f"Fehler bei Generierung/Ausführung der MongoDB-Abfrage für '{nl_query}': {e}"
                )
            except Exception as e:
                logger.error(
                    f"Unerwarteter Fehler für '{nl_query}': {e}", exc_info=True
                )
    else:
        print(
            "\nÜberspringe NL-zu-MongoDB-Abfragetest: OpenAI-Bibliothek nicht installiert oder API-Schlüssel/Basis-URL nicht konfiguriert."
        )

    if mongo_connector:
        try:
            if (
                mongo_connector.client
                and test_db_name in mongo_connector.list_databases()
            ):
                mongo_connector.client.drop_database(test_db_name)
                logger.info(f"Testdatenbank '{test_db_name}' erfolgreich gelöscht.")
        except Exception as e:
            logger.warning(
                f"Konnte Testdatenbank '{test_db_name}' nicht löschen oder sie existierte nicht: {e}"
            )

    if mongo_connector:
        mongo_connector.close_connection()
