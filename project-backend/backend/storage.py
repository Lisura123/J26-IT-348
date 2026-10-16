"""Storage configuration and MongoDB client factory for project-backend.
Handles Atlas connection strings, role-based accounts (writer vs read-only reader),
timeouts, and credential sanitization.
Never logs or exposes credentials.
"""
import logging
import os
import re
from typing import Any, Dict, Optional
import urllib.parse
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.database import Database

load_dotenv()

logger = logging.getLogger("backend.storage")

DEFAULT_DATABASE = "solar_pv"
DEFAULT_SERVER_SELECTION_TIMEOUT_MS = 10000
DEFAULT_CONNECT_TIMEOUT_MS = 10000
DEFAULT_SOCKET_TIMEOUT_MS = 60000
DEFAULT_RETRY_WRITES = True


def sanitize_uri(uri: str) -> str:
    """Masks credentials in MongoDB connection URI.
    Example: mongodb+srv://user:pass@cluster.mongodb.net -> mongodb+srv://***:***@cluster.mongodb.net
    """
    if not uri:
        return ""
    try:
        p = urllib.parse.urlsplit(uri)
        if "@" in p.netloc:
            host_part = p.netloc.split("@")[-1]
            masked_netloc = f"***:***@{host_part}"
            return urllib.parse.urlunsplit((p.scheme, masked_netloc, p.path, p.query, p.fragment))
        return uri
    except Exception:
        return re.sub(r"://([^:]+):([^@]+)@", r"://***:***@", uri)


def sanitize_error(err: Any) -> str:
    """Strips secrets and sensitive connection credentials from error messages and exception text."""
    if err is None:
        return ""
    msg = str(err)

    # Mask standard URI credential patterns
    msg = re.sub(r"://([^:]+):([^@]+)@", r"://***:***@", msg)

    # Mask known environment secret values
    secret_keys = [
        "MONGO_ROOT_PASSWORD",
        "MONGO_APP_PASSWORD",
        "MONGO_READER_PASSWORD",
        "MONGO_URI",
        "API_KEY",
        "READ_API_KEY",
        "INFLUX_TOKEN",
    ]
    for key in secret_keys:
        val = os.environ.get(key)
        if val and len(val.strip()) >= 4:
            msg = msg.replace(val.strip(), "***")

    return msg


def get_database_name() -> str:
    """Returns the effective MongoDB database name from MONGO_DB or default 'solar_pv'."""
    db_name = os.environ.get("MONGO_DB", "").strip()
    return db_name if db_name else DEFAULT_DATABASE


def build_role_uri(
    base_uri: str,
    role: str = "app",
    username: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Injects or overrides role-specific credentials in base_uri if configured.
    For mongodb+srv (or standard mongodb) URIs, replaces ONLY the user and password
    components of the URI netloc, preserving hostname, cluster address, port, path,
    and query parameters (such as retryWrites=true&w=majority).

    Roles:
      - 'app': writer account (MONGO_APP_USER / MONGO_APP_PASSWORD)
      - 'reader': read-only account (MONGO_READER_USER / MONGO_READER_PASSWORD)
      - 'root': administrative account (MONGO_ROOT_USER / MONGO_ROOT_PASSWORD)
    If role-specific user/password are not set in environment or arguments, returns base_uri untouched.
    Never prints or logs the raw credentials.
    """
    if not base_uri:
        return ""

    user: Optional[str] = username
    passwd: Optional[str] = password

    if user is None and passwd is None:
        if role == "reader":
            user = os.environ.get("MONGO_READER_USER", "").strip() or None
            passwd = os.environ.get("MONGO_READER_PASSWORD", "").strip() or None
        elif role == "root":
            user = os.environ.get("MONGO_ROOT_USER", "").strip() or None
            passwd = os.environ.get("MONGO_ROOT_PASSWORD", "").strip() or None
        else:  # default 'app'
            user = os.environ.get("MONGO_APP_USER", "").strip() or None
            passwd = os.environ.get("MONGO_APP_PASSWORD", "").strip() or None

    if not user and not passwd:
        return base_uri

    try:
        p = urllib.parse.urlsplit(base_uri)
        host_part = p.netloc.split("@")[-1]
        user_enc = urllib.parse.quote_plus(user) if user else ""
        pass_enc = urllib.parse.quote_plus(passwd) if passwd else ""
        cred_part = f"{user_enc}:{pass_enc}" if (user_enc or pass_enc) else ""
        new_netloc = f"{cred_part}@{host_part}" if cred_part else host_part
        return urllib.parse.urlunsplit((p.scheme, new_netloc, p.path, p.query, p.fragment))
    except Exception:
        return base_uri


def get_mongo_client(
    role: str = "app",
    uri: Optional[str] = None,
    server_selection_timeout_ms: int = DEFAULT_SERVER_SELECTION_TIMEOUT_MS,
    connect_timeout_ms: int = DEFAULT_CONNECT_TIMEOUT_MS,
    socket_timeout_ms: int = DEFAULT_SOCKET_TIMEOUT_MS,
    retry_writes: bool = DEFAULT_RETRY_WRITES,
) -> MongoClient:
    """Creates a PyMongo MongoClient instance configured with Atlas timeout requirements
    and role-based credentials.
    Atlas defaults:
      serverSelectionTimeoutMS = 10000 (10s)
      connectTimeoutMS         = 10000 (10s)
      socketTimeoutMS          = 60000 (60s)
      retryWrites              = True
    """
    raw_uri = uri or os.environ.get("MONGO_URI", "").strip()
    if not raw_uri:
        raise ValueError(
            "MongoDB connection URI is not configured. "
            "Please declare MONGO_URI in your environment or .env file."
        )

    effective_uri = build_role_uri(raw_uri, role=role)

    try:
        client: MongoClient = MongoClient(
            effective_uri,
            serverSelectionTimeoutMS=server_selection_timeout_ms,
            connectTimeoutMS=connect_timeout_ms,
            socketTimeoutMS=socket_timeout_ms,
            retryWrites=retry_writes,
        )
        return client
    except Exception as exc:
        sanitized = sanitize_error(exc)
        logger.error("Failed to initialize MongoClient: %s", sanitized)
        raise RuntimeError(f"Failed to initialize MongoClient: {sanitized}") from exc


def get_database(
    client: Optional[MongoClient] = None,
    role: str = "app",
) -> Database:
    """Returns the target Database instance using the specified client or creating one."""
    active_client = client or get_mongo_client(role=role)
    db_name = get_database_name()
    return active_client[db_name]
