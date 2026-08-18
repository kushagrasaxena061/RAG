import sqlite3, json, os, traceback

class SQLiteManager:
    def __init__(self, db_path="./data/rag_state.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (hash_key TEXT PRIMARY KEY, response TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS documents (doc_hash TEXT PRIMARY KEY, metadata TEXT, version TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS page_hashes (doc_name TEXT, page_num INTEGER, page_hash TEXT, PRIMARY KEY(doc_name, page_num))")
        self.conn.execute("CREATE TABLE IF NOT EXISTS crash_logs (id INTEGER PRIMARY KEY, error TEXT, trace TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
        self.conn.commit()

    def log_crash(self, error: Exception):
        self.conn.execute("INSERT INTO crash_logs (error, trace) VALUES (?, ?)", (str(error), traceback.format_exc()))
        self.conn.commit()

    def save_document(self, doc_hash, metadata, version="1.0"):
        self.conn.execute("INSERT OR REPLACE INTO documents (doc_hash, metadata, version) VALUES (?, ?, ?)", (doc_hash, json.dumps(metadata), version))
        self.conn.commit()

    def get_page_hash(self, doc_name: str, page_num: int):
        cursor = self.conn.execute("SELECT page_hash FROM page_hashes WHERE doc_name = ? AND page_num = ?", (doc_name, page_num))
        row = cursor.fetchone()
        return row[0] if row else None

    def save_page_hash(self, doc_name: str, page_num: int, page_hash: str):
        self.conn.execute("INSERT OR REPLACE INTO page_hashes (doc_name, page_num, page_hash) VALUES (?, ?, ?)", (doc_name, page_num, page_hash))
        self.conn.commit()