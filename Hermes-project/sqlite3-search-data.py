import sqlite3


class CustomSqlite:
    def __init__(self, name_db: str):
        self.connect = sqlite3.connect(name_db)
        self.cursor = self.connect.cursor()

    def executeCommand(self, cmd: str):
        return self.cursor.execute(cmd)

    def get_answer(self):
        rows = self.cursor.fetchall()
        return rows if rows else "Ответ пришёл пустым"

    def get_tables(self):
        self.executeCommand("SELECT name FROM sqlite_master WHERE type='table';")
        return self.get_answer()

    def get_executions(self, cnt_rows: int):
        answer = []
        for row in self.executeCommand(
            f"SELECT job_id, status, claimed_at, finished_at, error "
            f"FROM executions ORDER BY rowid DESC LIMIT {cnt_rows}"
        ):
            answer.append(row)
        return answer

    def get_cron_incidents(self, cnt_rows: int):
        answer = []
        for row in self.executeCommand(
            f"SELECT job_id, state, failure_type, first_seen_at, error "
            f"FROM cron_incidents ORDER BY rowid DESC LIMIT {cnt_rows}"
        ):
            answer.append(row)
        return answer  # ← добавили return

    def close(self):
        self.connect.close()


def main():
    sqlite = None
    try:
        sqlite = CustomSqlite(name_db="executions.db")
        tables = sqlite.get_tables()
        print(tables)
        for row_execut in sqlite.get_executions(1000):
            print(row_execut)
        for row_inic in sqlite.get_cron_incidents(1000):
            print(row_inic)
    except Exception as ex:
        print(f"Произошла ошибка : {ex}")
    finally:
        if sqlite is not None:
            sqlite.close()


if __name__ == "__main__":
    main()
