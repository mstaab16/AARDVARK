import psycopg2

conn = psycopg2.connect(database="aardvark_pg",
                        host="localhost",
                        user="aardvark_user",
                        password="arpes",
                        port="5432")

cursor = conn.cursor()
# cursor.execute("SELECT * FROM DB_table WHERE id = 1")
# print(cursor.fetchone())
print(conn)
conn.close()

print(conn)