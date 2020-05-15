import pymysql.cursors
import settings

connection = pymysql.connect(host='localhost',
                             user=settings.settings['DB_USR'],
                             password=settings.settings['DB_PW'],
                             db='pokemon',
                             cursorclass=pymysql.cursors.DictCursor)


def get_facts(name):
    try:
        with connection.cursor() as cursor:
            sql = "select * from pokemon where name = %s"
            cursor.execute(sql, name)
            result = cursor.fetchone()
            return result
    finally:
        connection.close()
