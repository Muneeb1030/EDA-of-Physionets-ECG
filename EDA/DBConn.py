import psycopg2

hostname = "localhost"
username = "postgres"
password = "admin"
portid = 5432
dbname = "EDA"
try:
    with psycopg2.connect(
        host = hostname,
        port =  portid,
        dbname = dbname,
        password = password,
        user = username
    ) as conn:

        with conn.cursor() as cur:
            
            # createQuery = '''CREATE TABLE IF NOT EXISTS EMP(
            #     id int primary key,
            #     name varchar(40))'''
            # cur.execute(createQuery)
            
            
            # insertQuery = '''INSERT INTO EMP (id, name) Values (%s,%s)'''
            # insertValue =  (1,"Ali")
            # cur.execute(insertQuery,insertValue)
            
            deleteQuery =  '''DELETE FROM EMP WHERE id = %s'''
            deleteValue = (1,)
            cur.execute(deleteQuery,deleteValue)
            
            

except Exception as error:
    print(error)
finally:
        conn.close()
    