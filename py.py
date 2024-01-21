import psycopg2

# Configura los detalles de conexión
dbname = "music-app"
user = "root"
password = "musicapp"
host = "localhost"                             #O la dirección IP de tu máquina si ejecutas Python desde otra máquina
port = "5432"




# Intenta establecer la conexión
try:
    connection = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Conexión exitosa!")

    # Ejemplo de consulta para imprimir datos de la tabla
    with connection.cursor() as cursor:
        # Cambia "mi_tabla" por el nombre real de tu tabla
        table_name = "public.city"

        # Consulta para seleccionar todos los registros de la tabla
        query = f"SELECT * FROM {table_name};"
        
        cursor.execute(query)
        rows = cursor.fetchall()

        # Imprime los datos de la tabla
        print(f"\nContenido de la tabla {table_name}:\n")
        for row in rows:
            print(row)


except Exception as e:
    print("Error al conectarse a la base de datos:", e)


finally:


    # Cierra la conexión al finalizar
    if connection:
        connection.close()
        print("Conexión cerrada.")
