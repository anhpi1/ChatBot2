import pyodbc
import numpy as np
import ast
def print_Bt(bt):

    # Thông tin kết nối SQL Server
    server = ''
    database = ''
    username = ''
    password = ''
    topics =[]
    content = []
    # tải tham số
    with open("parameter.ta", "r") as file:
        lines = file.readlines()
    for line in lines:
        # Bỏ qua các dòng trống
        if not line.strip():
            continue
        # Tách dòng thành key và value
        key, value = line.split(" = ")
        key = key.strip()
        value = value.strip()
        if key == "server":
            server = value.strip("'")
        if key == "database":
            database = value.strip("'")
        if key == "username":
            username = value.strip("'")
        if key == "file_input_train":
            password = value.strip("'")
        if key == "topics":
            topics = value
        if line.strip().startswith("topics = "):
            # Trích xuất chuỗi sau 'topics = '
            topics_str = line.strip()[len("topics = "):].strip()
            
            # Dùng ast.literal_eval để chuyển chuỗi thành danh sách Python
            topics = ast.literal_eval(topics_str)

    Bt=[int(i) for i in bt]
    
    content = []

    # Connect to SQL Server
    conn = pyodbc.connect(
        f'DRIVER={{SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
    )

    # Create a cursor object
    cursor = conn.cursor()

    # Loop through each table and corresponding ID in Bt
    for topic, id_value in zip(topics, Bt):
        # Convert numpy int to standard int if necessary
        
        id_value = str(id_value)  # Convert to Python str
        #print("SELECT content FROM dbo.{} WHERE id = {};".format(topic, id_value))
        # Execute query for each table and ID
        cursor.execute("SELECT content FROM dbo.{} WHERE id = {};".format(topic,id_value))
        
        # Fetch the results and append to content list
        content.append(cursor.fetchall())

    # Close the connection
    conn.close()
    
    # Print the fetched content
    for index, data in enumerate(content):
        print(f"{topics[index]}: {data}")

