from pymongo import MongoClient
from datetime import date, datetime

client = MongoClient('mongodb+srv://sumant-dusane:sumant123456@sumant-dusane.3donyvz.mongodb.net/?retryWrites=true&w=majority')
db = client['iot-marketplace']
collection = db['datasets']


def upload_file(files):
    if files:
        binary_files = []
        for filepath in files:
            with open(filepath, 'rb') as f:
                binary_data = f.read()
                binary_files.append(binary_data)
                f.close()
        
        collection.insert_one({
            'title': 'Dataset of ' + str(date.today()),
            'data': binary_files,
            'timestamp': datetime.now(),
        })