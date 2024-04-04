from pymongo import MongoClient
from gridfs import GridFS

client = MongoClient('mongodb+srv://sumant-dusane:sumant123456@sumant-dusane.3donyvz.mongodb.net/?retryWrites=true&w=majority')
db = client['iot-marketplace']
collection = db['datasets']
fs = (GridFS(db, collection="datasets"))

def upload_file(file):
    if file:
        filename = str(file.name)[6:].replace("/","")
       
        file_data = file.read()
        file_id = fs.put(file_data, filename=filename)
        print(file_id)