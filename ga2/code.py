import functions_framework
from google.cloud import storage

# Triggered by a change in a storage bucket
@functions_framework.cloud_event
def toji_zenin(cloud_event):
  data = cloud_event.data
  bucket = data["bucket"]
  name = data["name"]

  bucket_name = storage.Client().bucket(bucket)
  txt_file = bucket_name.blob(name)

  if not txt_file:
    print(f"There is no file named {name} in the bucket {bucket}")
    return
  else:
    txt_file_content = txt_file.download_as_text()
    lines = txt_file_content.split('\n')
    count = len(lines)

    print(f"The number of lines in {name} is {count}")





