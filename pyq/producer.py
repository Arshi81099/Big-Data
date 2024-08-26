import json
import time
from kafka import KafkaProducer
from google.cloud import storage

kafka_topic = "quickstart-events"
kafka_server = "34.131.174.82:9092"

producer = KafkaProducer(
    bootstrap_servers=kafka_server,
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

def read_gcs_file(bucket_name, file_name):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_name)
    content = blob.download_as_text()
    return content

bucket_name = "pyq"
file_name = "test/part-00039-808f9971-b2b6-4a6f-b8cf-0822a68f365f-c000.csv"
text_content = read_gcs_file(bucket_name, file_name)

# Splitting the text content into paragraphs
paragraphs = text_content.split("\n")

batch_size = 1000
for i in range(0, len(paragraphs), batch_size):
    batch = paragraphs[i:i + batch_size]
    for paragraph in batch:
        paragraph = paragraph.strip()  # Remove any leading or trailing whitespace
        if paragraph:  # Check if paragraph is not empty
            record = {"text": paragraph}
            producer.send(kafka_topic, value=record)
    print(f"Message Sent. Batch No: {i//batch_size + 1}")
    time.sleep(20)  # Simulate a delay between batches

producer.flush()
producer.close()
