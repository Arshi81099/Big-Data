import functions_framework
from google.cloud import pubsub_v1

PROJECT_ID = "ibd-ga3"
TOPIC_NAME = "ibd-ga6-pubsub"

publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(PROJECT_ID, TOPIC_NAME)

@functions_framework.cloud_event
def ibd_ga6(cloud_event):
    data = cloud_event.data
    bucket = data["bucket"]
    name = data["name"]
    file_path = f"gs://{bucket}/{name}"

    # Publish the file path to Pub/Sub
    publisher.publish(topic_path, file_path.encode("utf-8"))
    print(f"File: {file_path} uploaded and published to Pub/Sub")