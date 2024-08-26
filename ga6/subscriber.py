from google.cloud import pubsub_v1
from google.cloud import storage

def process_file(file_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket("ibd-ga6-bucket")
    blob = bucket.blob(file_name)
    with blob.open("r") as file:
        line_count = sum(1 for _ in file)
    print(f"Number of lines in {file_name}: {line_count}")

def message_handler(message):
    file_name = message.data.decode('utf-8')
    print(f"Received message for file: {file_name}")
    process_file(file_name)
    message.ack()  # Acknowledge the message

def subscribe_to_topic():
    subscriber = pubsub_v1.SubscriberClient()
    subscription_path = subscriber.subscription_path('ibd-ga6', 'ibd-ga6-pubsub-su')
    print(f"Listening for messages on {subscription_path}")
    future = subscriber.subscribe(subscription_path, callback=message_handler)
    return future

if __name__ == "__main__":
    future = subscribe_to_topic()
    try:
        future.result()
    except KeyboardInterrupt:
        future.cancel()
        print("Subscription cancelled.")