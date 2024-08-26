from pyspark.sql import SparkSession
from pykafka import KafkaClient
import time
import pandas as pd
import json

def preprocess_data(df):
    # Handle missing values and incorrect data
    # Fill missing 'Arrival time' with a placeholder if needed
    df['Arrival time'].fillna('00:00:00', inplace=True)
    df['Departure Time'].fillna('00:00:00', inplace=True)
    
    # Convert Arrival and Departure times to datetime
    df['Arrival time'] = pd.to_datetime(df['Arrival time'], format='%H:%M:%S', errors='coerce')
    df['Departure Time'] = pd.to_datetime(df['Departure Time'], format='%H:%M:%S', errors='coerce')
    
    # Handle rows where conversion failed
    df = df.dropna(subset=['Arrival time', 'Departure Time'])
    
    # Adjust time values for consistency
    # Assuming departure time should be later than arrival time
    df = df[df['Departure Time'] > df['Arrival time']]
    
    # Convert Timestamp columns to strings to make them JSON serializable
    df['Arrival time'] = df['Arrival time'].dt.strftime('%H:%M:%S')
    df['Departure Time'] = df['Departure Time'].dt.strftime('%H:%M:%S')
    
    return df

def send_data(topic):
    # Create a Kafka client and producer
    client = KafkaClient(hosts="34.131.9.138:9092")
    producer = client.topics[topic].get_producer()

    # Read the train schedule CSV file into a DataFrame
    csv_path = "gs://oppe-bucket-ibd/Final_Assignment_25Aug2024_Arshi/Train_details_22122017.csv"
    df = pd.read_csv(csv_path)

    # Preprocess the data
    df = preprocess_data(df)
    
    # Iterate over each row and send to Kafka as JSON
    for _, row in df.iterrows():
        row_dict = row.dropna().to_dict()  # Convert row to dictionary, drop NaNs
        producer.produce(json.dumps(row_dict).encode('utf-8'))

        print(f"Sent data for Train {row['Train No']} at {row['Station Name']}\n")

        # Wait for the next interval (optional)
        time.sleep(10)

# Initialize Spark session
spark = SparkSession.builder.appName("TrainScheduleToKafka").getOrCreate()
print("\nSpark session started\n")

try:
    topic_name = 'quickstart-events'
    send_data(topic_name)
except KeyboardInterrupt:
    print("\nProducer terminated.")
finally:
    spark.stop()
