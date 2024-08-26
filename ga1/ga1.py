from google.cloud import storage

def count_lines_in_gcs_file(bucket_name, file_name):
        # Initialize a Google Cloud Storage client
        client = storage.Client()
        # Get the bucket containing the file
        bucket = client.get_bucket(bucket_name)

        # Get the blob (file object) from the bucket
        blob = bucket.blob(file_name)

        # Download the file's content as a string
        file_content = blob.download_as_text()

        # Count the number of lines in the file
        line_count = len(file_content.splitlines())
        return line_count

 # Example usage
if __name__ == "__main__":
    bucket_name = '21f3002806'
    file_name = 'GA-1/GA-1-IBD.txt'
    print(f"Number of lines in {file_name}: {count_lines_in_gcs_file(bucket_name, file_name)}")