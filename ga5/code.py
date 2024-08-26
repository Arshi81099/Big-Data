from datetime import datetime
from pyspark.sql import SparkSession

### Helpers
def create_spark_session(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()


def read_csv(spark, path):
    return spark.read.csv(path, header=True, inferSchema=True)


def get_current_date():
    return datetime.now().strftime("%d-%m-%Y")

### Core
def create_scd_type_2_query():
    return """
    SELECT ROW_NUMBER() OVER (ORDER BY validity_start) as Index, *
    FROM (
        SELECT 
            master.Name, 
            DOB, 
            validity_start, 
            {current_date} as validity_end
        FROM {master_data} master 
        INNER JOIN {update_data} update 
            ON (master.Name = update.Name) 
            AND (master.DOB != update.updated_DOB)
        WHERE to_date(validity_end, {format}) > to_date({current_date}, {format})
        
        UNION
        
        SELECT 
            master.Name, 
            DOB, 
            validity_start, 
            validity_end
        FROM {master_data} master 
        LEFT JOIN {update_data} update 
            ON update.Name = master.Name
        WHERE update.Name is NULL
        
        UNION
        
        SELECT 
            Name, 
            updated_DOB as DOB, 
            {current_date} as validity_start, 
            {end_date} as validity_end
        FROM {update_data} update
    )
    """

### Runner
def main():
    spark = create_spark_session("SCD_Type_2")

    original = read_csv(spark, "gs://ibd-ga5/original.csv")
    original.createOrReplaceTempView("master_data")

    updated = read_csv(spark, "gs://ibd-ga5/updated.csv")
    updated.createOrReplaceTempView("update_data")

    current_date = get_current_date()
    sql_query = create_scd_type_2_query()

    updated_data = spark.sql(
        sql_query,
        master_data=original,
        update_data=updated,
        current_date=current_date,
        end_date="31-12-9999",
        format="dd-mm-yyyy",
    )
    updated_data.show()

    spark.stop()


if __name__ == "__main__":
    main()
 # type: ignore