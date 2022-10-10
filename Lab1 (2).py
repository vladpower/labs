#!/usr/bin/env python
# coding: utf-8

# In[3]:


from calendar import weekheader
from itertools import count
from pyspark.sql import DataFrame as df
from pyspark.sql import SparkSession
from pyspark.sql.functions import count 
from pyspark.sql.functions import desc
from pyspark.sql.functions import lower, col
from pyspark.sql.functions import column
from pyspark.sql.functions import round
from pyspark.sql.types import IntegerType

spark = SparkSession.builder.master("local[1]") \
                    .appName('SparkApp') \
                    .getOrCreate()

tiktokData2022 = spark\
  .read\
  .option("inferSchema", "true")\
  .option("header", "true")\
  .csv("/home/student/Desktop/TikTok_songs_2022.csv")

tiktokData2022.createOrReplaceTempView("TikTok_songs_2022")


# In[6]:


sqlWay = spark.sql("""
SELECT DISTINCT artist_name, artist_pop
FROM TikTok_songs_2022
WHERE artist_pop > 90
ORDER BY artist_pop DESC
""")

dW = tiktokData2022\
    .select("artist_name" ,
    "artist_pop")\
    .where("artist_pop > 90")\
    .orderBy(desc("artist_pop"))

print(dataFrameWay.rdd.map(lambda x: x[0]).distinct().collect())


# In[14]:


sqlWay1 = spark.sql("""
SELECT DISTINCT artist_name, COUNT(artist_name) as count_trek
FROM TikTok_songs_2022
GROUP BY artist_name
HAVING COUNT(artist_name) > 2
ORDER BY count_trek DESC
""")

dW1 = tiktokData2022\
    .groupBy("artist_name")\
    .agg(count("artist_name").alias("count"))\
    .where("count > 2")\
    .sort(desc("count"))\
    .show()



# In[17]:


sqlWay2 = spark.sql("""
SELECT track_name
FROM TikTok_songs_2022
WHERE loudness BETWEEN -10 and -9
""")

dW2 = tiktokData2022\
    .select("track_name")\
    .filter("loudness between -10 and -9")\
    .show()


# In[18]:


sqlWay3 = spark.sql("""
SELECT track_name
FROM TikTok_songs_2022
WHERE lower(track_name) LIKE lower('%Love%')
""")

dW3 = tiktokData2022\
    .select("track_name")\
    .where(lower(col("track_name")).like("%love%"))\
    .show()


# In[20]:


sqlWay4 = spark.sql("""
SELECT track_name, artist_name
FROM TikTok_songs_2022
WHERE artist_name = 'Ariana Grande' OR artist_name = 'Astelle'
ORDER BY artist_name , track_name
""")

dW4 = tiktokData2022\
    .select("track_name", "artist_name")\
    .where(col("artist_name").like("Ariana Grande") | col("artist_name").like("%Astelle%"))\
    .orderBy("artist_name","track_name")\
    .show()


# In[23]:


from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.sql import Row
from pyspark.ml.feature import MinMaxScaler
from IPython.display import display
import numpy as np

k = 20

np.set_printoptions(precision=3, suppress=True)

columns_to_scale = ["danceability", "energy", "speechiness", "acousticness",
                    "instrumentalness","liveness", "valence", "duration_ms",
                    "tempo", "track_pop"]
assemblers = [VectorAssembler(inputCols = [col], outputCol = col + "_vec") for col in columns_to_scale]
scalers = [MinMaxScaler(inputCol = col + "_vec", outputCol = col + "_scaled") for col in columns_to_scale]

vectorAssembler = VectorAssembler()\
  .setInputCols(["danceability_scaled", "energy_scaled", "speechiness_scaled", "acousticness_scaled",
                 "instrumentalness_scaled", "liveness_scaled", "valence_scaled", "duration_ms_scaled",
                 "tempo_scaled", "track_pop_scaled"])\
  .setOutputCol("features")

transformationPipeline = Pipeline()\
  .setStages(assemblers + scalers + [vectorAssembler])

fittedPipeline = transformationPipeline.fit(tiktokData2022)
transformedDF = fittedPipeline.transform(tiktokData2022)

kmeans = KMeans()\
  .setK(k)\
  .setSeed(300)

kmModel = kmeans.fit(transformedDF)

df_pred = kmModel.transform(transformedDF)

centers = kmModel.clusterCenters()
for i in range(0, k):
    print("group " + str(i))
    print(centers[i])
    df = df_pred.selectExpr("track_name", "danceability", "energy", "speechiness",
                        "acousticness","instrumentalness","liveness", "valence",
                        "duration_ms", "tempo", "track_pop")\
        .where("prediction = " + str(i))\
        .toPandas()
    display(df)


# In[25]:


# Задание 6:
# 1) Это метод кластеизации, суть которого минимизировать сильное отклоенение данных от средних значений.
# 2) Также можно использовать линейную регрессию.
# 3) Благодаря полученным группах данных можно понять какая группа песен более универсальная или имеют превосходство в нескольких особенностях песен


# In[ ]:




