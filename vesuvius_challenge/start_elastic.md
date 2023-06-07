docker pull docker.elastic.co/elasticsearch/elasticsearch:8.8.0

docker network create elastic

docker run --name es01 --net elastic -p 9200:9200 -it docker.elastic.co/elasticsearch/elasticsearch:8.8.0

docker cp es01:/usr/share/elasticsearch/config/certs/http_ca.crt .