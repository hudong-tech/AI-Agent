
FROM ubuntu:latest

RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && apt-get update && apt-get install -y coturn python3 python3-pip redis-server && rm -rf /var/lib/apt/lists/*

RUN pip3 install upgrade pip && pip3 install fastapi uvicorn langchain-core langchain-openai langchain langchain-community openai redis google-search-results

COPY turnserver.conf /etc/turnserver.conf

COPY redis.conf /etc/redis/redis.conf

VOLUME /data

WORKDIR /app

COPY . /app

EXPOSE 8000 3478 6379

CMD ["sh", "-c", "turnserver -c /etc/turnserver.conf --listening-ip=0.0.0.0 --listening-port=3478 & redis-server /etc/redis/redis.conf --protected-mode no & sleep 3 && uvicorn server:app --host 0.0.0.0 --port 8000"]