from redis import Redis

cache = Redis(host="redis", port=6379, db=0)