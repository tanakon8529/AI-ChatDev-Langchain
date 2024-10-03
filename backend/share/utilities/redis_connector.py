
import redis

from fastapi import HTTPException
from utilities.log_controler import LogControler
log_controler = LogControler()

from settings.configs import PORT_REDIS, REDIS_PASSWORD, HOST_REDIS

def get_cluster_client():
    try:
        redis_cluster = redis.RedisCluster(
            startup_nodes=[
                {"host": "localhost", "port": "7000"},
                {"host": "localhost", "port": "7001"},
                {"host": "localhost", "port": "7002"},
            ],
            decode_responses=True,
        )
    except Exception as e:
        log_controler.log_error(f"Error connecting to Redis Cluster: {e}")
        raise HTTPException(status_code=500, detail='Redis Cluster not Connected')
    return 

def get_client():
    try:
        redis_client = redis.Redis(
            host=HOST_REDIS,
            port=PORT_REDIS,
            db=0,
            password=REDIS_PASSWORD,
            decode_responses=True,
        )
        redis_client.ping()
    except Exception as e:
        log_controler.log_error(f"Error connecting to Redis: {e}")
        raise HTTPException(status_code=500, detail='Redis not Connected')
    return redis_client