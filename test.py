import redis
import json

REDIS_HOST="tcia-dev-2.ad.uams.edu"
REDIS_QUEUE="deface_queue"

redis_db = redis.StrictRedis(host=REDIS_HOST, db=0)


dat = [
[3 ,           16682],
[11 ,           16680],
[12 ,           16696],
[1 ,           16694],
[2 ,           16688],
[4 ,           16692],
[6 ,           16684],
[7 ,           16628],
[8 ,           16686],
[10 ,           16690],
[5 ,           16698],
[9 ,           16678],
]


for pair in dat:
    redis_db.lpush(REDIS_QUEUE, json.dumps(pair))
