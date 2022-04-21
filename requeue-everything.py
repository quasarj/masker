#!/usr/bin/env python3

import psycopg2
import redis
import os
import json

REDIS_HOST=os.environ['POSDA_REDIS_HOST']
REDIS_QUEUE=os.environ.get('FACE_EATER_REDIS_QUEUE', 'defacing_queue')

def main():
    redis_db = redis.StrictRedis(host=REDIS_HOST, db=0)

    with psycopg2.connect(dbname="posda_files") as conn:
        cur = conn.cursor()
        cur.execute("""\
            select file_nifti_defacing_id, from_nifti_file
            from file_nifti_defacing
        """)

        for defacing_id, file_id in cur:
            print(defacing_id)
            redis_db.lpush(REDIS_QUEUE, json.dumps([defacing_id, file_id]))


if __name__ == '__main__':
    main()
