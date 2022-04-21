#!/usr/bin/python3 -u
"""
    face_eater.py - a daemon for defacing NIfTI files from Posda

    This program follows this general pattern:
        1 Block on redis, retreiving a new (defacing_id, nifti_file_id) tuple
        2 Retrive the nifti file from Posda API
        3 Executing defacing program
        4 Upload resulting files (nifti and 3d renderings) to Posda
        5 Call success/failure endpoint to record new status (and file_ids)
"""
import redis
import subprocess
import requests
import os
import hashlib
import tempfile
import argparse
import json
import logging
import graypy
from enum import Enum
from pprint import pprint

REDIS_HOST=os.environ['POSDA_REDIS_HOST']
REDIS_PORT=int(os.environ.get('POSDA_REDIS_PORT', 6379))
REDIS_QUEUE=os.environ.get('FACE_EATER_REDIS_QUEUE', 'defacing_queue')
POSDA_API=os.environ['POSDA_API_URL']
GRAYLOG_HOST=os.environ.get('GRAYLOG_HOST', None)
GRAYLOG_PORT=os.environ.get('GRAYLOG_PORT', None)
DEBUG=bool(os.environ.get('FACE_EATER_DEBUG', False))


CHUNK_SIZE=10485760 # 10MiB
LOG = None

# Error codes as returned by masker
class ErrorCode(Enum):
    UNKNOWN = 1
    ARG_ERROR = 2
    NO_FACE = 3
    UNSUPPORTED_ORIENTATION = 4
    BAD_WINDOW = 5
    FEW_SLICES = 6


def md5sum(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def add_file(filename):
    url = POSDA_API + "/v1/import/file"
    digest = md5sum(filename)
    with open(filename, "rb") as infile:
        r = requests.put(url, params={
            'digest': digest,
            'localpath': filename,
        }, data=infile)

        try:
            resp = r.json()
            return resp['file_id']
        except:
            print(r.content)
            raise

def process(defacing_id, file_id):
    # get file details for error detection
    req = requests.get(f"{POSDA_API}/v1/files/{file_id}")
    info = req.json()

    LOG.debug(info)

    # download the file from posda
    _, temp_file = tempfile.mkstemp(suffix='.nii')
    output_dir = tempfile.TemporaryDirectory()
    req = requests.get(f"{POSDA_API}/v1/files/{file_id}/data", stream=True)
    req.raise_for_status()

    LOG.debug(f"using temp file {temp_file} and output dir {output_dir.name}")

    LOG.debug(f"beginning download of file with file_id {file_id}")
    read_digest = hashlib.md5()
    with open(temp_file, "wb") as f:
        bytes_read = 0
        for chunk in req.iter_content(chunk_size=CHUNK_SIZE):
            bytes_read += len(chunk)
            f.write(chunk)
            read_digest.update(chunk)
            LOG.debug(f"% read: {int(bytes_read / info['size'] * 100)}")


    LOG.debug(f"total bytes read: {bytes_read}")
    LOG.debug(f"expected size: {info['size']}")
    LOG.debug(f"expected digest: {info['digest']}")
    LOG.debug(f"actual digest: {read_digest.hexdigest()}")

    if info['digest'] != read_digest.hexdigest():
        raise RuntimeError("downloaded file does not match expected digest")


    error_code = None
    try:
        out = subprocess.check_output([
            "python3",
            "/app/masker/masker.py",
            "-i", temp_file,
            "-o", output_dir.name
        ], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        error_code = ErrorCode(e.returncode) 
        out = e.output

    LOG.info("Masker complete")
    # lines = out.split(b'\n')
    # pprint(lines)
    LOG.debug(out)

    if error_code is not None:
        LOG.info(f"Masker returned error code: {error_code}")
        process_error(error_code, defacing_id, output_dir.name)
        output_dir.cleanup()
        return

    process_success(defacing_id, output_dir.name)
    output_dir.cleanup()
    LOG.info("Masker was successful")

    return

def process_error(error_code, defacing_id, output_dir):
    LOG.debug("Processing error state")
    if error_code != error_code.UNKNOWN and error_code != error_code.UNSUPPORTED_ORIENTATION:
        vr_before = os.path.join(output_dir, 'vr_before.png')
        vr_before_id = add_file(vr_before)
    else:
        vr_before_id = None

    LOG.debug(f"vr_before_id={vr_before_id}")

    # call api to associate masked_file file_id with defacing_id
    req = requests.post(POSDA_API + f"/v1/deface/{defacing_id}/error",
                        json={
                            "three_d_rendered_face": vr_before_id,
                            "error_code": str(error_code)
                        })

    LOG.debug(req.status_code)
    LOG.debug(req.content)

def process_success(defacing_id, output_dir):
    LOG.debug("Processing success state")
    masked_file = os.path.join(output_dir, 'masked.nii.gz')
    vr_masked = os.path.join(output_dir, 'vr_masked.png')
    vr_before = os.path.join(output_dir, 'vr_before.png')
    vr_facebox = os.path.join(output_dir, 'vr_facebox.png')

    # upload all files to posda
    masked_file_id = add_file(masked_file)
    vr_masked_id = add_file(vr_masked)
    vr_before_id = add_file(vr_before)
    vr_facebox_id = add_file(vr_facebox)

    LOG.debug(f"vr_before_id={vr_before_id}")
    LOG.debug(f"vr_facebox_id={vr_facebox_id}")
    LOG.debug(f"vr_masked_id={vr_masked_id}")
    LOG.debug(f"masked_file_id={masked_file_id}")

    # call api to associate masked_file file_id with defacing_id
    req = requests.post(POSDA_API + f"/v1/deface/{defacing_id}/complete",
                        json={
                            "nifti_file_id": masked_file_id,
                            "three_d_rendered_face": vr_before_id,
                            "three_d_rendered_face_box": vr_facebox_id,
                            "three_d_rendered_defaced": vr_masked_id,
                        })

    LOG.debug(req.status_code)
    LOG.debug(req.content)


def main(exit_after=False):

    redis_db = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    LOG.debug(f"connected to redis ({REDIS_HOST})")

    while True:
        # ask for an item of work, and timeout after 5 seconds of waiting
        sr = redis_db.brpop(REDIS_QUEUE, 5)
        LOG.debug("got redis message: %s", sr)
        # if we timed out, just loop to try again
        if sr is None:
            if exit_after is True:
                LOG.info("Redis queue is empty and --exit-on-empty was "
                             "specified. Exiting normally.")
                return
            continue

        _, message = sr
        defacing_id, file_id = json.loads(message)


        set_logged_defacing_id(defacing_id)
        LOG.info(f"got work: defacing_id={defacing_id}, file_id={file_id}")
        process(defacing_id, file_id)

def main_for_single(defacing_id):
    LOG.info(f"Running for defacing_id: {defacing_id}")

    # look up the file_id for this defacing_id
    req = requests.get(f"{POSDA_API}/v1/deface/{defacing_id}")
    info = req.json()
    file_id = info['from_nifti_file']

    LOG.debug(f"Got nifti file_id {file_id} from api")

    return process( defacing_id=defacing_id, file_id=file_id)

def parse_args():
    parser = argparse.ArgumentParser(
        description=f"""\
        %(prog)s is a daemon which performs defacing on NIfTI files using
        Chris Wardell's Masker system.

        If run without arguments, it will connect to redis ({REDIS_HOST}) 
        and poll the queue '{REDIS_QUEUE}' waiting for work to perform.
        """
    )
    parser.add_argument('--defacing_id',
                        help='process this defacing_id and exit')
    parser.add_argument(
        '--exit-on-empty',
        action='store_true',
        help='exit if the queue is empty, instead of waiting'
    )

    return parser.parse_args()    

def configure_logging():
    global DEBUG
    global LOG

    print("DEBUG is: ", DEBUG)

    loglevel=logging.DEBUG if DEBUG else logging.INFO

    logging.basicConfig(level=loglevel,
                        format='%(asctime)s %(levelname)s %(name)s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger()  # the root logger, as configured above
    if GRAYLOG_HOST is not None:
        print(GRAYLOG_HOST, GRAYLOG_PORT)
        logger.info("logging to Graylog: %s", GRAYLOG_HOST)
        #logger = logging.getLogger("face_eater")
        logger.setLevel(loglevel)
        graypy_handler = graypy.GELFTCPHandler(GRAYLOG_HOST, int(GRAYLOG_PORT))
        logger.addHandler(graypy_handler)

        adapter = logging.LoggerAdapter(logger, {
            "defacing_id": 0,
        })

        # use this custom adapter for the global logger
        LOG = adapter
    else:
        LOG = logger

def set_logged_defacing_id(defacing_id):
    global LOG
    if isinstance(LOG, logging.LoggerAdapter):
        LOG.extra['defacing_id'] = defacing_id


if __name__ == "__main__":
    print("Face Eater, a face removing daemon")
    args = parse_args()
    if args.defacing_id is None:
        configure_logging()
        print(LOG)
        LOG.debug(args)
        main(args.exit_on_empty)
    else:
        DEBUG=True
        configure_logging()
        set_logged_defacing_id(args.defacing_id)
        main_for_single(args.defacing_id)
