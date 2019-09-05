import argparse, logging, os.path
from time import time

from bottle import get, run, template, post, request
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket
from gevent.lock import BoundedSemaphore

from scipy import signal
from scipy.io.wavfile import read as wav_read

import deepspeech
import numpy as np
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=20,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )

parser = argparse.ArgumentParser(description='')
parser.add_argument('-m', '--model', required=True,
                    help='Path to the model (protocol buffer binary file, or directory containing all files for model)')
parser.add_argument('-a', '--alphabet', nargs='?', const='alphabet.txt',
                    help='Path to the configuration file specifying the alphabet used by the network. Default: alphabet.txt')
parser.add_argument('-l', '--lm', nargs='?', const='lm.binary',
                    help='Path to the language model binary file. Default: lm.binary')
parser.add_argument('-t', '--trie', nargs='?', const='trie',
                    help='Path to the language model trie file created with native_client/generate_trie. Default: trie')
parser.add_argument('--lw', type=float, default=1.5,
                    help='The alpha hyperparameter of the CTC decoder. Language Model weight. Default: 1.5')
parser.add_argument('--vwcw', type=float, default=2.25,
                    help='Valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary. Default: 2.25')
parser.add_argument('--bw', type=int, default=1024,
                    help='Beam width used in the CTC decoder when building candidate transcriptions. Default: 1024')
parser.add_argument('-p', '--port', default=8080,
                    help='Port to run server on. Default: 8080')
parser.add_argument('--debuglevel', default=20,
                    help='Debug logging level. Default: 20')
ARGS = parser.parse_args()

logging.getLogger().setLevel(int(ARGS.debuglevel))

server_state = {
    "start_time": None, 
    "semaphore_acquired": False, 
    "sctx": None,  # stream context/state
    "semaphore": BoundedSemaphore(1)  # Only one Deepspeech instance available at a time 
}

if os.path.isdir(ARGS.model):
    model_dir = ARGS.model
    ARGS.model = os.path.join(model_dir, 'output_graph.pb')
    ARGS.alphabet = os.path.join(model_dir, ARGS.alphabet if ARGS.alphabet else 'alphabet.txt')
    if ARGS.lm: ARGS.lm = os.path.join(model_dir, ARGS.lm)
    if ARGS.trie: ARGS.trie = os.path.join(model_dir, ARGS.trie)

LM_WEIGHT = ARGS.lw
VALID_WORD_COUNT_WEIGHT = ARGS.vwcw
BEAM_WIDTH = ARGS.bw
N_FEATURES = 26
N_CONTEXT = 9
AUDIO_RATE = 16000

print('Initializing model...')
logger.info("ARGS.model: %s", ARGS.model)
logger.info("ARGS.alphabet: %s", ARGS.alphabet)

model = deepspeech.Model(ARGS.model, N_FEATURES, N_CONTEXT, ARGS.alphabet, BEAM_WIDTH)
if ARGS.lm and ARGS.trie:
    logger.info("ARGS.lm: %s", ARGS.lm)
    logger.info("ARGS.trie: %s", ARGS.trie)
    model.enableDecoderWithLM(ARGS.alphabet,
                              ARGS.lm,
                              ARGS.trie,
                              LM_WEIGHT,
                              VALID_WORD_COUNT_WEIGHT)
    logger.info("Model load complete.")

# perform metadata flattening to a python dict for easy serialization
def regularize_metadata(metadata):
    # https://github.com/mozilla/DeepSpeech/blob/4c14c6b78b3daf90b67f840035a991bb94d9e1fa/native_client/deepspeech.h#L26
    return_data = { "probability": round(metadata.probability, 4), "text":[], "duration":0.0, "items":[] }
    word_data = {"text":[], "start":0, "duration":0}

    def promote_word(return_data, word_data, time_eos=None):
        word_text = "".join(word_data["text"])
        return_data["items"].append({"text":word_text, "start":word_data["start"], "duration":time_eos})
        if time_eos is None:
            return_data["items"][-1]["duration"] = return_data["items"][-1]["start"] + 0.02 # 20ms chunks
        return_data["items"][-1]["duration"] = round(return_data["items"][-1]["duration"] - return_data["items"][0]["start"], 3)  # adjust for start
        return_data["duration"] = round(return_data["duration"] + return_data["items"][-1]["duration"], 3)  # include last word
        return_data["start"] = return_data["items"][0]["start"]  # utilize start of first
        return_data["text"].append(word_text)  # append last word

    for item_idx in range(metadata.num_items):
        new_item = metadata.items[item_idx]
        start_item = round(new_item.start_time, 3)  # chunks of 20ms time
        if new_item.character == " ":  # found the end of a word, promote
            promote_word(return_data, word_data, start_item)
            word_data = {"text":[], "start":0, "duration":0}
        else:
            if word_data["start"] == 0:
                word_data["start"] = start_item
            word_data["text"].append(new_item.character)

    if word_data["text"]:  # clear out last word in queue
        promote_word(return_data, word_data)
    return_data["text"] = " ".join(return_data["text"])
    return return_data

def data_resample(file_handle):
    """
    Data may not arraive at our native processing sampling rate, so
    resample from input_rate to RATE_PROCESS here for webrtcvad and
    deepspeech
    Args:
        data (binary): Input audio stream
        input_rate (int): Input audio rate to resample from
    """
    input_rate, data = wav_read(file_handle)
    data16 = np.fromstring(string=data, dtype=np.int16)
    resample_size = int(len(data16) / input_rate * AUDIO_RATE)
    resample = signal.resample(data16, resample_size)
    resample16 = np.array(resample, dtype=np.int16)
    return resample16.tostring()


def data_process(data, rich_return=False):
    return_str = None
    # logger.log(5, "got websocket data: %r", data)

    if isinstance(data, bytearray):
        # Receive stream data
        if not server_state["start_time"]:
            # Start of stream (utterance)
            server_state["start_time"] = time()
            server_state["sctx"] = model.setupStream()
            assert not server_state["semaphore_acquired"]
            # logger.debug("acquiring lock for deepspeech ...")
            server_state["semaphore"].acquire(blocking=True)
            server_state["semaphore_acquired"] = True
            # logger.debug("lock acquired")
        model.feedAudioContent(server_state["sctx"], np.frombuffer(data, np.int16))
        return_str = {} if rich_return else ''

    elif isinstance(data, str) and data == 'EOS':
        # End of stream (utterance)
        eos_time = time()
        metadata = regularize_metadata(model.finishStreamWithMetadata(server_state["sctx"]))
        logger.info("recognized: {:}".format(metadata))
        logger.info("recognized: %r", metadata["text"])
        logger.info("    time: total=%s post_eos=%s", time()-server_state["start_time"], time()-eos_time)
        if rich_return:
            return_str = json.dumps(metadata)
        else: 
            return_str = metadata["text"]

        # FIXME: handle ConnectionResetError & geventwebsocket.exceptions.WebSocketError
        # logger.debug("releasing lock ...")
        server_state["semaphore"].release()
        server_state["semaphore_acquired"] = False
        # logger.debug("lock released")
        server_state["start_time"] = None

    else:
        # Lost connection
        logger.debug("dead websocket")
        if server_state["semaphore_acquired"]:
            # logger.debug("releasing lock ...")
            server_state["semaphore"].release()
            server_state["semaphore_acquired"] = False
            # logger.debug("lock released")
    return return_str

@get('/recognize', apply=[websocket])
def recognize(ws):
    logger.debug("new websocket")

    while True:
        data = ws.receive()
        return_str = data_process(data, False)
        if return_str is None:
            break
        else:
            ws.send(return_str)

@get('/recognize_meta', apply=[websocket])
def recognize_meta(ws):
    logger.debug("new websocket")

    while True:
        data = ws.receive()
        return_str = data_process(data, False)
        if return_str is None:
            break
        else:
            ws.send(return_str)

@post('/recognize_file')
def recognize_file():
    enhanced = request.params.get('enhanced', default=False, type=int)
    enhanced = enhanced != False and int(enhanced) != 0  # input normalize
    data_file = request.files.get('file')
    data = data_resample(data_file.file) if data_file is not None else None
    logger.debug(f"file: file: {'(missing file)' if not data_file else data_file.filename}, " + \
                    f"len: {0 if not data_file else data_file.content_length}, " +  \
                    f"len-audio: {0 if not data else len(data)}, enhanced: {enhanced}")
    return_str = ""
    if data:  # only if the data was non-zero
        return_str = data_process(bytearray(data), enhanced)
        return_str = data_process("EOS", enhanced)  # send immediate EOS
    return return_str


@get('/')
def index():
    return template('index')

run(host='0.0.0.0', port=ARGS.port, server=GeventWebSocketServer)

# python server.py --model ../models/daanzu-30330/output_graph.pb --alphabet ../models/daanzu-30330/alphabet.txt --lm ../models/daanzu-30330/lm.binary --trie ../models/daanzu-30330/trie
# python server.py --model ../models/daanzu-30330.2/output_graph.pb --alphabet ../models/daanzu-30330.2/alphabet.txt --lm ../models/daanzu-30330.2/lm.binary --trie ../models/daanzu-30330.2/trie
