import argparse, logging, os.path
from time import time

from bottle import get, run, template
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket
from gevent.lock import BoundedSemaphore

import deepspeech
import numpy as np

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

gSem = BoundedSemaphore(1) #Only one Deepspeech instance available at a time

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

@get('/recognize', apply=[websocket])
def recognize(ws):
    logger.debug("new websocket")
    sctx = model.setupStream()
    start_time = None
    logger.debug("acquiring lock for deepspeech ...")
    gSem.acquire(blocking=True)
    gSem_acquired = True
    logger.debug("lock acquired")
    while True:
        data = ws.receive()
        # logger.log(5, "got websocket data: %r", data)
        if isinstance(data, bytearray):
            if not start_time: start_time = time()
            if not gSem_acquired:
                logger.debug("acquiring lock for deepspeech ...")
                gSem.acquire(blocking=True)
                gSem_acquired = True
                logger.debug("lock acquired")
            model.feedAudioContent(sctx, np.frombuffer(data, np.int16))
        elif isinstance(data, str) and data == 'EOS':
            eos_time = time()
            text = model.finishStream(sctx)
            logger.info("recognized: %r", text)
            logger.debug("    time: total=%s post_eos=%s", time()-start_time, time()-eos_time)
            ws.send(text)
            # FIXME: handle ConnectionResetError & geventwebsocket.exceptions.WebSocketError
            sctx = model.setupStream()
            logger.debug("releasing lock ...")
            gSem.release()
            gSem_acquired = False
            logger.debug("lock released")
            start_time = None
        else:
            logger.debug("dead websocket")
            try:
                logger.debug("releasing lock ...")
                gSem.release()
                gSem_acquired = False
                logger.debug("lock released")
            except ValueError:
                logger.debug("Overrelease error: failed to release semaphore, already released!")
            break

@get('/')
def index():
    return template('index')

run(host='127.0.0.1', port=ARGS.port, server=GeventWebSocketServer)

# python server.py --model ../models/daanzu-30330/output_graph.pb --alphabet ../models/daanzu-30330/alphabet.txt --lm ../models/daanzu-30330/lm.binary --trie ../models/daanzu-30330/trie
# python server.py --model ../models/daanzu-30330.2/output_graph.pb --alphabet ../models/daanzu-30330.2/alphabet.txt --lm ../models/daanzu-30330.2/lm.binary --trie ../models/daanzu-30330.2/trie
