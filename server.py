import argparse, logging, os.path

from bottle import get, run, template
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket

import deepspeech
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=10,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )
logging.getLogger().setLevel(20)

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
ARGS = parser.parse_args()

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
    while True:
        data = ws.receive()
        logger.debug("got websocket data: %r", data)
        if isinstance(data, bytearray):
            model.feedAudioContent(sctx, np.frombuffer(data, np.int16))
        elif isinstance(data, str) and data == 'EOS':
            text = model.finishStream(sctx)
            logger.info("recognized: %r", text)
            ws.send(text)
            # FIXME: handle ConnectionResetError & geventwebsocket.exceptions.WebSocketError
            sctx = model.setupStream()
        else:
            logger.info("dead websocket")
            break

@get('/')
def index():
    return template('index')

run(host='127.0.0.1', port=ARGS.port, server=GeventWebSocketServer)

# python server.py --model ../models/daanzu-30330/output_graph.pb --alphabet ../models/daanzu-30330/alphabet.txt --lm ../models/daanzu-30330/lm.binary --trie ../models/daanzu-30330/trie
# python server.py --model ../models/daanzu-30330.2/output_graph.pb --alphabet ../models/daanzu-30330.2/alphabet.txt --lm ../models/daanzu-30330.2/lm.binary --trie ../models/daanzu-30330.2/trie
