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
parser.add_argument('-m', '--model', required=True, default='output_graph.pb',
                    help='Path to the model (protocol buffer binary file, or directory containing all files for model)')
parser.add_argument('-a', '--alphabet', nargs='?', const='alphabet.txt',
                    help='Path to the configuration file specifying the alphabet used by the network')
parser.add_argument('-l', '--lm', nargs='?', const='lm.binary',
                    help='Path to the language model binary file')
parser.add_argument('-t', '--trie', nargs='?', const='trie',
                    help='Path to the language model trie file created with native_client/generate_trie')
parser.add_argument('--lw', type=float, default=1.5,
                    help='The alpha hyperparameter of the CTC decoder. Language Model weight')
parser.add_argument('--vwcw', type=float, default=2.25,
                    help='Valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary')
parser.add_argument('--bw', type=int, default=1024,
                    help='Beam width used in the CTC decoder when building candidate transcriptions')
args = parser.parse_args()

if os.path.isdir(args.model):
    model_dir = args.model
    args.model = os.path.join(model_dir, 'output_graph.pb')
    args.alphabet = os.path.join(model_dir, args.alphabet if args.alphabet else 'alphabet.txt')
    if args.lm: args.lm = os.path.join(model_dir, args.lm)
    if args.trie: args.trie = os.path.join(model_dir, args.trie)

LM_WEIGHT = args.lw
VALID_WORD_COUNT_WEIGHT = args.vwcw
BEAM_WIDTH = args.bw
N_FEATURES = 26
N_CONTEXT = 9

print('Initializing model...')
logger.info("args.model: %s", args.model)
logger.info("args.alphabet: %s", args.alphabet)

model = deepspeech.Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
if args.lm and args.trie:
    logger.info("args.lm: %s", args.lm)
    logger.info("args.trie: %s", args.trie)
    model.enableDecoderWithLM(args.alphabet,
                              args.lm,
                              args.trie,
                              LM_WEIGHT,
                              VALID_WORD_COUNT_WEIGHT)

@get('/websocket', apply=[websocket])
def echo(ws):
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
            sctx = model.setupStream()
        else:
            logger.info("dead websocket")
            break

@get('/')
def index():
    return template('index')

run(host='127.0.0.1', port=8080, server=GeventWebSocketServer)

# python server.py --model ../models/daanzu-30330/output_graph.pb --alphabet ../models/daanzu-30330/alphabet.txt --lm ../models/daanzu-30330/lm.binary --trie ../models/daanzu-30330/trie
# python server.py --model ../models/daanzu-30330.2/output_graph.pb --alphabet ../models/daanzu-30330.2/alphabet.txt --lm ../models/daanzu-30330.2/lm.binary --trie ../models/daanzu-30330.2/trie
