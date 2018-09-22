import argparse, logging

from bottle import get, run, template
from bottle.ext.websocket import GeventWebSocketServer
from bottle.ext.websocket import websocket

import deepspeech
import numpy as np

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='DeepSpeech speech-to-text from microphone')
parser.add_argument('--model', required=True,
                    help='Path to the model (protocol buffer binary file)')
parser.add_argument('--alphabet', required=True,
                    help='Path to the configuration file specifying the alphabet used by the network')
parser.add_argument('--lm', nargs='?',
                    help='Path to the language model binary file')
parser.add_argument('--trie', nargs='?',
                    help='Path to the language model trie file created with native_client/generate_trie')
args = parser.parse_args()

LM_WEIGHT = 1.5
VALID_WORD_COUNT_WEIGHT = 2.25
N_FEATURES = 26
N_CONTEXT = 9
BEAM_WIDTH = 512

print('Initializing model...')

model = deepspeech.Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
if args.lm and args.trie:
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
        else:
            logger.warning("dead websocket")
            break

@get('/')
def index():
    return template('index')

logging.basicConfig(level=10,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )
logging.getLogger().setLevel(20)
run(host='127.0.0.1', port=8080, server=GeventWebSocketServer)

# python server.py --model ../models/daanzu-30330/output_graph.pb --alphabet ../models/daanzu-30330/alphabet.txt --lm ../models/daanzu-30330/lm.binary --trie ../models/daanzu-30330/trie
# python server.py --model ../models/daanzu-30330.2/output_graph.pb --alphabet ../models/daanzu-30330.2/alphabet.txt --lm ../models/daanzu-30330.2/lm.binary --trie ../models/daanzu-30330.2/trie
