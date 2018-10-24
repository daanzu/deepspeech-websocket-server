# DeepSpeech WebSocket Server

This is a [WebSocket](https://en.wikipedia.org/wiki/WebSocket) server (& client) for Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech), to allow easy real-time speech recognition, using a separate client & server that can be run in different environments, either locally or remotely.

Work in progress. Developed to quickly test new models running DeepSpeech in [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) using microphone input from host Windows. Available to save others some time.

## Features

* Server
    - Streams raw audio data from client via WebSocket
    - Streaming inference via DeepSpeech v0.2+
    - Single-user (issues with concurrent streams)
* Client
    - Streams raw audio data from microphone to server via WebSocket
    - Voice activity detection (VAD) to ignore noise and segment microphone input into separate utterances

## Installation

This package is developed in Python 3.
Activate a virtualenv, then install the requirements for the server and/or client, depending on usage:

```bash
pip install -r requirements-server.txt
### AND/OR ###
pip install -r requirements-client.txt
```

To run the server in an environment, you also need to install DeepSpeech, which requires choosing either the CPU xor GPU version:

```bash
pip install deepspeech
### XOR ###
pip install deepspeech-gpu
```

Upgrade to the latest DeepSpeech with `pip install deepspeech --upgrade` (or gpu version). This package works with v0.3.0.

## Server

```
> python server.py --model ../models/daanzu-6h-512l-0001lr-425dr/ -l -t
Initializing model...
2018-10-06 AM 05:55:16.357: __main__: INFO: <module>(): args.model: ../models/daanzu-6h-512l-0001lr-425dr/output_graph.pb
2018-10-06 AM 05:55:16.357: __main__: INFO: <module>(): args.alphabet: ../models/daanzu-6h-512l-0001lr-425dr/alphabet.txt
TensorFlow: v1.6.0-18-g5021473
DeepSpeech: v0.2.0-0-g009f9b6
Warning: reading entire model file into memory. Transform model file into an mmapped graph to reduce heap usage.
2018-10-06 05:55:16.358385: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2018-10-06 AM 05:55:16.395: __main__: INFO: <module>(): args.lm: ../models/daanzu-6h-512l-0001lr-425dr/lm.binary
2018-10-06 AM 05:55:16.395: __main__: INFO: <module>(): args.trie: ../models/daanzu-6h-512l-0001lr-425dr/trie
Bottle v0.12.13 server starting up (using GeventWebSocketServer())...
Listening on http://127.0.0.1:8080/
Hit Ctrl-C to quit.

2018-10-06 AM 05:55:30.194: __main__: INFO: echo(): recognized: 'alpha bravo charlie'
2018-10-06 AM 05:55:32.297: __main__: INFO: echo(): recognized: 'delta echo foxtrot'
2018-10-06 AM 05:55:54.747: __main__: INFO: echo(): dead websocket
^CKeyboardInterrupt
```

```
> python server.py -h
usage: server.py [-h] -m MODEL [-a [ALPHABET]] [-l [LM]] [-t [TRIE]] [--lw LW]
                 [--vwcw VWCW] [--bw BW] [-p PORT]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Path to the model (protocol buffer binary file, or
                        directory containing all files for model)
  -a [ALPHABET], --alphabet [ALPHABET]
                        Path to the configuration file specifying the alphabet
                        used by the network. Default: alphabet.txt
  -l [LM], --lm [LM]    Path to the language model binary file. Default:
                        lm.binary
  -t [TRIE], --trie [TRIE]
                        Path to the language model trie file created with
                        native_client/generate_trie. Default: trie
  --lw LW               The alpha hyperparameter of the CTC decoder. Language
                        Model weight. Default: 1.5
  --vwcw VWCW           Valid word insertion weight. This is used to lessen
                        the word insertion penalty when the inserted word is
                        part of the vocabulary. Default: 2.25
  --bw BW               Beam width used in the CTC decoder when building
                        candidate transcriptions. Default: 1024
  -p PORT, --port PORT  Port to run server on. Default: 8080
```

## Client

```
λ py client.py
Listening...
Recognized: alpha bravo charlie
Recognized: delta echo foxtrot
^C
```

```
λ py client.py -h
usage: client.py [-h] [-s SERVER]

optional arguments:
  -h, --help            show this help message and exit
  -s SERVER, --server SERVER
                        Default: ws://localhost:8080/recognize
```
