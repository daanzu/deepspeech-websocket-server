# DeepSpeech WebSocket Server

This is a [WebSocket](https://en.wikipedia.org/wiki/WebSocket) server (& client) for Mozilla's [DeepSpeech](https://github.com/mozilla/DeepSpeech), to allow easy real-time speech recognition, using a separate client & server that can be run in different environments, either locally or remotely.

Work in progress. Developed to quickly test new DeepSpeech models running DeepSpeech in [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) using microphone input from host Windows.

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

```
pip install -r requirements-server.txt
### AND/OR ###
pip install -r requirements-client.txt
```

To run the server in an environment, you also need to install DeepSpeech, which requires choosing either the CPU xor GPU version:

```
pip install deepspeech
### XOR ###
pip install deepspeech-gpu
```

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
```

## Client

```
λ py client.py
Listening...
Recognized: alpha bravo charlie
Recognized: delta echo foxtrot
```
