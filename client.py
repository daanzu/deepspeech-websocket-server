import time, logging
import threading, queue
import pyaudio, wave

logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 10
BLOCK_SIZE = 80
# BLOCK_SIZE = 2000
# BLOCK_SIZE = RATE / 10
BLOCK_SIZE = int(RATE / float(BLOCKS_PER_SECOND))

class Audio(object):
    def __init__(self, callback):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=RATE,
                              input=True,
                              # output=True,
                              frames_per_buffer=BLOCK_SIZE,
                              stream_callback=callback)
        self.stream.start_stream()
    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    @staticmethod
    def write_wav(filename, data):
        logger.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(data.tostring())
        wf.close()

    @classmethod
    def main(cls, callback):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)
        audio = cls(proxy_callback)
        while audio.stream.is_active():
            time.sleep(0.1)
        audio.destroy()

    @classmethod
    def main_threaded(cls, callback):
        t = threading.Thread(target=cls.main, args=(callback,))
        t.start()
        return t

from lomond import WebSocket, events
websocket = WebSocket('ws://localhost:8080/websocket')
# TODO: compress?
ready = False

q = queue.Queue(maxsize=50)
i = 0
def callback(data):
    # q.put(data)
    if ready and websocket.is_active:
        websocket.send_binary(data)
        global i
        i += 1
        print(i)
        if i > 30:
            websocket.send_text('EOS')
            websocket.close()
Audio.main_threaded(callback)

def on_event(event):
    if isinstance(event, events.Ready):
        global ready
        ready = True
    elif isinstance(event, events.Text):
        print(event.text)
    elif 1:
        print(event)

for event in websocket:
    try:
        on_event(event)
    except:
        logger.exception('error handling %r', event)
        websocket.close()
