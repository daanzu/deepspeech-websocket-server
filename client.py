import time, logging
import threading, collections, queue
import pyaudio, wave

logger = logging.getLogger(__name__)

FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 50
BLOCK_SIZE = 80
# BLOCK_SIZE = 2000
# BLOCK_SIZE = RATE / 10
BLOCK_SIZE = int(RATE / float(BLOCKS_PER_SECOND))

class Audio(object):
    def __init__(self, callback=None):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: proxy_callback = None
        self.sample_rate = RATE
        self.block_size = BLOCK_SIZE
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT,
                              channels=CHANNELS,
                              rate=self.sample_rate,
                              input=True,
                              # output=True,
                              frames_per_buffer=self.block_size,
                              stream_callback=proxy_callback)
        self.stream.start_stream()

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()

    frame_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    @staticmethod
    def write_wav(filename, data):
        logger.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data.tostring())
        wf.close()

    @classmethod
    def main(cls, callback):
        audio = cls(proxy_callback)
        while audio.stream.is_active():
            time.sleep(0.1)
        audio.destroy()

class VADAudio(Audio):
    def __init__(self, consumer, aggressiveness):
        super().__init__()
        self.aggressiveness = aggressiveness
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)

        t = threading.Thread(target=consumer, args=(self, self.vad_collector(100),))
        t.start()

    def frame_generator(self):
        while self.stream.is_active():
            yield self.stream.read(self.block_size)

    def vad_collector(self, pre_padding_ms, frames=None):
        if frames is None: frames = self.frame_generator()
        num_padding_frames = pre_padding_ms // self.frame_duration_ms
        buff = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                if is_speech:
                    triggered = True
                    for f in buff:
                        yield f
                    buff.clear()
                    yield frame
                else:
                    buff.append(frame)

            else:
                if is_speech:
                    yield frame
                else:
                    triggered = False
                    yield None
                    buff.append(frame)


from lomond import WebSocket, events
websocket = WebSocket('ws://localhost:8080/websocket')
# TODO: compress?
ready = False

def consumer(self, frames):
    length_ms = 0
    for frame in frames:
        if ready and websocket.is_active:
            if frame is not None:
                logging.log(5, "sending frame")
                websocket.send_binary(frame)
                length_ms += self.frame_duration_ms
            else:
                logging.log(5, "sending EOS")
                logging.info("sent audio length_ms: %d" % length_ms)
                length_ms = 0
                websocket.send_text('EOS')
VADAudio(consumer, 1)

def on_event(event):
    if isinstance(event, events.Ready):
        global ready
        ready = True
    elif isinstance(event, events.Text):
        print("Recognized: %s" % event.text)
    elif 1:
        logging.debug(event)

logging.basicConfig(level=10,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )
logging.getLogger().setLevel(10)
logging.getLogger('lomond').setLevel(30)

for event in websocket:
    try:
        on_event(event)
    except:
        logger.exception('error handling %r', event)
        websocket.close()
