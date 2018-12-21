import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import wave
import pyaudio
import webrtcvad
from lomond import WebSocket, events
from halo import Halo

logger = logging.getLogger(__name__)
logging.basicConfig(level=30,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )
logging.getLogger('lomond').setLevel(30)

class Audio(object):
    """Streams raw audio from microphone. Data is received in a separate thread, and stored in a buffer, to be read from."""

    FORMAT = pyaudio.paInt16
    RATE = 16000
    CHANNELS = 1
    BLOCKS_PER_SECOND = 50

    def __init__(self, callback=None, buffer_s=0, flush_queue=True):
        def proxy_callback(in_data, frame_count, time_info, status):
            callback(in_data)
            return (None, pyaudio.paContinue)
        if callback is None: callback = lambda in_data: self.buffer_queue.put(in_data, block=False)
        self.sample_rate = self.RATE
        self.flush_queue = flush_queue
        self.buffer_queue = queue.Queue(maxsize=(buffer_s * 1000 // self.block_duration_ms))
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=self.FORMAT,
                                   channels=self.CHANNELS,
                                   rate=self.sample_rate,
                                   input=True,
                                   frames_per_buffer=self.block_size,
                                   stream_callback=proxy_callback)
        self.stream.start_stream()
        self.active = True

    def destroy(self):
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        self.active = False

    def read(self):
        """Return a block of audio data, blocking if necessary."""
        if self.active or (self.flush_queue and not self.buffer_queue.empty()):
            return self.buffer_queue.get()
        else:
            return None

    def read_loop(self, callback):
        """Block looping reading, repeatedly passing a block of audio data to callback."""
        for block in iter(self):
            callback(block)

    def __iter__(self):
        """Generator that yields all audio frames from microphone."""
        while True:
            block = self.read()
            if block is None:
                break
            yield block

    block_size = property(lambda self: int(self.sample_rate / float(self.BLOCKS_PER_SECOND)))
    block_duration_ms = property(lambda self: 1000 * self.block_size // self.sample_rate)

    def write_wav(self, filename, data):
        logging.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert self.FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
        wf.close()

class VADAudio(Audio):
    """Filter & segment audio with voice activity detection."""

    def __init__(self, aggressiveness=3):
        super().__init__()
        self.vad = webrtcvad.Vad(aggressiveness)

    def vad_collector_simple(self, pre_padding_ms, frames=None):
        if frames is None: frames = iter(self)
        num_padding_frames = padding_ms // self.block_duration_ms
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

    def vad_collector(self, padding_ms=300, ratio=0.75, frames=None):
        """Generator that yields series of consecutive audio frames comprising each utterence, separated by yielding a single None.
            Determines voice activity by ratio of frames in padding_ms. Uses a buffer to include padding_ms prior to being triggered.
            Example: (frame, ..., frame, None, frame, ..., frame, None, ...)
                      |---utterence---|        |---utterence---|
        """
        if frames is None: frames = iter(self)
        num_padding_frames = padding_ms // self.block_duration_ms
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False

        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)

            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > ratio * ring_buffer.maxlen:
                    triggered = True
                    for f, s in ring_buffer:
                        yield f
                    ring_buffer.clear()

            else:
                yield frame
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > ratio * ring_buffer.maxlen:
                    triggered = False
                    yield None
                    ring_buffer.clear()

    @classmethod
    def test_vad(cls, aggressiveness):
        self = cls(aggressiveness=aggressiveness)
        frames = iter(self)
        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            print('|' if is_speech else '.', end='', flush=True)


def main_test(ARGS):
    if 0:
        def consumer(self, frames):
            length_ms = 0
            for frame in frames:
                if frame is not None:
                    print('|', end='', flush=True)
                    length_ms += self.frame_duration_ms
                else:
                    print('.', end='', flush=True)
                    length_ms = 0
        VADAudio(consumer)
    elif 1:
        VADAudio.test_vad(3)

def main(ARGS):
    websocket = WebSocket(ARGS.server)
    # TODO: compress?
    print("Connecting to '%s'..." % websocket.url)
    ready = False

    def consumer(vad_audio):
        spinner = None
        if not ARGS.nospinner: spinner = Halo(spinner='line') # circleHalves point arc boxBounce2 bounce line
        length_ms = 0
        wav_data = bytearray()
        for frame in vad_audio.vad_collector():
            if ready and websocket.is_active:
                if frame is not None:
                    if not length_ms:
                        logging.debug("begin utterence")
                    if spinner: spinner.start()
                    logging.log(5, "sending frame")
                    websocket.send_binary(frame)
                    if ARGS.savewav: wav_data.extend(frame)
                    length_ms += vad_audio.block_duration_ms
                else:
                    if spinner: spinner.stop()
                    if not length_ms: raise RuntimeError("ended utterence without beginning")
                    logging.debug("end utterence")
                    if ARGS.savewav:
                        vad_audio.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                        wav_data = bytearray()
                    logging.info("sent audio length_ms: %d" % length_ms)
                    logging.log(5, "sending EOS")
                    websocket.send_text('EOS')
                    length_ms = 0

    vad_audio = VADAudio(aggressiveness=ARGS.aggressiveness)
    print("Listening (ctrl-C to exit)...")
    consumer_thread = threading.Thread(target=lambda: consumer(vad_audio))
    consumer_thread.start()

    def on_event(event):
        if isinstance(event, events.Ready):
            nonlocal ready
            if not ready:
                print("Connected!")
            ready = True
        elif isinstance(event, events.Text):
            if 1: print("Recognized: %s" % event.text)
        elif 1:
            logging.debug(event)

    for event in websocket:
        try:
            on_event(event)
        except:
            logger.exception('error handling %r', event)
            websocket.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Streams raw audio data from microphone with VAD to server via WebSocket")
    parser.add_argument('-s', '--server', default='ws://localhost:8080/recognize',
        help="Default: ws://localhost:8080/recognize")
    parser.add_argument('-a', '--aggressiveness', type=int, default=3,
        help="Set aggressiveness of VAD: an integer between 0 and 3, 0 being the least aggressive about filtering out non-speech, 3 the most aggressive. Default: 3")
    parser.add_argument('--nospinner', action='store_true',
        help="Disable spinner")
    parser.add_argument('-w', '--savewav',
        help="Save .wav files of utterences to given directory. Example for current directory: -w .")
    parser.add_argument('-v', '--verbose', action='store_true',
        help="Print debugging info")
    ARGS = parser.parse_args()

    if ARGS.verbose: logging.getLogger().setLevel(10)
    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)

    if 0:
        main_test(ARGS)
    else:
        main(ARGS)
