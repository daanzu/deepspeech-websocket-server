import time, logging
from datetime import datetime
import threading, collections, queue, os, os.path
import wave
import pyaudio
from lomond import WebSocket, events
from halo import Halo

logger = logging.getLogger(__name__)
logging.basicConfig(level=30,
    format="%(asctime)s.%(msecs)03d: %(name)s: %(levelname)s: %(funcName)s(): %(message)s",
    datefmt="%Y-%m-%d %p %I:%M:%S",
    )
logging.getLogger('lomond').setLevel(30)

FORMAT = pyaudio.paInt16
RATE = 16000
CHANNELS = 1
BLOCKS_PER_SECOND = 50
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

    def write_wav(self, filename, data):
        logger.info("write wav %s", filename)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        # wf.setsampwidth(self.pa.get_sample_size(FORMAT))
        assert FORMAT == pyaudio.paInt16
        wf.setsampwidth(2)
        wf.setframerate(self.sample_rate)
        wf.writeframes(data)
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
        import webrtcvad
        self.vad = webrtcvad.Vad(aggressiveness)
        if consumer:
            t = threading.Thread(target=consumer, args=(self, self.vad_collector(300),))
            t.start()

    def frame_generator(self):
        while self.stream.is_active():
            yield self.stream.read(self.block_size)

    def vad_collector_simple(self, pre_padding_ms, frames=None):
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

    def vad_collector(self, padding_ms, ratio=0.75, frames=None):
        if frames is None: frames = self.frame_generator()
        num_padding_frames = padding_ms // self.frame_duration_ms
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
    def vad_test(cls, aggressiveness):
        self = cls(aggressiveness=aggressiveness)
        frames = self.frame_generator()
        for frame in frames:
            is_speech = self.vad.is_speech(frame, self.sample_rate)
            print('|' if is_speech else '.', end='', flush=True)


def main_test():
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
        VADAudio.vad_test(3)


def main():
    websocket = WebSocket(ARGS.server)
    # TODO: compress?
    print("Connecting to '%s'..." % websocket.url)
    ready = False

    def consumer(self, frames):
        spinner = None
        if not ARGS.nospinner: spinner = Halo(spinner='line') # circleHalves point arc boxBounce2 bounce line
        length_ms = 0
        wav_data = bytearray()
        for frame in frames:
            if ready and websocket.is_active:
                if frame is not None:
                    if not length_ms:
                        logging.debug("begin utterence")
                    if spinner: spinner.start()
                    logging.log(5, "sending frame")
                    websocket.send_binary(frame)
                    if ARGS.savewav: wav_data.extend(frame)
                    length_ms += self.frame_duration_ms
                else:
                    if spinner: spinner.stop()
                    if not length_ms: raise RuntimeError("ended utterence without beginning")
                    logging.debug("end utterence")
                    if ARGS.savewav: 
                        self.write_wav(os.path.join(ARGS.savewav, datetime.now().strftime("savewav_%Y-%m-%d_%H-%M-%S_%f.wav")), wav_data)
                        wav_data = bytearray()
                    logging.info("sent audio length_ms: %d" % length_ms)
                    logging.log(5, "sending EOS")
                    websocket.send_text('EOS')
                    length_ms = 0

    VADAudio(consumer, aggressiveness=ARGS.aggressiveness)
    print("Listening...")

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
        help="Save .wav files of utterences to given directory")
    global ARGS
    ARGS = parser.parse_args()
    # logging.getLogger().setLevel(10)

    if ARGS.savewav: os.makedirs(ARGS.savewav, exist_ok=True)

    if 0:
        main_test()
    else:
        main()
