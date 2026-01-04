"""Audio in/out put
"""

import numpy as np
import sounddevice as sd


class HardwareInterface:
    def __init__(self, fs: float = 48000.0):
        self.fs = float(fs)

    def _max_normalized_correlation(self, x: np.ndarray, template: np.ndarray) -> tuple[float, int]:
        x = np.asarray(x, dtype=float)
        template = np.asarray(template, dtype=float)

        L = len(template)
        if len(x) < L or L == 0:
            return 0.0, 0

        dots = np.correlate(x, template, mode="valid")
        template_energy = float(np.dot(template, template)) + 1e-12
        seg_energy = np.convolve(x * x, np.ones(L, dtype=float), mode="valid") + 1e-12

        corr = np.abs(dots) / np.sqrt(template_energy * seg_energy)
        idx = int(np.argmax(corr))
        return float(corr[idx]), idx

    def create_input_stream(self, blocksize: int, channels: int = 1, dtype: str = "float32", device=None):
        return sd.InputStream(
            samplerate=self.fs,
            channels=channels,
            dtype=dtype,
            blocksize=int(blocksize),
            device=device,
        )

    def receive_preamble_postamble(
        self,
        preamble: np.ndarray,
        postamble: np.ndarray,
        preamble_threshold: float = 0.70,
        postamble_threshold: float = 0.70,
        max_listen_s: float = 60.0,
        block_s: float = 0.08,
        max_rolling_s: float = 2.0,
    ) -> np.ndarray:
        block = int(round(float(block_s) * self.fs))

        rolling = np.zeros(0, dtype=np.float32)
        capture = np.zeros(0, dtype=np.float32)

        max_rolling = int(round(float(max_rolling_s) * self.fs))
        max_capture = int(round(float(max_listen_s) * self.fs))

        preamble = np.asarray(preamble, dtype=float)
        postamble = np.asarray(postamble, dtype=float)

        state = "SEARCH_PREAMBLE"

        with self.create_input_stream(blocksize=block, channels=1, dtype="float32") as stream:
            while True:
                frames, _ = stream.read(block)
                chunk = frames.reshape(-1)

                if state == "SEARCH_PREAMBLE":
                    rolling = np.concatenate([rolling, chunk])
                    if len(rolling) > max_rolling:
                        rolling = rolling[-max_rolling:]

                    peak, idx = self._max_normalized_correlation(rolling, preamble)
                    if peak >= preamble_threshold:
                        capture = rolling[idx:].copy()
                        state = "CAPTURE"

                else:
                    capture = np.concatenate([capture, chunk])
                    if len(capture) > max_capture:
                        break

                    tail = capture[-max_rolling:]
                    peak, idx = self._max_normalized_correlation(tail, postamble)
                    idx_global = (len(capture) - len(tail)) + idx

                    if peak >= postamble_threshold and idx_global > len(preamble):
                        end = idx_global + len(postamble)
                        if end <= len(capture):
                            capture = capture[:end]
                            break

        return np.asarray(capture, dtype=np.float32)

    def play(self, signal: np.ndarray, blocking: bool = True, normalize: bool = True) -> None:
        audio = np.asarray(signal, dtype=np.float32)
        if normalize and audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = (audio / peak) * 0.9
        sd.play(audio, samplerate=self.fs, blocking=blocking)

    def transmit(self, signal: np.ndarray, blocking: bool = True) -> None:
        self.play(signal, blocking=blocking)
