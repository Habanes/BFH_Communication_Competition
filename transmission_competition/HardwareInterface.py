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
        preamble_threshold: float = 0.55,
        postamble_threshold: float = 0.45,
        max_listen_s: float = 60.0,
        block_s: float = 0.08,
        max_rolling_s: float = 2.0,
        min_payload_s: float = 0.5,
        confirm_blocks: int = 2,
        strong_margin: float = 0.15,
        silence_guard_s: float = 0.25,
        silence_ratio: float = 0.25,
        debug: bool = False,
    ) -> np.ndarray:
        block = int(round(float(block_s) * self.fs))

        rolling = np.zeros(0, dtype=np.float32)
        capture = np.zeros(0, dtype=np.float32)

        max_rolling = int(round(float(max_rolling_s) * self.fs))
        max_capture = int(round(float(max_listen_s) * self.fs))
        min_payload = int(round(float(min_payload_s) * self.fs))
        guard = int(round(float(silence_guard_s) * self.fs))

        preamble = np.asarray(preamble, dtype=float)
        postamble = np.asarray(postamble, dtype=float)

        state = "SEARCH_PREAMBLE"
        total_read = 0
        last_report_at = 0
        best_preamble = 0.0
        best_postamble = 0.0
        candidate_hits = 0
        candidate_idx_global = None
        pending_end = None

        with self.create_input_stream(blocksize=block, channels=1, dtype="float32") as stream:
            while True:
                frames, _ = stream.read(block)
                chunk = frames.reshape(-1)
                total_read += len(chunk)

                if total_read >= max_capture and state == "SEARCH_PREAMBLE":
                    if debug:
                        print(f"Timeout waiting for preamble. best_preamble={best_preamble:.3f}")
                    break

                if debug and total_read - last_report_at >= int(self.fs):
                    print(f"preamble_peak={best_preamble:.3f} postamble_peak={best_postamble:.3f} state={state}")
                    last_report_at = total_read

                if state == "SEARCH_PREAMBLE":
                    rolling = np.concatenate([rolling, chunk])
                    if len(rolling) > max_rolling:
                        rolling = rolling[-max_rolling:]

                    # Optimization: Only search the new part of the buffer + overlap
                    search_len = len(chunk) + len(preamble)
                    if len(rolling) >= len(preamble):
                        search_window = rolling[-search_len:] if len(rolling) > search_len else rolling
                        peak, idx_local = self._max_normalized_correlation(search_window, preamble)
                        
                        # Map local index back to rolling buffer index
                        idx = (len(rolling) - len(search_window)) + idx_local
                        
                        if peak > best_preamble:
                            best_preamble = peak
                        if peak >= preamble_threshold:
                            if debug:
                                print(f"Preamble detected. peak={peak:.3f}")
                            capture = rolling[idx:].copy()
                            state = "CAPTURE"

                else:
                    capture = np.concatenate([capture, chunk])
                    if len(capture) > max_capture:
                        if debug:
                            print(f"Timeout waiting for postamble. best_postamble={best_postamble:.3f}")
                        break

                    # Optimization: Only search the new part + overlap
                    search_len = len(chunk) + len(postamble)
                    search_window = capture[-search_len:] if len(capture) > search_len else capture
                    
                    peak, idx_local = self._max_normalized_correlation(search_window, postamble)
                    
                    # Map local index back to capture buffer index
                    idx_global = (len(capture) - len(search_window)) + idx_local
                    
                    if peak > best_postamble:
                        best_postamble = peak

                    if peak >= postamble_threshold and idx_global > (len(preamble) + min_payload):
                        if candidate_idx_global is None or abs(idx_global - candidate_idx_global) <= block:
                            candidate_hits += 1
                            candidate_idx_global = idx_global
                        else:
                            candidate_hits = 1
                            candidate_idx_global = idx_global

                        strong = peak >= (postamble_threshold + float(strong_margin))
                        if strong or candidate_hits >= int(confirm_blocks):
                            end = idx_global + len(postamble)
                            if end <= len(capture):
                                pending_end = end
                    else:
                        candidate_hits = 0
                        candidate_idx_global = None

                    if pending_end is not None and guard > 0 and pending_end + guard <= len(capture):
                        before = capture[max(0, pending_end - guard) : pending_end]
                        after = capture[pending_end : pending_end + guard]

                        rms_before = float(np.sqrt(np.mean(before.astype(float) ** 2))) if before.size else 0.0
                        rms_after = float(np.sqrt(np.mean(after.astype(float) ** 2))) if after.size else 0.0

                        if rms_before > 0 and rms_after <= float(silence_ratio) * rms_before:
                            if debug:
                                print(f"Postamble accepted. peak={peak:.3f} rms_after/rms_before={rms_after/(rms_before+1e-12):.3f}")
                            capture = capture[:pending_end]
                            break
                        else:
                            if debug:
                                ratio = rms_after / (rms_before + 1e-12)
                                print(f"Postamble rejected. peak={peak:.3f} rms_after/rms_before={ratio:.3f}")
                            pending_end = None

        return np.asarray(capture, dtype=np.float32)

    def receive_for_duration(self, duration_s: float, block_s: float = 0.08) -> np.ndarray:
        block = int(round(float(block_s) * self.fs))
        total = int(round(float(duration_s) * self.fs))

        chunks = []
        captured = 0
        with self.create_input_stream(blocksize=block, channels=1, dtype="float32") as stream:
            while captured < total:
                frames, _ = stream.read(block)
                chunk = frames.reshape(-1)
                chunks.append(chunk)
                captured += len(chunk)

        signal = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
        return signal[:total].astype(np.float32)

    def play(self, signal: np.ndarray, blocking: bool = True, normalize: bool = True) -> None:
        audio = np.asarray(signal, dtype=np.float32)
        if normalize and audio.size:
            peak = float(np.max(np.abs(audio)))
            if peak > 0:
                audio = (audio / peak) * 0.9
        sd.play(audio, samplerate=self.fs, blocking=blocking)

    def transmit(self, signal: np.ndarray, blocking: bool = True) -> None:
        self.play(signal, blocking=blocking)
