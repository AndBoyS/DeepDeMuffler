import numpy as np
from pydub import AudioSegment
from scipy import signaltt


def firwin_lowpass_filter(
        data: np.ndarray,
        cutoff: float = 40,
        fs: float = 1600,
        numtaps: int = 10,
        ) -> np.ndarray:
    h = signal.firwin(numtaps=numtaps, cutoff=cutoff, fs=fs)
    data = signal.lfilter(h, 1.0, data)
    return data


class AudioSegmentDual(AudioSegment):
    """
    AudioSegment with convenient numpy.array export and import methods
    """

    def to_numpy(self) -> np.ndarray:
        audio_array = self.get_array_of_samples()
        audio_array = np.array(audio_array)

        return audio_array.reshape((self.channels, -1), order='F')

    def from_numpy(self, audio_array: np.ndarray):
        """
        Update self according to the data in the np.array
        """
        audio_array = audio_array.reshape(-1, order='F')
        new_audio = self._spawn(audio_array.astype('int16'))
        self.__dict__ = new_audio.__dict__


def muffle_audio(
        audio: AudioSegmentDual,
        cutoff: float,
        fs: float,
        order: int,
        ):
    audio_array = audio.to_numpy()

    audio_array = firwin_lowpass_filter(
        audio_array,
        cutoff,
        fs,
        order,
    )
    audio.from_numpy(audio_array)
