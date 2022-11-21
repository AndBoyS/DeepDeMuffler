import numpy as np
from pydub import AudioSegment
from scipy import signal


def butter_lowpass_filter(
        data: np.ndarray,
        cutoff: float,
        fs: float,
        order: int = 5
        ) -> np.ndarray:
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    sos = signal.butter(order, normal_cutoff, btype='low', analog=False, output='sos')
    data = signal.sosfilt(sos, data)
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

    audio_array = butter_lowpass_filter(
        audio_array,
        cutoff=cutoff,
        fs=fs,
        order=order,
    )

    audio.from_numpy(audio_array)
