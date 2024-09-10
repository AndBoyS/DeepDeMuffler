"""Utilities that modify audio"""

from typing import Any, Self
import numpy as np
from pydub import AudioSegment
from scipy import signal  # type: ignore


class AudioSegmentDual(AudioSegment):
    """
    AudioSegment with convenient numpy.array export and import methods
    """

    def to_numpy(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Export to numpy array

        Returns
        -------
        np.ndarray
            Audio in numpy array
        """
        audio_array = np.array(self.get_array_of_samples())

        return audio_array.reshape((self.channels, -1), order="F")

    def from_numpy(self, audio_array: np.ndarray[Any, np.dtype[Any]]) -> Self:
        """Return new audio segment according to the data in the np.array

        Parameters
        ----------
        audio_array : np.ndarray
            Audio in numpy array

        Returns
        -------
        _type_
            New audio segment instance
        """

        audio_array = audio_array.reshape(-1, order="F")
        new_instance = self._spawn(audio_array.astype("int16"))  # type: ignore
        return new_instance


def firwin_lowpass_filter(
    data: np.ndarray[Any, np.dtype[Any]],
    cutoff: float = 40,
    fs: float = 1600,
    numtaps: int = 10,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Apply firwin lowpass filter to audio

    Parameters
    ----------
    data : np.ndarray
        Audio
    cutoff : float, optional
        _description_, by default 40
    fs : float, optional
        _description_, by default 1600
    numtaps : int, optional
        How strong the filter is, by default 10

    Returns
    -------
    np.ndarray
        Audio
    """
    h = signal.firwin(numtaps=numtaps, cutoff=cutoff, fs=fs)
    data = signal.lfilter(h, 1.0, data)
    return data
