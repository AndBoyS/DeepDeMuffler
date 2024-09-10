"""Create dataset with muffled audio"""

from pathlib import Path
from tqdm import tqdm

from src.data_utils import AudioSegmentDual, firwin_lowpass_filter


def main() -> None:
    """
    Entry point
    """
    # Filter settings
    cutoff = 40
    fs = 1600
    filter_strength = 15

    base_dir = Path().resolve().parent
    raw_data_dir = base_dir / "data/raw"
    train_data_dir = base_dir / "data/muffled"

    train_data_dir.mkdir(exist_ok=True)

    fps = filter(lambda x: x.suffix in [".mp3", ".wav"], raw_data_dir.glob("*"))

    for fp in tqdm(list(fps)):
        output_fp = train_data_dir / fp.with_suffix(".wav").name

        audio = AudioSegmentDual.from_mp3(fp)
        audio_array = audio.to_numpy()

        audio_array = firwin_lowpass_filter(
            audio_array,
            cutoff=cutoff,
            fs=fs,
            numtaps=filter_strength,
        )
        audio = audio.from_numpy(audio_array)
        audio.export(output_fp, format="wav")


if __name__ == "__main__":
    main()
