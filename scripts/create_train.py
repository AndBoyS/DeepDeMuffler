from pathlib import Path
from tqdm import tqdm

from deep_demuffle.data_utils import AudioSegmentDual, muffle_audio


def main():
    # Filter settings
    cutoff = 3.667
    fs = 30.0
    order = 6

    base_dir = Path().resolve().parent

    raw_data_dir = base_dir / 'data/raw'
    train_data_dir = base_dir / 'data/train'
    train_data_dir.mkdir(exist_ok=True)

    fps = list(raw_data_dir.glob('*'))
    for fp in tqdm(fps):
        output_fp = train_data_dir / fp.name.replace('mp3', 'wav')

        audio = AudioSegmentDual.from_mp3(fp)
        muffle_audio(audio, cutoff, fs, order)
        audio.export(output_fp, format='wav')


if __name__ == '__main__':
    main()
