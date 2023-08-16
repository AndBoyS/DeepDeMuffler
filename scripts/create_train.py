from pathlib import Path
from tqdm import tqdm

from deep_demuffle.data_utils import AudioSegmentDual, muffle_audio


def main():
    # Filter settings
    cutoff = 40
    fs = 1600
    # How strong the filter is
    order = 15

    base_dir = Path().resolve().parent
    raw_data_dir = base_dir / 'data/raw'
    train_data_dir = base_dir / 'data/muffled'

    train_data_dir.mkdir(exist_ok=True)

    fps = filter(lambda x: x.suffix in ['.mp3', '.wav'],
                 raw_data_dir.glob('*'))

    for fp in tqdm(list(fps)):

        output_fp = train_data_dir / fp.with_suffix('.wav').name

        audio = AudioSegmentDual.from_mp3(fp)
        muffle_audio(audio, cutoff, fs, order)
        audio.export(output_fp, format='wav')


if __name__ == '__main__':
    main()
