def segment_cough(file_path, output_dir, segment_length=3, sample_rate=44100):
    """
    Processes cough audio to ensure at least one representative 3-second segment per ID.
    """
    y, sr = librosa.load(file_path, sr=sample_rate)
    segment_samples = int(segment_length * sr)

    if len(y) <= segment_samples:
        # Pad if the audio is shorter than the desired segment length
        y = pad_or_truncate(y, segment_length, sr)
        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_padded.flac")
        sf.write(segment_path, y, sr)
        print(f"Saved padded cough segment: {segment_path}")
    else:
        # Split into multiple segments
        segments = [y[i:i + segment_samples] for i in range(0, len(y), segment_samples) if len(y[i:i + segment_samples]) == segment_samples]

        if not segments:
            print(f"No valid segments detected for {file_path}. Falling back to default.")
            selected_segment = y[:segment_samples]
        else:
            # Select the most representative segment (highest energy)
            selected_segment = max(segments, key=lambda s: np.sum(s**2))

        os.makedirs(output_dir, exist_ok=True)
        segment_path = os.path.join(output_dir, f"{os.path.basename(file_path).split('.')[0]}_selected.flac")
        sf.write(segment_path, selected_segment, sr)
        print(f"Saved representative or fallback cough segment: {segment_path}")
