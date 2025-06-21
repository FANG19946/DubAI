def parse_srt(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    entries = content.split('\n\n')  # Separate blocks
    subtitles = []

    for entry in entries:
        lines = entry.strip().split('\n')
        if len(lines) >= 3:
            index = int(lines[0])
            timestamp = lines[1]
            text = ' '.join(lines[2:]).strip()
            subtitles.append({
                "index": index,
                "timestamp": timestamp,
                "text": text
            })

    return subtitles

def write_srt_from_parsed(subtitles, output_path):
    """
    subtitles: list of dicts with 'index', 'timestamp', 'text'
    output_path: path to save the translated .srt
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in subtitles:
            f.write(f"{entry['index']}\n")
            f.write(f"{entry['timestamp']}\n")
            f.write(f"{entry['text']}\n\n")
