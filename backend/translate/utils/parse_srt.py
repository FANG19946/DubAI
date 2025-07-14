def parse_timestamp_ms(timestamp):
    def hms_to_ms(hms_str):
        h, m, s_ms = hms_str.split(":")
        s, ms = s_ms.split(",")
        return (int(h) * 3600000) + (int(m) * 60000) + (int(s) * 1000) + int(ms)
    
    start_str, end_str = timestamp.split(" --> ")
    start_ms = hms_to_ms(start_str.strip())
    end_ms = hms_to_ms(end_str.strip())
    return start_ms, end_ms



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
            start_ms, end_ms = parse_timestamp_ms(timestamp)
            subtitles.append({
                "index": index,
                "timestamp": timestamp,
                "text": text,
                "start": start_ms,
                "end": end_ms
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

