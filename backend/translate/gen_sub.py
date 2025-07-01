from translate_utils import translate_subtitles_to_hindi
from parse_srt import parse_srt, write_srt_from_parsed  # assumes you have this already

def main():
    input_srt = "test_sample2.srt"
    output_srt = "test_sample2_translated.srt"

    # Parse original English subtitles
    subs = parse_srt(input_srt)

    # Translate to Hindi
    translated_subs = translate_subtitles_to_hindi(subs)

    # Save to new .srt file
    write_srt_from_parsed(translated_subs, output_srt)

    print(f"✅ Translated subtitles saved to {output_srt}")

if __name__ == "__main__":
    main()
