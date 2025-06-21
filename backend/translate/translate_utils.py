import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
BATCH_SIZE = 4

# === Load model + tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True).to(DEVICE)
model.eval()

# === Load IndicProcessor === #
ip = IndicProcessor(inference=True)


def batch_translate(lines):
    """
    Translates a list of lines from SRC_LANG to TGT_LANG
    """
    translations = []
    for i in range(0, len(lines), BATCH_SIZE):
        batch = lines[i:i + BATCH_SIZE]
        batch = ip.preprocess_batch(batch, src_lang=SRC_LANG, tgt_lang=TGT_LANG)

        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=128,
                num_beams=5,
                num_return_sequences=1
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned = ip.postprocess_batch(decoded, lang=TGT_LANG)
        translations.extend(cleaned)

    return translations


def translate_subtitles_to_hindi(subtitles):
    """
    subtitles: list of dicts with keys: index, timestamp, text
    Returns same list with translated 'text' field
    """
    lines = [entry['text'] for entry in subtitles]
    translated_lines = batch_translate(lines)

    for i, entry in enumerate(subtitles):
        entry['text'] = translated_lines[i]

    return subtitles
