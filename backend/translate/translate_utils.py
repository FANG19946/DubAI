import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from transformers import BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
QUANTIZATION = ""  # Options: "", "4-bit", "8-bit"
USE_FLASH_ATTENTION = True


MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
BATCH_SIZE = 4

def get_attention_impl():
    if USE_FLASH_ATTENTION:
        if is_flash_attn_2_available() and is_flash_attn_greater_or_equal_2_10():
            print("Using Flash Attention 2")
            return "flash_attention_2"
    return "eager"


def get_quant_config():
    if QUANTIZATION == "4-bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif QUANTIZATION == "8-bit":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_compute_dtype=torch.bfloat16,
        )
    return None


# === Load model + tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
quant_config = get_quant_config()
attn_impl = get_attention_impl()

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    quantization_config=quant_config,
    attn_implementation=attn_impl,
    low_cpu_mem_usage=True,
)

if quant_config is None:
    model = model.to(DEVICE)
    model.half()

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
