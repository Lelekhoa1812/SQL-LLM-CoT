# translation.py
# translation.py
import os, logging, json, re, asyncio
from google import genai   
from llm_ut import retry_with_backoff

log = logging.getLogger("translation-gemini")
log.info("🚀 Translator module Gemini boost up...")

log = logging.getLogger("translation")
G_API_KEY = os.getenv("GEMINI_FLASH_API_KEY")
if not G_API_KEY:
    raise RuntimeError("⚠️  GEMINI_FLASH_API_KEY env-var is missing")
llm_client = genai.Client(G_API_KEY)

# ------- helpers ------------------------------------------------------------
MODEL = "gemini-2.5-flash-preview-04-17"

def _clean_md(text: str) -> str:
    """Remove ``` fences if model wrapped the answer."""
    if "```" in text:
        m = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if m:
            return m[0].strip()
    return text.strip()

@retry_with_backoff(retries=4, delay=1.5) # Retry if error persist
def _gemini(prompt: str, temperature: float = .7) -> str:
    try:
        rsp = llm_client.models.generate_content(
            model=MODEL,
            contents=prompt)
        return _clean_md(rsp.text)
    except Exception as e:
        log.error(f"[Gemini] translation failure: {e}")
        raise

# ------- public API ---------------------------------------------------------
def vie_to_en(vie: str) -> str:
    prompt = (
        "You are an accurate Vietnamese→English translator.\n"
        "Translate the following text to fluent English **without losing any meaning** and contextual maintained:\n\n"
        f"{vie}"
    )
    res = _gemini(prompt)
    log.info(f"[Gemini Vi-En] translation to: {res}")
    return res

def en_to_vie(en: str) -> str:
    prompt = (
        "You are an accurate English→Vietnamese translator.\n"
        "Translate the following text to fluent Vietnamese, contextual maintained (markdown formatted):\n\n"
        f"{en}"
    )
    res = _gemini(prompt)
    log.info(f"[Gemini En-Vi] translation to: {res}")
    return res

# ---------- async wrappers (for Qwen) ----------------------------------------
async def a_vie_to_en(txt: str) -> str:
    return await asyncio.to_thread(vie_to_en, txt)

async def a_en_to_vie(txt: str) -> str:
    return await asyncio.to_thread(en_to_vie, txt)
