# rotator.py
import os, logging, itertools, threading, time
from google import genai
from google.api_core import exceptions as gexc

log = logging.getLogger("gemini-rotator")

# This is a safe wrapper
class GeminiModelWrapper:
    def __init__(self, parent):
        self.parent = parent

    def generate_content(self, *args, **kw): # Enable parallel workers
        is_stream = kw.get("stream", False)
        if "stream" in kw:
            del kw["stream"]
        for attempt in range(5):
            try:
                if is_stream:
                    return self._client.models.stream_generate_content(*args, **kw)
                else:
                    return self._client.models.generate_content(*args, **kw)
            except Exception as e:
                if self._should_rotate(e):
                    log.warning(f"[Retry {attempt+1}/5] Rotating key due to: {e}")
                    self._swap_key()
                    time.sleep(1)
                    continue
                raise


    def count_tokens(self, *args, **kwargs):
        return self.parent.count_tokens(*args, **kwargs)

class RotatingGeminiClient:
    """
    Wraps google.generativeai Client with automatic key-rotation.
    • Pass any kwargs that genai.Client accepts to __init__.
    • Call `.generate_content(model, contents, **kw)` exactly
      like you would on a normal client.
    """
    def __init__(self, env_prefix: str = "GEMINI_FLASH_API_KEY"):
        # collect GEMINI_FLASH_API_KEY_1 … _N  (or single key without suffix)
        keys = [
            val for name, val in os.environ.items()
            if name.startswith(env_prefix) and val
        ]
        if not keys:
            raise RuntimeError("No Gemini API keys found")
        self._lock = threading.RLock()
        self._cycle = itertools.cycle(keys)
        self._client = genai.Client(api_key=next(self._cycle))

    @property
    def models(self):
        return GeminiModelWrapper(self)

    
    # ---------- internal helpers ------------------------------------
    def _swap_key(self):
        with self._lock:
            key = next(self._cycle)
            log.warning("🔄 Switching Gemini key")
            self._client = genai.Client(api_key=key)

    def _should_rotate(self, exc: Exception) -> bool:
        # 429 RESOURCE_EXHAUSTED or PERMISSION_DENIED indicate exhausted / disabled key
        if isinstance(exc, gexc.ResourceExhausted):
            return True
        if isinstance(exc, gexc.PermissionDenied):
            return True
        return False

    # ---------- public proxy ----------------------------------------
    def generate_content(self, *args, **kw):
        for attempt in range(5):           # rotate up to 5 keys max
            try:
                return self._client.models.generate_content(*args, **kw)
            except Exception as e:
                if self._should_rotate(e):
                    self._swap_key()
                    time.sleep(1)          # small delay before retry
                    continue
                raise

    # convenience passthrough for other genai methods you might need
    def __getattr__(self, item):
        return getattr(self._client, item)
