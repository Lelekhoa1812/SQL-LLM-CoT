# rotator.py
import os, logging, itertools, threading, time
from google import genai
from google.api_core import exceptions as gexc

log = logging.getLogger("gemini-rotator")
MODEL = "gemini-2.5-flash-preview-04-17"

# This is a safe wrapper
class GeminiModelWrapper:
    def __init__(self, parent):
        self.parent = parent

    def generate_content(self, *args, **kw):
        model_name = kw.pop("model", MODEL)
        is_stream = kw.pop("stream", False)
        for attempt in range(5):
            try:
                client = self.parent._client
                if is_stream:
                    return client.models.generate_content(model=model_name, stream=True, *args, **kw)
                else:
                    return client.models.generate_content(model=model_name, *args, **kw)
            except Exception as e:
                if self.parent._should_rotate(e):
                    log.warning(f"[Retry {attempt+1}/5] Rotating Gemini key due to: {e}")
                    self.parent._swap_key()
                    time.sleep(1)
                    continue
                raise

    def count_tokens(self, *args, **kwargs):
        return self.parent._client.count_tokens(*args, **kwargs)

class RotatingGeminiClient:
    """
    Wraps google.generativeai Client with automatic key-rotation.
    • Pass any kwargs that genai.Client accepts to __init__.
    • Call `.generate_content(model, contents, **kw)` exactly
      like you would on a normal client.
    """
    def __init__(self, env_prefix: str = "GEMINI_FLASH_API_KEY"):
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
            self._client = genai.Client(api_key=key)

    def _should_rotate(self, exc: Exception) -> bool:
        return isinstance(exc, (gexc.ResourceExhausted, gexc.PermissionDenied))


    # ---------- public proxy ----------------------------------------
    def generate_content(self, *args, **kw):
        return self.models.generate_content(*args, **kw)
    
    # convenience passthrough for other genai methods you might need
    def __getattr__(self, item):
        return getattr(self._client, item)
