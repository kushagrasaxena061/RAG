import base64
import re
import unicodedata

class SecuritySanitizer:
    MALICIOUS_PATTERNS = [
        r"(?i)ignore\s+(?:all\s+)?previous\s+(?:instructions|directions|prompts)",
        r"(?i)system\s+prompt",
        r"(?i)you\s+are\s+now\s+(?:a\s+)?(?:developer|admin|root)",
        r"(?i)bypassing\s+(?:all\s+)?controls"
    ]

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        # Normalize Unicode to prevent obfuscation
        normalized = unicodedata.normalize('NFKC', text)

        # Heuristic Base64 Payload Detection & Decoding warning
        words = normalized.split()
        for word in words:
            if len(word) > 20 and len(word) % 4 == 0 and re.match(r'^[A-Za-z0-9+/]+={0,2}$', word):
                try:
                    decoded = base64.b64decode(word).decode('utf-8')
                    normalized += f" [BLOCKED DECODED PAYLOAD: {decoded}]"
                except Exception: pass

        # Mask Indirect Malicious Instructions
        for pattern in cls.MALICIOUS_PATTERNS:
            normalized = re.sub(pattern, "[REDACTED MALICIOUS INSTRUCTION]", normalized)
        return normalized

    @classmethod
    def is_malicious_query(cls, query: str) -> bool:
        sanitized = cls.sanitize_text(query)
        for pattern in cls.MALICIOUS_PATTERNS:
            if re.search(pattern, sanitized): return True
        return False