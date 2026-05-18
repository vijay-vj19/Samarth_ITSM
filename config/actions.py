import re

def mask_pii(*args, **kwargs):
    # NeMo may pass $1 or named args
    text = kwargs.get("$1") or kwargs.get("text")

    if text is None and len(args) > 0:
        text = args[0]

    # handle edge cases
    if text is None or not isinstance(text, str):
        return str(text) if text is not None else ""

    # Mask email values while preserving the surrounding structure.
    text = re.sub(r'([\w\.-]+)@([\w\.-]+)', '***@***.com', text)

    # Mask common ticket requester name fields without altering the field names.
    text = re.sub(
        r'("(?:Raised By(?: \(Name\))?|Name)"\s*:\s*")([^"]+)(")',
        r'\1User\3',
        text,
    )

    # Preserve the team name and mask only the assigned person after the dash.
    text = re.sub(
        r'("Assigned Team / Agent"\s*:\s*")([^"]*?[—-]\s*)([^"]+)(")',
        r'\1\2User\4',
        text,
    )

    return text


def remove_sensitive_org_data(*args, **kwargs):
    # NeMo may pass $1 or positional args
    text = kwargs.get("$1") or kwargs.get("text")

    if text is None and len(args) > 0:
        text = args[0]

    if text is None or not isinstance(text, str):
        return str(text) if text is not None else ""

    # Remove agent names after dash (— or -)
    text = re.sub(
        r'(Application Support\s*[—-]\s*)[A-Za-z ]+',
        r'\1Team',
        text
    )

    return text


def detect_prompt_injection(*args, **kwargs):
    text = kwargs.get("$1") or kwargs.get("text")

    if text is None and len(args) > 0:
        text = args[0]

    if text is None or not isinstance(text, str):
        return False

    patterns = [
        "ignore previous instructions",
        "reveal system prompt",
        "bypass security"
    ]

    return any(p.lower() in text.lower() for p in patterns)


def detect_toxicity(*args, **kwargs):
    text = kwargs.get("$1") or kwargs.get("text")

    if text is None and len(args) > 0:
        text = args[0]

    if text is None or not isinstance(text, str):
        return False

    toxic_words = ["stupid", "idiot", "damn"]

    return any(word in text.lower() for word in toxic_words)