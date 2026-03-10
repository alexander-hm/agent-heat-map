def caesar_cipher(text, shift):
    """Encrypt text using a Caesar cipher with the given shift. Preserves case and non-alpha."""
    result = []
    for ch in text:
        if ch.isalpha():
            shifted = chr(ord(ch) + shift)  # BUG: no modular wrap around alphabet
            result.append(shifted)
        else:
            result.append(ch)
    return "".join(result)
