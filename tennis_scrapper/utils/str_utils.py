def remove_characters(base_string: str, characters_to_remove: str) -> str:
    return "".join(c for c in base_string if c not in characters_to_remove)


def remove_digits(base_string: str) -> str:
    return "".join(c for c in base_string if not c.isdigit())


def remove_punctuation(base_string: str) -> str:
    import string

    return "".join(c for c in base_string if c not in string.punctuation)


def remove_non_digits(base_string: str) -> str:
    return "".join(c for c in base_string if c.isdigit())
