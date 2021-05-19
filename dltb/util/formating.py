"""Formating utilities.  A collection of functions to allow for
nicer output of different kinds of entities.
"""


def format_size(num, suffix='B'):
    """Format memory sizes as text.

    """
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if num < 1024:
            return f"{num}{unit}{suffix}"
        num //= 1024
    return f"{num}Yi{suffix}"
