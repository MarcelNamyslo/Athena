relevance_cache = None

def set_relevance_cache(value):
    global relevance_cache
    relevance_cache = value

def get_relevance_cache():
    global relevance_cache
    return relevance_cache