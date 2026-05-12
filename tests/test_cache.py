from inferengine.core.cache import KVCacheManager


def test_cache_allocate_append_release():
    c = KVCacheManager(max_pages=8, page_size=4)
    c.allocate("a", prompt_tokens=3, max_new_tokens=5)
    assert c.stats()["used_pages"] == 2
    c.append_token("a")
    c.complete("a")
    c.release("a")
    assert c.stats()["used_pages"] == 0


def test_cache_rejects_request_larger_than_capacity():
    c = KVCacheManager(max_pages=2, page_size=4)
    try:
        c.allocate("too-big", prompt_tokens=1, max_new_tokens=100)
    except MemoryError:
        pass
    else:
        raise AssertionError("expected MemoryError")


def test_cache_evicts_completed_sequence():
    c = KVCacheManager(max_pages=2, page_size=4)
    c.allocate("a", prompt_tokens=1, max_new_tokens=7)
    c.complete("a")
    c.allocate("b", prompt_tokens=1, max_new_tokens=7)
    stats = c.stats()
    assert stats["active_sequences"] == 1
    assert stats["evictions"] == 1
