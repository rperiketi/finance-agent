from langgraph_finance.retrieval.vector_store import FinanceVectorStore


def test_categorization_collection_auto_seeds():
    store = FinanceVectorStore.in_memory()
    assert store.categorization_collection.count() > 0


def test_query_similar_examples_returns_relevant_category():
    store = FinanceVectorStore.in_memory()
    results = store.query_similar_examples("STARBUCKS STORE 9981", k=3)

    assert len(results) == 3
    categories = {r["category"] for r in results}
    assert "Dining" in categories


def test_knowledge_collection_auto_seeds_and_is_queryable():
    store = FinanceVectorStore.in_memory()
    results = store.query_knowledge("dining and food delivery spend", k=2)

    assert len(results) == 2
    assert all(isinstance(r, str) and r for r in results)


def test_stores_are_isolated_between_instances():
    store_a = FinanceVectorStore.in_memory()
    store_b = FinanceVectorStore.in_memory()

    assert store_a.categorization_collection.count() == store_b.categorization_collection.count()
    assert store_a.client is not store_b.client
