from condition_based_filter_new import ContextAwareJSONFilter
import json

def test_integration():
    engine = ContextAwareJSONFilter()
    
    # 1. Simulate the ASIA R2 query (Testing grounding and tags)
    json_data = {
        "type": "table",
        "meta": {
            "section": "Full Table List of Projects",
            "tags": ["projects", "asia", "r2", "nation"]
        },
        "data": [
            {"Projects": "Hagar", "Nation": "ASIA R2", "Status": "Active"},
            {"Projects": "Buri", "Nation": "ASIA R1", "Status": "Active"}
        ]
    }
    
    print("\n--- Test: ASIA R2 grounding ---")
    res1 = engine.process_query(json_data, "list projects in ASIA R2")
    print(f"Matched: {len(res1['data'])} rows")

    # 2. Simulate the 'highest' query (Testing Python-based logic)
    json_stats = {
        "type": "table",
        "meta": {
            "section": "Region Stats",
            "tags": ["regions", "counts"]
        },
        "data": [
            {"Region": "A", "Count": 10},
            {"Region": "B", "Count": 50},
            {"Region": "C", "Count": 25}
        ]
    }
    
    print("\n--- Test: Highest retrieval ---")
    res2 = engine.process_query(json_stats, "which region has highest count")
    print(f"Result: {res2['data']}")

    # 3. Simulate 'show all' (Testing full view trigger)
    print("\n--- Test: Full view trigger ---")
    res3 = engine.process_query(json_stats, "show all the statistics")
    print(f"Matched: {len(res3['data'])} rows (Total was {len(json_stats['data'])})")

if __name__ == "__main__":
    test_integration()
