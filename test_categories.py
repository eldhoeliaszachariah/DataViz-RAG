
import json
import sys
import os

# Add current directory to path so we can import the module
sys.path.append(os.getcwd())

from condition_based_filter_new import ContextAwareJSONFilter

# Sample Data from original file
json6 = {
    "type": "table",
    "meta": {
        "section": "Full Table List of Projects in asia",
        "tags": ["", "projects_list", "individual_projects", "asia", "status", "per_project", "completion_percentages", "full table list", "Projects in asia"]
    },
    "data": [
        {"Projects": "Aunga", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "0%", "OT_Finished": "0%", "NT_Finished": "0%"},
        {"Projects": "Buri", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "53%", "OT_Finished": "39%", "NT_Finished": "100%"},
        {"Projects": "Chino", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "35%", "OT_Finished": "27%", "NT_Finished": "61%"},
        {"Projects": "Dega", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "66%", "OT_Finished": "56%", "NT_Finished": "100%"},
        {"Projects": "Disi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "50%", "OT_Finished": "35%", "NT_Finished": "100%"},
        {"Projects": "Galan", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "46%", "OT_Finished": "30%", "NT_Finished": "100%"},
        {"Projects": "Gavi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "48%", "OT_Finished": "32%", "NT_Finished": "100%"},
        {"Projects": "Goli", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "13%", "OT_Finished": "0%", "NT_Finished": "58%"},
        {"Projects": "Hado", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Inactive Project", "Full_Bible_Finished": "23%", "OT_Finished": "0%", "NT_Finished": "100%"},
        {"Projects": "Hanila", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "60%", "OT_Finished": "48%", "NT_Finished": "100%"},
        {"Projects": "Hapi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "40%", "OT_Finished": "22%", "NT_Finished": "100%"},
        {"Projects": "Harem", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "54%", "OT_Finished": "40%", "NT_Finished": "100%"},
        {"Projects": "Hiama", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "55%", "OT_Finished": "42%", "NT_Finished": "100%"},
        {"Projects": "Hidoli", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "60%", "OT_Finished": "49%", "NT_Finished": "100%"},
        {"Projects": "Holiboli", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "55%", "OT_Finished": "41%", "NT_Finished": "100%"},
        {"Projects": "Jado", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "63%", "OT_Finished": "52%", "NT_Finished": "100%"},
        {"Projects": "Japu", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "51%", "OT_Finished": "37%", "NT_Finished": "100%"},
        {"Projects": "Jia", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "31%", "OT_Finished": "22%", "NT_Finished": "62%"},
        {"Projects": "Kanvadhi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "57%", "OT_Finished": "44%", "NT_Finished": "100%"},
        {"Projects": "Komi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "51%", "OT_Finished": "37%", "NT_Finished": "100%"},
        {"Projects": "Lasho", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "17%", "OT_Finished": "0%", "NT_Finished": "74%"},
        {"Projects": "Liga", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "13%", "OT_Finished": "0%", "NT_Finished": "56%"},
        {"Projects": "Lihayo", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "62%", "OT_Finished": "50%", "NT_Finished": "100%"},
        {"Projects": "Likasar", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "47%", "OT_Finished": "32%", "NT_Finished": "100%"},
        {"Projects": "Lire", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "51%", "OT_Finished": "36%", "NT_Finished": "100%"},
        {"Projects": "Mahnahai", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "59%", "OT_Finished": "46%", "NT_Finished": "100%"},
        {"Projects": "Manal", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "30%", "OT_Finished": "9%", "NT_Finished": "100%"},
        {"Projects": "Mapiyala", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "59%", "OT_Finished": "46%", "NT_Finished": "100%"},
        {"Projects": "Nari", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "48%", "OT_Finished": "33%", "NT_Finished": "100%"},
        {"Projects": "Nepo", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "61%", "OT_Finished": "49%", "NT_Finished": "100%"},
        {"Projects": "Nila", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "13%", "OT_Finished": "0%", "NT_Finished": "57%"},
        {"Projects": "Pakkhi", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "67%", "OT_Finished": "57%", "NT_Finished": "100%"},
        {"Projects": "Priya", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "13%", "OT_Finished": "0%", "NT_Finished": "57%"},
        {"Projects": "Raisa", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "60%", "OT_Finished": "48%", "NT_Finished": "100%"},
        {"Projects": "Rimu", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "63%", "OT_Finished": "52%", "NT_Finished": "100%"},
        {"Projects": "Roja", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "61%", "OT_Finished": "49%", "NT_Finished": "100%"},
        {"Projects": "Sida", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "62%", "OT_Finished": "51%", "NT_Finished": "100%"},
        {"Projects": "Yak", "Region": "ASIA", "Nation": "ASIA R1", "OfficialStatus": "Active Project", "Full_Bible_Finished": "66%", "OT_Finished": "56%", "NT_Finished": "100%"},
        {"Projects": "Hagar", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "26%", "OT_Finished": "7%", "NT_Finished": "90%"},
        {"Projects": "Kashi", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "15%", "OT_Finished": "0%", "NT_Finished": "64%"},
        {"Projects": "Kukku", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "14%", "OT_Finished": "0%", "NT_Finished": "60%"},
        {"Projects": "Lina", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "26%", "OT_Finished": "7%", "NT_Finished": "90%"},
        {"Projects": "Nathu", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "51%", "OT_Finished": "36%", "NT_Finished": "100%"},
        {"Projects": "Techi", "Region": "ASIA", "Nation": "ASIA R2", "OfficialStatus": "Active Project", "Full_Bible_Finished": "14%", "OT_Finished": "0%", "NT_Finished": "62%"}
    ]
}

json7 = { 
    "type": "table",
    "meta": { 
        "section": "All regions that has project and their region wise counts", 
        "tags":["regions", "region_wise", "counts", "geographic", "all_regions", "summary", "aggregated"]
    }, 
    "data": [ 
        {"Region": "ASIA", "Total Projects": 54, "Active Projects": 43, "Missing Data": 11, "Ticket Count": 10 }, 
        {"Region": "EURASIA", "Total Projects": 18, "Active Projects": 8, "Missing Data": 28, "Ticket Count": 0 }, 
        {"Region": "SEA", "Total Projects": 76, "Active Projects": 70, "Missing Data": 214, "Ticket Count": 0 }, 
        {"Region": "SOUTHERN AFRICA", "Total Projects": 140, "Active Projects": 109, "Missing Data": 20, "Ticket Count": 0 }, 
        { "Region": "NORTHERN AFRICA", "Total Projects": 146, "Active Projects": 126, "Missing Data": 807, "Ticket Count": 3 }, 
        { "Region": "CENTRAL AFRICA", "Total Projects": 59, "Active Projects": 59, "Missing Data": 121, "Ticket Count": 0 }, 
        { "Region": "EUROPE", "Total Projects": 10, "Active Projects": 8, "Missing Data": 22, "Ticket Count": 0 } 
    ] 
}

# Categories
CATEGORIES = {
    "B": {
        "name": "SEMANTIC / FUZZY LOGIC",
        "questions": [
            (json6, "Show half completed projects"),
            (json6, "List projects with moderate progress on full bible"),
            (json6, "Show partially completed projects"),
            (json6, "Which projects are incomplete"),
            (json6, "Projects that are barely started (0-10%)"),
            (json6, "Show projects that are nearly finished (>90%)"),
            (json6, "List projects with very little progress"),
            (json6, "Projects with substantial completion (>50%)"),
            (json6, "Show me the stalled projects (0% full bible)"),
            (json6, "List projects that are in early stages (<20%)")
        ]
    },
    "C": {
        "name": "EXPLICIT RANGE QUERIES",
        "questions": [
            (json6, "Projects between 40% and 60% full bible finished"),
            (json6, "Projects with OT finished over 50%"),
            (json6, "Full bible finished less than 20%"),
            (json6, "NT finished between 0% and 50%"),
            (json6, "Projects with OT > 40%"),
            (json6, "Full bible < 10%"),
            (json6, "OT finished between 20% and 40%"),
            (json6, "NT finished under 80%"),
            (json6, "Projects where OT is greater than 30%"),
            (json6, "Full bible finished between 50% and 100%")
        ]
    },
    "D": {
        "name": "SPECIFIC ITEM RETRIEVAL (OR LOGIC)",
        "questions": [
            (json6, "Details of Goli, Liga, and Nila"),
            (json6, "Show me Hagar, Kashi, and Kukku"),
            (json6, "Projects in ASIA R1 and ASIA R2"),
            (json6, "Info on Aunga, Buri, and Chino"),
            (json6, "List status for Hado, Hanila, and Hapi"),
            (json6, "Details of Aunga and Hado"),
            (json6, "Show projects in Goli and Liga"),
            (json6, "Info on ASIA R1 and ASIA R2 nations"),
            (json6, "Projects Goli, Liga, Nila, and Priya"),
            (json6, "Show me Pakkhi and Rimu details")
        ]
    },
    "E": {
        "name": "COMPLEX CONDITIONS (COMBINED)",
        "questions": [
            (json6, "Active projects in ASIA R1 with >50% completion"),
            (json6, "Inactive projects with NT finished 100%"),
            (json6, "Projects in ASIA R2 with over 20% full bible"),
            (json6, "Active projects with full bible > 50%"),
            (json6, "Projects in ASIA R1 that are incomplete"),
            (json6, "Show inactive projects in ASIA with > 20% bible"),
            (json6, "Active projects with OT < 10%"),
            (json6, "List projects in ASIA R2 with moderate progress (30-70%)"),
            (json6, "Projects in ASIA R1 with NT finished 100%"),
            (json6, "Details of active projects with full bible < 50%")
        ]
    }
}

def run_test(category_key):
    if category_key not in CATEGORIES:
        print(f"Category {category_key} not found.")
        return

    cat = CATEGORIES[category_key]
    print(f"\n{'='*60}", flush=True)
    print(f"RUNNING CATEGORY {category_key}: {cat['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    engine = ContextAwareJSONFilter()
    
    results = []
    for i, (data, query) in enumerate(cat['questions']):
        print(f"\n[{category_key}.{i+1}] Query: '{query}'", flush=True)
        try:
            res = engine.process_query(data, query)
            count = len(res.get("data", []))
            print(f"Result: {count} records found.", flush=True)
            results.append({
                "category": category_key,
                "index": i+1,
                "query": query,
                "count": count,
                "status": "SUCCESS"
            })
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "category": category_key,
                "index": i+1,
                "query": query,
                "error": str(e),
                "status": "FAILED"
            })
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_categories.py <category_key>")
        sys.exit(1)
    
    cat_key = sys.argv[1].upper()
    run_test(cat_key)
