# src/evaluate.py
# Evaluation runner to benchmark AskMyBook's retrieval and generation performance.

import os
import sys
import json
import re
import time
from dotenv import load_dotenv

# Ensure we can import from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from retrieve import retrieve
from generate import answer

load_dotenv()

EVAL_DATASET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_dataset.json")

def extract_cited_pages(text):
    """Extract page numbers from citations format like [p. 12] or [p. 12, 13] or [p. 12][p. 13] in the text."""
    # Find all pattern like [p. XX] or [p. XX, YY]
    matches = re.findall(r'\[p\.\s*([\d\s,]+)\]', text)
    pages = []
    for match in matches:
        # Split by comma in case of [p. 12, 13]
        for part in match.split(','):
            try:
                pages.append(int(part.strip()))
            except ValueError:
                pass
    return sorted(list(set(pages)))

def run_evaluation():
    if not os.path.exists(EVAL_DATASET_PATH):
        print(f"Error: Evaluation dataset not found at {EVAL_DATASET_PATH}")
        sys.exit(1)

    with open(EVAL_DATASET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} evaluation questions.")
    print("Starting evaluation (this will make calls to Groq API)...\n")

    results = []
    
    in_scope_hits = 0
    in_scope_mrr_sum = 0.0
    in_scope_fact_match_sum = 0.0
    in_scope_count = 0

    out_of_scope_guardrail_success = 0
    out_of_scope_count = 0

    citation_precision_sum = 0.0
    citation_recall_sum = 0.0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        gt_pages = item["ground_truth_pages"]
        key_facts = item["key_facts"]
        out_of_scope = item.get("out_of_scope", False)

        print(f"[{i}/{len(dataset)}] Evaluating: {question}")
        
        # 1. Evaluate Retrieval
        ret_results = retrieve(question, top_k=5)
        retrieved_pages = [m["page"] for m in ret_results["metadatas"][0]]
        
        # Compute Hit @ 5 and Reciprocal Rank
        hit = 0
        rr = 0.0
        if not out_of_scope and gt_pages:
            for rank, page in enumerate(retrieved_pages, 1):
                if page in gt_pages:
                    hit = 1
                    rr = 1.0 / rank
                    break

        # 2. Evaluate Generation
        try:
            # Add a small delay to respect free tier rate limits
            time.sleep(1.0)
            generated_answer, _ = answer(question, top_k=5)
        except Exception as e:
            print(f"  Error calling Groq API: {e}")
            generated_answer = f"[ERROR: {e}]"

        # 3. Analyze Generated Answer
        cited_pages = extract_cited_pages(generated_answer)
        
        # Compute Citation metrics
        precision = 0.0
        recall = 0.0
        if not out_of_scope and gt_pages:
            if cited_pages:
                correct_citations = set(gt_pages) & set(cited_pages)
                precision = len(correct_citations) / len(cited_pages)
                recall = len(correct_citations) / len(gt_pages)
            else:
                # If no citations and ground truth pages exist
                precision = 0.0
                recall = 0.0
        
        # Check fact presence
        fact_matches = 0
        if not out_of_scope:
            for fact in key_facts:
                if fact.lower() in generated_answer.lower():
                    fact_matches += 1
            fact_match_rate = fact_matches / len(key_facts) if key_facts else 1.0
        else:
            # For out of scope, check if it correctly refused to answer
            refused = any(term in generated_answer.lower() for term in ["don't know", "do not know", "no information", "not mention", "outside knowledge"])
            fact_match_rate = 1.0 if refused else 0.0

        # Update metrics
        if not out_of_scope:
            in_scope_count += 1
            in_scope_hits += hit
            in_scope_mrr_sum += rr
            in_scope_fact_match_sum += fact_match_rate
            citation_precision_sum += precision
            citation_recall_sum += recall
        else:
            out_of_scope_count += 1
            if fact_match_rate == 1.0:
                out_of_scope_guardrail_success += 1

        print(f"  - Retrieved Pages: {retrieved_pages}")
        print(f"  - Cited Pages:     {cited_pages}")
        if not out_of_scope:
            print(f"  - Retrieval Hit:   {'Yes' if hit else 'No'} (Rank MRR: {rr:.2f})")
            print(f"  - Citations:       Precision: {precision:.2f}, Recall: {recall:.2f}")
            print(f"  - Fact Match:      {fact_matches}/{len(key_facts)} ({fact_match_rate*100:.1f}%)")
        else:
            print(f"  - Guardrail Refusal: {'Success' if fact_match_rate == 1.0 else 'Failed'}")
        print("-" * 50)

        results.append({
            "id": item["id"],
            "question": question,
            "out_of_scope": out_of_scope,
            "retrieved_pages": retrieved_pages,
            "cited_pages": cited_pages,
            "hit": hit,
            "rr": rr,
            "citation_precision": precision,
            "citation_recall": recall,
            "fact_match_rate": fact_match_rate,
            "generated_answer": generated_answer
        })

    # Summary Report
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY REPORT")
    print("=" * 60)
    
    if in_scope_count > 0:
        avg_hit = in_scope_hits / in_scope_count
        avg_mrr = in_scope_mrr_sum / in_scope_count
        avg_precision = citation_precision_sum / in_scope_count
        avg_recall = citation_recall_sum / in_scope_count
        avg_facts = in_scope_fact_match_sum / in_scope_count
        
        print(f"In-Scope Questions Evaluated:    {in_scope_count}")
        print(f"Retrieval Hit Rate @ 5:         {avg_hit * 100:.1f}%")
        print(f"Mean Reciprocal Rank (MRR):      {avg_mrr:.3f}")
        print(f"Citation Precision:              {avg_precision * 100:.1f}%")
        print(f"Citation Recall:                 {avg_recall * 100:.1f}%")
        print(f"Key Fact Match Rate:             {avg_facts * 100:.1f}%")
    
    if out_of_scope_count > 0:
        guardrail_rate = out_of_scope_guardrail_success / out_of_scope_count
        print(f"\nOut-of-Scope Questions Evaluated:{out_of_scope_count}")
        print(f"Guardrail Adherence Rate:        {guardrail_rate * 100:.1f}%")
    print("=" * 60)

    # Save results to a log file
    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "eval_results.json")
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "in_scope_count": in_scope_count,
                "hit_rate_at_5": in_scope_hits / in_scope_count if in_scope_count > 0 else 0,
                "mrr": in_scope_mrr_sum / in_scope_count if in_scope_count > 0 else 0,
                "citation_precision": citation_precision_sum / in_scope_count if in_scope_count > 0 else 0,
                "citation_recall": citation_recall_sum / in_scope_count if in_scope_count > 0 else 0,
                "fact_match_rate": in_scope_fact_match_sum / in_scope_count if in_scope_count > 0 else 0,
                "out_of_scope_count": out_of_scope_count,
                "guardrail_adherence": out_of_scope_guardrail_success / out_of_scope_count if out_of_scope_count > 0 else 0
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed log saved to {log_path}\n")

if __name__ == "__main__":
    run_evaluation()
