from datasets import load_dataset
import html
import re

def strip_html(text):
    # Unescape HTML entities
    text = html.unescape(text)
    # Remove HTML tags
    return re.sub(r"<[^>]*>", "", text)

def format_nq_simplified(split="docs"):
    # Load the simplified train set
    dataset = load_dataset("natural_questions", "simplified_nq", split="train")

    formatted_data = []
    for entry in dataset:
        question = entry.get("question_text", "").strip()
        answer_list = entry.get("annotations", [])

        # Extract the first long answer (if present)
        answer = ""
        for ann in answer_list:
            if ann.get("long_answer", {}).get("start_token", -1) != -1:
                answer = ann["long_answer"].get("text", "")
                break
        
        # If no long answer found, fallback to short answer list
        if not answer:
            for ann in answer_list:
                if "short_answers" in ann and ann["short_answers"]:
                    answer = " ".join(sa.get("text", "") for sa in ann["short_answers"])
                    break

        # Clean up
        question = strip_html(question)
        answer = strip_html(answer)

        if question and answer:
            input((question,answer))
            formatted_data.append((question, answer))

    return formatted_data

if __name__ == "__main__":
    data = format_nq_simplified("docs")
    print(f"Sample: {data[0]}")
    print(f"Total formatted Q/A pairs: {len(data)}")
