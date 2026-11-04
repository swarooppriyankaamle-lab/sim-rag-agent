"""LLM-as-judge grading of an answer against an expert reference answer."""
import config
from llm import make_client

GRADE_TOOL = {
    "name": "record_grade",
    "description": "Record the grade for this answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
            "reason": {"type": "string", "description": "One sentence explaining the grade."},
        },
        "required": ["label", "reason"],
    },
}

GRADER_SYSTEM = (
    "You grade answers from a simulation-engineering assistant against a reference answer written by an expert.\n"
    "correct: states the reference's key facts and numbers (wording may differ; numbers within rounding).\n"
    "partial: some key facts right, but something important is missing or imprecise.\n"
    "incorrect: wrong, contradicts the reference, or does not answer the question.\n"
    "Judge only against the reference. Extra correct detail is fine; extra wrong detail lowers the grade."
)

SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


def grade(question: str, reference: str, answer: str, client=None) -> dict:
    client = client or make_client()
    resp = client.messages.create(
        model=config.GRADER_MODEL, max_tokens=300, system=GRADER_SYSTEM,
        tools=[GRADE_TOOL], tool_choice={"type": "tool", "name": "record_grade"},
        messages=[{"role": "user", "content": (
            f"Question:\n{question}\n\nReference answer:\n{reference}\n\nAnswer to grade:\n{answer}"
        )}],
    )
    out = next(b.input for b in resp.content if b.type == "tool_use")
    return {"label": out.get("label", "incorrect"), "reason": out.get("reason", "")}
