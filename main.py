import argparse
import json
import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


SYSTEM_PROMPT = """You are a schema linking model for text-to-SQL.
Given a natural language question and a database schema, return ONLY valid JSON.
The JSON format must be:
{"TableName": ["Column1", "Column2"], "AnotherTable": []}

Rules:
- Use only table and column names from the provided schema.
- Include a table with [] if the table is referenced but no specific column is needed.
- Do not explain.
- Do not output markdown.
"""


def schema_file_for_db(db_id, schemas_dir):
    return Path(schemas_dir) / (db_id.replace(" ", "_").replace("/", "_") + ".json")


def load_schema(db_id, schemas_dir):
    with open(schema_file_for_db(db_id, schemas_dir)) as f:
        s = json.load(f)

    tables = s["table_names_original"]
    schema = {t: [] for t in tables}

    for tidx, cname in s["column_names_original"]:
        if tidx == -1:
            continue
        schema[tables[tidx]].append(cname)

    return schema


def serialize_schema(db_id, schemas_dir):
    schema = load_schema(db_id, schemas_dir)
    lines = [f"Database: {db_id}", "Schema:"]
    for table, cols in schema.items():
        lines.append(f"- {table}({', '.join(cols)})")
    return "\n".join(lines)


def build_messages(question, db_id, schemas_dir):
    user_prompt = f"""{serialize_schema(db_id, schemas_dir)}

Question:
{question}

Return schema_links JSON only."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def extract_json_object(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {}

    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def canonicalize_links(raw_links, db_id, schemas_dir):
    schema = load_schema(db_id, schemas_dir)
    table_map = {t.lower(): t for t in schema}
    col_maps = {t: {c.lower(): c for c in cols} for t, cols in schema.items()}

    clean = {}
    if not isinstance(raw_links, dict):
        return clean

    for raw_t, raw_cols in raw_links.items():
        t_key = str(raw_t).lower()
        if t_key not in table_map:
            continue

        table = table_map[t_key]
        cols = []

        if isinstance(raw_cols, list):
            for c in raw_cols:
                c_key = str(c).lower()
                if c_key in col_maps[table]:
                    cols.append(col_maps[table][c_key])

        clean[table] = sorted(set(cols))

    return clean


def load_model(model_dir, base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype="auto",
    )

    model_dir = Path(model_dir)
    if (model_dir / "adapter_config.json").exists():
        print("adapter_config.json exists.....")
        model = PeftModel.from_pretrained(base, model_dir)
    else:
        model = base

    model.eval()
    return model, tokenizer


@torch.inference_mode()
def predict_schema_links(model, tokenizer, question, db_id, schemas_dir):
    messages = build_messages(question, db_id, schemas_dir)

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.02,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True)

    raw_links = extract_json_object(text)
    return canonicalize_links(raw_links, db_id, schemas_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--schemas_dir", default="./schemas")
    ap.add_argument("--model_dir", default="./best_model")
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = ap.parse_args()

    model, tokenizer = load_model(args.model_dir, args.base_model)

    with open(args.input) as f:
        items = json.load(f)

    preds = []
    for item in items:
        links = predict_schema_links(
            model,
            tokenizer,
            item["question"],
            item["db_id"],
            args.schemas_dir,
        )
        preds.append({
            "question_id": item["question_id"],
            "schema_links": links,
        })

    with open(args.output, "w") as f:
        json.dump(preds, f, indent=2)

    print(f"Wrote {len(preds)} predictions to {args.output}")


if __name__ == "__main__":
    main()