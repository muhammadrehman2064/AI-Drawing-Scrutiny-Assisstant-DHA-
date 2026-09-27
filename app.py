"""
app.py
------
DHA Architectural Drawing Scrutiny - RAG MVP

Flow:
1. User uploads architectural drawing PDF.
2. Pages are rendered to images.
3. Groq vision model extracts only visible drawing facts.
4. FAISS retrieves relevant clauses from the supplied DHA Byelaws.
5. Groq produces a strictly structured, evidence-grounded scrutiny report.
6. Application rejects observations whose cited Byelaw chunk does not exist.

IMPORTANT:
This is an AI-assisted preliminary scrutiny tool, not an official DHA approval.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image
from groq import Groq

from ingest import search_byelaws


APP_TITLE = "DHA Architectural Drawing Scrutiny"
BYELAW_PDF = Path("byelaws/Updated-DHA-Const-Dev-Byelaws-2026.pdf")

VISION_MODEL = "qwen/qwen3.8-27b"
TEXT_MODEL = "qwen/qwen3.8-27b"

MAX_DRAWING_PAGES = 30
IMAGE_DPI = 140
JPEG_QUALITY = 82


st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🏗️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    key = st.session_state.get("groq_api_key")
    if key:
        return key

    try:
        key = st.secrets["GROQ_API_KEY"]
    except Exception:
        key = os.getenv("GROQ_API_KEY", "")

    return key.strip()


def pdf_to_images(pdf_bytes: bytes, max_pages: int = MAX_DRAWING_PAGES):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total = min(len(doc), max_pages)
    images = []

    for i in range(total):
        page = doc.load_page(i)
        pix = page.get_pixmap(
            dpi=IMAGE_DPI,
            alpha=False,
        )
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((i + 1, img))

    doc.close()
    return images


def image_to_data_url(img: Image.Image) -> str:
    # Keep request sizes manageable.
    img = img.copy()
    max_side = 1800

    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        )

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def call_groq_vision(
    client: Groq,
    page_no: int,
    img: Image.Image,
    project_info: str,
) -> Dict[str, Any]:

    prompt = f"""
You are extracting FACTS from an architectural drawing for preliminary
building-byelaw scrutiny.

PROJECT INFORMATION:
{project_info}

DRAWING PAGE:
{page_no}

STRICT RULES:
1. Report only information visibly present on this drawing page.
2. Never guess a dimension, area, room use, setback, coverage, height,
   north direction, plot size, or compliance.
3. Do not calculate missing dimensions.
4. If text/dimension is unreadable, use null or an empty list.
5. Every extracted item must include exact page number and short visual
   evidence describing what is visible.
6. Distinguish clearly between a written dimension and your interpretation.
7. This step is EXTRACTION ONLY. Do not compare with building byelaws.
8. Return valid JSON only.

Extract:
- plot dimensions if shown
- plot area if shown
- road/frontage information if shown
- north direction if shown
- building footprint / covered area if explicitly stated or dimensioned
- front, rear and side setbacks/clear spaces if dimensioned
- floor names
- room names and room dimensions
- stairs, mumtee, basement
- porch/verandah
- boundary wall/gate/guard room
- swimming pool
- floor-to-floor/overall height if shown
- parking
- any explicit notes referring to byelaws
"""

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(img)},
                    },
                ],
            }
        ],
        response_format={"type": "json_object"},
        temperature=0,
        max_completion_tokens=3000,
    )

    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Safe fallback: no facts rather than inventing facts.
        data = {
            "page": page_no,
            "extraction_error": "Model returned invalid JSON.",
            "facts": [],
        }

    data["page"] = page_no
    return data


def normalize_facts(page_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []

    for result in page_results:
        page = result.get("page")
        facts = result.get("facts", [])

        if isinstance(facts, dict):
            facts = [facts]

        if not isinstance(facts, list):
            facts = []

        for fact in facts:
            if not isinstance(fact, dict):
                continue

            normalized.append(
                {
                    "page": page,
                    "category": str(fact.get("category", "general")),
                    "item": str(fact.get("item", "")),
                    "value": fact.get("value"),
                    "evidence": str(fact.get("evidence", "")),
                    "confidence": str(fact.get("confidence", "unknown")),
                }
            )

    return normalized


def build_rag_query(
    project_type: str,
    plot_size: str,
    phase: str,
    facts: List[Dict[str, Any]],
) -> str:
    fact_lines = []

    for fact in facts:
        value = fact.get("value")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        fact_lines.append(
            f"{fact.get('category')}: {fact.get('item')} = {value}; "
            f"evidence: {fact.get('evidence')}"
        )

    return f"""
Residential architectural drawing scrutiny.
Project type: {project_type}
Plot size: {plot_size}
DHA phase: {phase}

Find the Byelaws clauses relevant to:
clear spaces/setbacks, covered area, maximum height, basement,
staircase/mumtee, swimming pool, guard room, plinth/sunken area,
parking/access, boundary wall and any other requirements applicable
to the facts below.

DRAWING FACTS:
{chr(10).join(fact_lines[:120])}
""".strip()


def format_evidence(chunks: List[Dict[str, Any]]) -> str:
    blocks = []

    for c in chunks:
        blocks.append(
            f"""[BYELAW_CHUNK_ID={c['id']}]
Page: {c['page']}
Section: {c.get('section', '')}
Text:
{c['text']}
"""
        )

    return "\n".join(blocks)


def make_report_schema() -> Dict[str, Any]:
    observation = {
        "type": "object",
        "properties": {
            "item": {"type": "string"},
            "status": {
                "type": "string",
                "enum": ["COMPLIANT", "NON-COMPLIANT", "UNABLE TO VERIFY"],
            },
            "drawing_page": {"type": "string"},
            "drawing_evidence": {"type": "string"},
            "byelaw_chunk_id": {"type": "string"},
            "byelaw_reference": {"type": "string"},
            "requirement": {"type": "string"},
            "explanation": {"type": "string"},
        },
        "required": [
            "item",
            "status",
            "drawing_page",
            "drawing_evidence",
            "byelaw_chunk_id",
            "byelaw_reference",
            "requirement",
            "explanation",
        ],
        "additionalProperties": False,
    }

    return {
        "type": "object",
        "properties": {
            "overall_summary": {"type": "string"},
            "observations": {
                "type": "array",
                "items": observation,
            },
        },
        "required": ["overall_summary", "observations"],
        "additionalProperties": False,
    }


def call_grounded_report(
    client: Groq,
    project_info: str,
    facts: List[Dict[str, Any]],
    chunks: List[Dict[str, Any]],
    language: str,
) -> Dict[str, Any]:

    allowed_ids = [c["id"] for c in chunks]
    evidence = format_evidence(chunks)

    facts_text = json.dumps(
        facts,
        ensure_ascii=False,
        indent=2,
    )

    language_instruction = (
        "Write the explanation and overall summary in English."
        if language == "English"
        else
        "Write the explanation and overall summary in simple Roman English "
        "(Latin/English letters). Do not use Urdu script. Keep technical "
        "Byelaw wording and references in their original English form."
    )

    system = f"""
You are a building-byelaw scrutiny assistant.

AUTHORITATIVE SOURCE:
Only the supplied BYELAW EVIDENCE below is authoritative for regulatory
requirements. Do not use outside knowledge.

ANTI-HALLUCINATION RULES:
1. You may NOT invent a Byelaw requirement.
2. You may NOT cite a Byelaw page, section or ID that is not present in
   BYELAW EVIDENCE.
3. Every observation must cite exactly one BYELAW_CHUNK_ID from the allowed
   list.
4. If the drawing does not visibly establish the required fact, status MUST
   be UNABLE TO VERIFY.
5. If the Byelaw evidence does not establish an exact requirement applicable
   to the item, status MUST be UNABLE TO VERIFY.
6. Do not infer plot dimensions, setbacks, areas or heights.
7. Do not perform architectural interpretation beyond what the drawing facts
   explicitly establish.
8. Do not claim official DHA approval/rejection.
9. COMPLIANT requires both: an applicable requirement AND drawing evidence
   sufficient to compare against it.
10. NON-COMPLIANT requires both: an applicable requirement AND drawing
    evidence that clearly contradicts it.
11. If either side is missing, use UNABLE TO VERIFY.
12. Keep the output concise but useful.

ALLOWED BYELAW CHUNK IDS:
{json.dumps(allowed_ids)}

{language_instruction}
"""

    user = f"""
PROJECT:
{project_info}

DRAWING FACTS:
{facts_text}

BYELAW EVIDENCE:
{evidence}

Prepare a preliminary scrutiny report.
"""

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "dha_scrutiny_report",
                "strict": True,
                "schema": make_report_schema(),
            },
        },
        temperature=0,
        max_completion_tokens=6000,
    )

    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def validate_report(report: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Final safety gate. If the model cites an unknown Byelaw chunk, the
    observation is downgraded to UNABLE TO VERIFY instead of allowing an
    unsupported citation.
    """
    allowed = {c["id"]: c for c in chunks}

    safe = {
        "overall_summary": str(report.get("overall_summary", "")),
        "observations": [],
    }

    for obs in report.get("observations", []):
        obs = dict(obs)

        chunk_id = obs.get("byelaw_chunk_id", "")
        drawing_evidence = str(obs.get("drawing_evidence", "")).strip()
        requirement = str(obs.get("requirement", "")).strip()

        if chunk_id not in allowed:
            obs["status"] = "UNABLE TO VERIFY"
            obs["byelaw_chunk_id"] = ""
            obs["byelaw_reference"] = "No verified Byelaw evidence"
            obs["requirement"] = ""
            obs["explanation"] = (
                "The system could not verify the cited Byelaw evidence. "
                "This observation has therefore been downgraded."
            )

        elif not drawing_evidence or not requirement:
            obs["status"] = "UNABLE TO VERIFY"
            obs["explanation"] = (
                "Required drawing evidence or Byelaw requirement was not "
                "sufficiently established."
            )

        safe["observations"].append(obs)

    return safe


def report_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# DHA Architectural Drawing Scrutiny",
        "",
        "**AI-Assisted Preliminary Scrutiny — Not Official DHA Approval**",
        "",
        report.get("overall_summary", ""),
        "",
        "## Observations",
        "",
        "| Item | Status | Drawing Evidence | Byelaw Requirement | Reference | Explanation |",
        "|---|---|---|---|---|---|",
    ]

    for o in report.get("observations", []):
        def clean(v):
            return str(v).replace("|", "\\|").replace("\n", " ")

        lines.append(
            f"| {clean(o.get('item'))} | {clean(o.get('status'))} | "
            f"{clean(o.get('drawing_evidence'))} | "
            f"{clean(o.get('requirement'))} | "
            f"{clean(o.get('byelaw_reference'))} | "
            f"{clean(o.get('explanation'))} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.title("🏗️ DHA Architectural Drawing Scrutiny")
st.caption(
    "RAG-based preliminary scrutiny against the supplied "
    "DHA Construction & Development Byelaws 2026."
)

with st.sidebar:
    st.header("Project Details")

    project_type = st.selectbox(
        "Project Type",
        ["Residential"],
    )

    plot_size = st.text_input(
        "Plot Size",
        placeholder="e.g. 1 Kanal / 10 Marla",
    )

    phase = st.text_input(
        "DHA Phase",
        placeholder="e.g. Phase 6",
    )

    language = st.radio(
        "Report Language",
        ["English", "Roman English"],
    )

    api_key_input = st.text_input(
        "Groq API Key",
        type="password",
        help="You can also store GROQ_API_KEY in Streamlit Secrets.",
    )

    if api_key_input:
        st.session_state["groq_api_key"] = api_key_input

    st.divider()
    st.info(
        "The Byelaws PDF placed in /byelaws is treated as the authoritative "
        "regulatory source."
    )

    if st.button("Build / Rebuild Byelaws Index"):
        if not BYELAW_PDF.exists():
            st.error(
                f"Byelaws PDF not found: {BYELAW_PDF}\n\n"
                "Place the supplied PDF inside the byelaws folder."
            )
        else:
            with st.spinner("Building FAISS index..."):
                try:
                    result = (BYELAW_PDF, force=True)
                    st.success(
                        f"Index built: {result.get('chunks', 0)} chunks."
                    )
                except Exception as exc:
                    st.exception(exc)

uploaded = st.file_uploader(
    "Upload Architectural Drawing PDF",
    type=["pdf"],
)

if uploaded:
    st.success(f"Uploaded: {uploaded.name}")

    if st.button("🔎 Start Scrutiny", type="primary"):
        api_key = get_api_key()

        if not api_key:
            st.error(
                "Groq API key is required. Enter it in the sidebar or add "
                "GROQ_API_KEY to Streamlit Secrets."
            )
            st.stop()

        if not plot_size.strip():
            st.warning(
                "Plot size was not entered. The system will continue, but "
                "checks depending on plot size may become UNABLE TO VERIFY."
            )

        if not BYELAW_PDF.exists():
            st.error(
                "Byelaws PDF not found. Put "
                "'Updated-DHA-Const-Dev-Byelaws-2026.pdf' inside /byelaws."
            )
            st.stop()

        if not Path("faiss_index/byelaws.faiss").exists():
            with st.spinner("First run: building Byelaws FAISS index..."):
                try:
                    (BYELAW_PDF, force=False)
                except Exception as exc:
                    st.exception(exc)
                    st.stop()

        client = Groq(api_key=api_key)

        project_info = (
            f"Project type: {project_type}\n"
            f"Plot size: {plot_size or 'Not provided'}\n"
            f"DHA phase: {phase or 'Not provided'}"
        )

        try:
            # 1. Render drawing
            with st.status("Reading architectural drawing...", expanded=True) as status:
                pdf_bytes = uploaded.getvalue()
                pages = pdf_to_images(pdf_bytes)

                if not pages:
                    st.error("No pages could be read from the uploaded PDF.")
                    st.stop()

                st.write(f"Drawing pages processed: {len(pages)}")

                # 2. Vision extraction
                page_results = []
                for page_no, image in pages:
                    st.write(f"Extracting visible facts from page {page_no}...")
                    result = call_groq_vision(
                        client,
                        page_no,
                        image,
                        project_info,
                    )
                    page_results.append(result)

                status.update(
                    label="Drawing extraction complete",
                    state="complete",
                )

            facts = normalize_facts(page_results)

            with st.expander("Extracted Drawing Facts", expanded=False):
                st.json(facts)

            # 3. RAG retrieval
            with st.spinner("Retrieving relevant DHA Byelaw clauses..."):
                rag_query = build_rag_query(
                    project_type,
                    plot_size,
                    phase,
                    facts,
                )
                chunks = search_byelaws(rag_query, top_k=12)

            if not chunks:
                st.error("No Byelaw evidence was retrieved.")
                st.stop()

            with st.expander("Retrieved Byelaw Evidence", expanded=False):
                for c in chunks:
                    st.markdown(
                        f"**{c['id']} — Page {c['page']} — "
                        f"{c.get('section', '')}**"
                    )
                    st.write(c["text"])
                    st.caption(f"Similarity score: {c.get('score', 0):.3f}")

            # 4. Grounded final report
            with st.spinner("Generating evidence-grounded scrutiny report..."):
                report = call_grounded_report(
                    client,
                    project_info,
                    facts,
                    chunks,
                    language,
                )

                report = validate_report(report, chunks)

            observations = report.get("observations", [])

            compliant = sum(
                1 for x in observations if x.get("status") == "COMPLIANT"
            )
            non_compliant = sum(
                1 for x in observations if x.get("status") == "NON-COMPLIANT"
            )
            unable = sum(
                1 for x in observations if x.get("status") == "UNABLE TO VERIFY"
            )

            st.divider()
            st.subheader("Scrutiny Summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Compliant", compliant)
            c2.metric("Non-Compliant", non_compliant)
            c3.metric("Unable to Verify", unable)

            st.info(
                "AI-Assisted Preliminary Scrutiny only. "
                "This report does not constitute official DHA approval."
            )

            st.markdown(report.get("overall_summary", ""))

            for obs in observations:
                status_value = obs.get("status", "UNABLE TO VERIFY")

                if status_value == "COMPLIANT":
                    icon = "✅"
                elif status_value == "NON-COMPLIANT":
                    icon = "❌"
                else:
                    icon = "⚠️"

                with st.container(border=True):
                    st.markdown(
                        f"### {icon} {obs.get('item', 'Observation')}"
                    )
                    st.write(f"**Status:** {status_value}")
                    st.write(
                        f"**Drawing Evidence:** "
                        f"{obs.get('drawing_evidence', '')}"
                    )
                    st.write(
                        f"**Byelaw Requirement:** "
                        f"{obs.get('requirement', '')}"
                    )
                    st.write(
                        f"**Reference:** "
                        f"{obs.get('byelaw_reference', '')}"
                    )
                    st.write(
                        f"**Explanation:** "
                        f"{obs.get('explanation', '')}"
                    )

            md = report_markdown(report)
            json_download = json.dumps(
                report,
                ensure_ascii=False,
                indent=2,
            )

            st.download_button(
                "⬇️ Download Markdown Report",
                data=md,
                file_name="dha_scrutiny_report.md",
                mime="text/markdown",
            )

            st.download_button(
                "⬇️ Download JSON Report",
                data=json_download,
                file_name="dha_scrutiny_report.json",
                mime="application/json",
            )

        except Exception as exc:
            st.error("Scrutiny stopped because an application error occurred.")
            st.exception(exc)

else:
    st.info(
        "Upload an architectural drawing PDF to begin. "
        "For the first test, use a clear residential architectural plan."
    )
