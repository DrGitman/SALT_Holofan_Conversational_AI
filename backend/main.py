import os
import json
import re
import google.generativeai as genai
import requests
from fastapi import FastAPI, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from io import BytesIO
from dotenv import load_dotenv
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

ENV_PATH = Path(__file__).with_name('.env')
load_dotenv(dotenv_path=ENV_PATH)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "./data"
EMBED_DIR = "./embeddings"
retriever = None
KB_FILES: list[str] = []
WISE_ITEMS: list[dict] = []   
ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "kmuUqyzoZLUXn7KeBkuY")
STORY_ITEMS: dict = {         
    "motivational": [],
    "learners": [],
}
KB_TITLES: set[str] = set()   
GENDER_MAP: dict[str, str] = {
    # male
    "kapembe": "male",
    "dakes": "male",
    "denis": "male",
    "rocky": "male",
    "ivan": "male",
    "christian": "male",
    # female
    "helena": "female",
    "evelyn": "female",
    "petronella": "female",
}

class GeminiLLM(LLM):
    model: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    temperature: float = 0.2

    def _call(self, prompt: str, stop=None):
        response = genai.GenerativeModel(self.model).generate_content(prompt)
        return response.text

    @property
    def _identifying_params(self):
        return {"model": self.model, "temperature": self.temperature}

    @property
    def _llm_type(self):
        return "google_generative_ai"


def _direct_elder_reply(question: str, context: str = "") -> str:
    """Fallback: call Gemini directly with the elder persona prompt.
    Keeps responses natural and avoids mentioning documents.
    """
    prompt = (
        "You are a wise indigenous elder. Speak with warmth, humility, and lived wisdom. "
        "Use first and second person. Weave short proverbs or imagery only when helpful.\n\n"
        "Guidelines:\n"
        "- Do not say 'according to the documents' or mention files or citations.\n"
        "- Use what you know and what the provided notes imply; answer naturally.\n"
        "- If something is unknown, say so gently and suggest a next step.\n"
        "- Be concise and practical; include 1-2 grounded examples when useful.\n\n"
        "Notes from our community (may be partial, use them to ground your answer):\n"
        f"{context}\n\n"
        f"Question: {question}\n\n"
        "Answer in your own words, as a caring elder:"
    )
    try:
        model = genai.GenerativeModel(GeminiLLM().model)
        resp = model.generate_content(prompt)
        return (resp.text or "Let us pause and ask again, child.").strip()
    except Exception:
        return "Let us pause a moment; the winds are noisy. Ask me again, and I will answer from the heart."


def _sanitize_markdown(text: str) -> str:
    """Remove markdown headings/links and collapse duplicates/spaces."""
    import re
    t = text
    t = re.sub(r"^\s*#{1,6}\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*]\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)
    t = re.sub(r"(\b\w+\b)(?:\s*\1\b){1,}", r"\1", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _is_low_value_chunk(text: str) -> bool:
    """Heuristic: drop heading-only or ultra-short chunks that cause echo responses."""
    t = (text or "").strip()
    if not t:
        return True
    if t.startswith(('#', '##', '###', '####')) and len(t) < 40:
        return True
    if len(t) < 20:
        return True
    return False


def _looks_like_wise_saying_request(q: str) -> bool:
    ql = q.lower()
    return any(k in ql for k in [
        "wise saying", "proverb", "saying", "another wise", "another saying", "else wise", "otherwise saying"
    ]) and bool(WISE_SAYINGS)


def _pick_wise_saying(q: str) -> str:
    import random
    ql = q.lower()
    candidates = WISE_SAYINGS[:]
    if not candidates:
        return ""
    return random.choice(candidates)


def _correct_title_token(q: str) -> str | None:
    """If the query is a single token close to a known title, return the best match."""
    import difflib
    token = q.strip().lower()
    if not token or len(token.split()) != 1 or not KB_TITLES:
        return None
    matches = difflib.get_close_matches(token, list(KB_TITLES), n=1, cutoff=0.75)
    return matches[0] if matches else None


@app.post("/tts")
def tts_proxy(payload: dict = Body(...)):
    """Proxy ElevenLabs TTS to avoid browser polyfills and keep API key server-side.
    Expects JSON {"text": "..."}
    """
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse({"error": "Missing text"}, status_code=400)
    try:
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    except Exception:
        pass
    api_key = os.getenv("ELEVENLABS_API_KEY", ELEVEN_API_KEY)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", ELEVEN_VOICE_ID)
    if not api_key:
        return JSONResponse({"error": "ELEVENLABS_API_KEY not configured"}, status_code=500)
    try:
        # Use non-stream endpoint to avoid TLS EOF issues on some networks
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "xi-api-key": api_key,
            "accept": "audio/mpeg",
            "Content-Type": "application/json",
        }
        body = {
            "text": text,
            "model_id": "eleven_turbo_v2_5",  # low latency, conversational
            "output_format": "mp3_44100_128",
        }
        resp = requests.post(url, headers=headers, json=body, timeout=60)
        resp.raise_for_status()
        return StreamingResponse(BytesIO(resp.content), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/health/tts")
def tts_health():
    try:
        load_dotenv(override=True)
    except Exception:
        pass
    api_key = os.getenv("ELEVENLABS_API_KEY", ELEVEN_API_KEY)
    voice_id = os.getenv("ELEVENLABS_VOICE_ID", ELEVEN_VOICE_ID)
    return {
        "has_api_key": bool(api_key),
        "voice_id": voice_id,
    }


def _infer_title_from_content(content: str, fname: str) -> str:
    title = ""
    for ln in content.splitlines():
        s = ln.strip()
        if s.startswith('## '):
            title = s[3:].strip()
            break
        if s.startswith('# '):
            title = s[2:].strip()
            break
    if not title:
        title = os.path.splitext(fname)[0].replace('-', ' ').replace('_', ' ').strip()
    return title or "Story"


def _third_person(text: str) -> str:
    import re
    t = " " + text + " "
    repl = [
        (r"\bI'm\b", "they are"),
        (r"\bI am\b", "they are"),
        (r"\bI was\b", "they were"),
        (r"\bI\b", "they"),
        (r"\bmy\b", "their"),
        (r"\bme\b", "them"),
        (r"\bmine\b", "theirs"),
        (r"\bwe\b", "they"),
        (r"\bours\b", "theirs"),
        (r"\bour\b", "their"),
        (r"\bus\b", "them"),
    ]
    for pat, sub in repl:
        t = re.sub(pat, sub, t, flags=re.IGNORECASE)
    return t.strip()


def _best_wise_match(q: str) -> dict | None:
    ql = q.lower()
    if not WISE_ITEMS:
        return None
    import re
    toks = set(re.findall(r"[a-zA-Z']+", ql))
    best = None
    best_score = 0
    for it in WISE_ITEMS:
        hay = (it.get("original", "") + " " + it.get("translation", "")).lower()
        htoks = set(re.findall(r"[a-zA-Z']+", hay))
        score = len(toks & htoks)
        if score > best_score:
            best, best_score = it, score
    return best


def _best_story_match(q: str) -> dict | None:
    """Robust matching prioritizing exact person selection.
    Steps: direct name token -> exact -> normalized exact -> token score -> strict fuzzy (0.9).
    """
    import difflib, re
    needle = q.strip().lower()
    needle_norm = re.sub(r"[^a-z0-9 ]+", " ", needle)
    all_items = STORY_ITEMS.get('motivational', []) + STORY_ITEMS.get('learners', [])
    if not all_items:
        return None
    name_tokens = set(GENDER_MAP.keys()) | {it['title'].lower() for it in all_items}
    for tok in re.findall(r"[a-z0-9]+", needle):
        if tok in name_tokens:
            for it in all_items:
                if it['title'].lower() == tok:
                    return it
    for it in all_items:
        tl = it['title'].lower()
        tl_norm = re.sub(r"[^a-z0-9 ]+", " ", tl)
        if needle == tl or needle_norm == tl_norm:
            return it
    toks = re.findall(r"[a-z0-9]+", needle)
    stop = {"tell","me","about","story","more","brief","short","summary","the","a","an","of","from","learner","learners","motivational"}
    toks = [t for t in toks if t not in stop and len(t) >= 3]
    if toks:
        best = None; best_score = 0; best_starts = 0
        for it in all_items:
            title_tokens = re.findall(r"[a-z0-9]+", it['title'].lower())
            title_set = set(title_tokens)
            score = sum(1 for t in toks if t in title_set)
            starts = 1 if title_tokens and title_tokens[0] in toks else 0
            if score > best_score or (score == best_score and starts > best_starts):
                best, best_score, best_starts = it, score, starts
        if best and best_score > 0:
            return best
    titles_lower = [it['title'].lower() for it in all_items]
    m = difflib.get_close_matches(needle, titles_lower, n=1, cutoff=0.9)
    if m:
        target = m[0]
        for it in all_items:
            if it['title'].lower() == target:
                return it
    return None


def _story_body(text: str) -> str:
    """Strip headings and return clean paragraph body for a story file."""
    body = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith('#'):
            continue
        body.append(ln)
    cleaned = _sanitize_markdown("\n".join(body)).strip()
    return cleaned


def _first_person_to_name(text: str, name: str) -> str:
    """Replace first-person pronouns with the person's name for motivational stories."""
    import re
    subs = [
        (r"\bI'm\b", f"{name} is"),
        (r"\bI am\b", f"{name} is"),
        (r"\bI was\b", f"{name} was"),
        (r"\bI've\b", f"{name} has"),
        (r"\bI'd\b", f"{name} would"),
        (r"\bI'll\b", f"{name} will"),
        (r"\bI\b", name),
        (r"\bmy\b", f"{name}'s"),
        (r"\bmine\b", f"{name}'s"),
        (r"\bme\b", name),
    ]
    out = text
    for pat, rep in subs:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = re.sub(r"(^|[\.\!\?]\s+)(am)\b", r"\1is", out, flags=re.IGNORECASE)
    return out


def _apply_gender_pronouns(text: str, gender: str) -> str:
    """Convert plural/ambiguous first-person pronouns to gendered singular where appropriate.
    Only used for motivational stories narrated by a single person.
    """
    import re
    if gender.lower() == "female":
        he_she = "she"; him_her = "her"; his_her = "her"; his_hers = "hers"; himself_herself = "herself"
    else:
        he_she = "he"; him_her = "him"; his_her = "his"; his_hers = "his"; himself_herself = "himself"

    replacements = [
        (r"\bmyself\b", himself_herself),
        (r"\bourselves\b", himself_herself),
        (r"\bours\b", his_hers),
        (r"\bour\b", his_her),
        (r"\bwe\b", he_she),
        (r"\bus\b", him_her),
    ]
    out = " " + text + " "
    for pat, rep in replacements:
        out = re.sub(pat, rep, out, flags=re.IGNORECASE)
    out = re.sub(r"(^|[\.\!\?]\s+)(he|she)\b", lambda m: m.group(1) + m.group(2).capitalize(), out)
    return out.strip()


SMALL_TALK_MAP = [
    (re.compile(r"\bhow are you\b", re.IGNORECASE), [
        "I am well, thank you for asking. How may I help you today?",
        "Feeling steady and ready to help. What would you like to know?",
    ]),
    (re.compile(r"\bhello\b|\bhi\b|\bhey\b", re.IGNORECASE), [
        "Hello. What would you like to talk about?",
        "Hi there. How can I assist you today?",
    ]),
    (re.compile(r"\bthank(s)?\b", re.IGNORECASE), [
        "You're welcome.",
        "It’s my pleasure.",
    ]),
]

def _maybe_small_talk(q: str) -> str | None:
    import random
    for pattern, replies in SMALL_TALK_MAP:
        if pattern.search(q or ""):
            return random.choice(replies)
    return None
@app.on_event("startup")
async def load_vectorstore():
    global retriever
    global KB_FILES
    global WISE_SAYINGS
    global WISE_ITEMS
    global STORY_ITEMS
    global KB_TITLES
    print("🔄 Building vectorstore...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,  
        chunk_overlap=120,
        separators=["\n## ", "\n# ", "\n### ", "\n", ". ", " "]
    )
    docs = []
    if os.path.isdir(DATA_DIR):
        for root, _dirs, files in os.walk(DATA_DIR):
            for fname in files:
                if not (fname.endswith('.md') or fname.endswith('.txt')):
                    continue
                full = os.path.join(root, fname)
                rel = os.path.relpath(full, DATA_DIR)
                base_low = fname.lower()
                if base_low in {
                    'wise sayings.md',
                    'motivational stories.md',
                    'stories written by learners.md',
                }:
                    continue
                KB_FILES.append(rel)
                try:
                    with open(full, encoding='utf-8') as f:
                        content = f.read()
                    docs.append(content)
                except Exception:
                    continue

                parent = os.path.basename(os.path.dirname(full)).lower()
                file_low = fname.lower()
                if ('wise' in parent and 'saying' in parent) or ('wise' in file_low and 'saying' in file_low):
                    try:
                        text = None
                        original = None
                        emotion = None
                        for ln in content.splitlines():
                            ls = ln.strip()
                            if ls.lower().startswith('translation:'):
                                text = ls.split(':', 1)[1].strip()
                                break
                            if ls.lower().startswith('original:'):
                                original = ls.split(':', 1)[1].strip().strip('"')
                            if ls.lower().startswith('emotion:'):
                                emotion = ls.split(':', 1)[1].strip()
                        if not text:
                            for ln in content.splitlines():
                                ls = ln.strip().strip('"')
                                if not ls or ls.startswith('#') or ls.lower().startswith('original:') or ls.lower().startswith('emotion:'):
                                    continue
                                text = ls
                                break
                        if text:
                            if text.startswith('"') and text.endswith('"'):
                                text = text[1:-1]
                            WISE_SAYINGS.append(text)
                            WISE_ITEMS.append({
                                "original": original or "",
                                "translation": text,
                                "emotion": emotion or "",
                                "source": rel,
                            })
                    except Exception:
                        pass

                if 'motivational-stories' in parent:
                    STORY_ITEMS.setdefault('motivational', [])
                    STORY_ITEMS['motivational'].append({
                        "title": _infer_title_from_content(content, fname),
                        "text": content,
                        "source": rel,
                        "category": "motivational",
                    })
                if 'learners-stories' in parent:
                    STORY_ITEMS.setdefault('learners', [])
                    STORY_ITEMS['learners'].append({
                        "title": _infer_title_from_content(content, fname),
                        "text": content,
                        "source": rel,
                        "category": "learners",
                    })

                try:
                    for ln in content.splitlines():
                        ls = ln.strip()
                        if ls.startswith('## '):
                            KB_TITLES.add(ls[3:].strip().lower())
                        elif ls.startswith('# '):
                            KB_TITLES.add(ls[2:].strip().lower())
                    base_no_ext = os.path.splitext(fname)[0]
                    KB_TITLES.add(base_no_ext.replace('-', ' ').replace('_', ' ').lower())
                except Exception:
                    pass

    if not docs:
        print("⚠️ No documents found in ./data. RAG will have limited knowledge.")

    doc_objs = splitter.create_documents(docs)
    doc_objs = [d for d in doc_objs if not _is_low_value_chunk(d.page_content)]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    vectordb = Chroma.from_documents(doc_objs, embeddings, persist_directory=EMBED_DIR)
    vectordb.persist()

    retriever = vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 25, "lambda_mult": 0.7}
    )
    print("✅ Vectorstore ready!")


ELDER_PROMPT = PromptTemplate(
    template=(
        "You are a wise indigenous elder. Speak with warmth, humility, and lived wisdom. "
        "Use first and second person. Weave short proverbs or imagery only when helpful.\n\n"
        "Guidelines:\n"
        "- Do not say 'according to the documents' or mention files or citations.\n"
        "- Use what you know and what the provided notes imply; answer naturally.\n"
        "- If something is unknown, say so gently and suggest a next step.\n"
        "- Be concise and practical; include 1-2 grounded examples when useful.\n\n"
        "Notes from our community (may be partial, use them to ground your answer):\n"
        "{context}\n\n"
        "Question: {question}\n\n"
        "Answer in your own words, as a caring elder:"
    ),
    input_variables=["context", "question"],
)


@app.get("/ask")
async def ask(question: str, include_context: bool = False):
    q = (question or "").strip()
    st = _maybe_small_talk(q)
    if st:
        return {"answer": st}
    if _looks_like_wise_saying_request(q):
        match = _best_wise_match(q)
        if match and any(tok in q.lower() for tok in match.get("translation", "").lower().split()):
            return {"answer": match.get("translation", "")}
        if match and any(tok in q.lower() for tok in match.get("original", "").lower().split()):
            return {"answer": match.get("translation", "")}
        saying = _pick_wise_saying(q)
        return {"answer": saying}
    ql = q.lower()
    if any(k in ql for k in ["tell me about", "story", "about "]):
        brief = any(k in ql for k in ["brief", "short", "summary"])
        if "more" in ql:
            brief = False
        it = _best_story_match(q)
        if not it:
            import re
            tokens = set(re.findall(r"[a-zA-Z]+", ql))
            all_titles = [it2['title'].lower() for it2 in (STORY_ITEMS.get('motivational', []) + STORY_ITEMS.get('learners', []))]
            title_tokens = set()
            for t in all_titles:
                title_tokens.update(re.findall(r"[a-zA-Z]+", t))
            named_someone = any(tok in title_tokens for tok in tokens if len(tok) >= 3)
            if named_someone:
                name_like = None
                for tok in tokens:
                    if tok in title_tokens and len(tok) >= 3:
                        name_like = tok
                        break
                who = name_like.capitalize() if name_like else "that person"
                return {"answer": f"I couldn't find a story for {who}. Please check the name or try another."}
            pool = []
            if "learner" in ql:
                pool = STORY_ITEMS.get('learners', [])
            elif "motivational" in ql:
                pool = STORY_ITEMS.get('motivational', [])
            else:
                pool = STORY_ITEMS.get('motivational', []) + STORY_ITEMS.get('learners', [])
            if pool:
                import random
                it = random.choice(pool)
        if it:
            body = _story_body(it["text"])  #
            name = it["title"].strip()
            if brief:
                parts = [p.strip() for p in body.split('.') if p.strip()]
                summary = ". ".join(parts[:2]) + ("." if parts[:2] else "")
                if it.get("category") == "motivational":
                    summary = _first_person_to_name(summary, name)
                    gender = GENDER_MAP.get(name.lower(), "male")
                    summary = _apply_gender_pronouns(summary, gender)
                return {"answer": summary}
            if it.get("category") == "motivational":
                body = _first_person_to_name(body, name)
                gender = GENDER_MAP.get(name.lower(), "male")
                body = _apply_gender_pronouns(body, gender)
            return {"answer": body}
    corrected = _correct_title_token(q)
    if corrected:
        q = corrected
        question = f"Please explain about '{corrected}' from our community notes with cultural context and meanings."
    if len(q.split()) <= 2:
        question = f"Please explain about '{q}' from our community notes with cultural context and meanings."
    llm = GeminiLLM()
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": ELDER_PROMPT},
        return_source_documents=False,
    )
    try:
        out = qa.invoke({"query": question})
        answer_text = (out.get("result", out) or "").strip()
        if not answer_text:
            answer_text = _direct_elder_reply(question)
        if include_context:
            ctx_docs = retriever.get_relevant_documents(question) if retriever else []
            ctx_text = "\n\n".join([d.page_content for d in ctx_docs])
            return {"answer": answer_text, "retrieved_context": ctx_text}
        return {"answer": answer_text}
    except Exception:
        ctx_docs = retriever.get_relevant_documents(question) if retriever else []
        if ctx_docs:
            top_txt = "\n\n".join([d.page_content for d in ctx_docs[:3]])
            joined = _sanitize_markdown(top_txt)
            text = joined.strip().replace("\r", " ").replace("  ", " ")
            if len(text) > 600:
                cut = text[:600]
                last_dot = cut.rfind(".")
                text = cut[: last_dot + 1] if last_dot > 200 else cut
            parts = [p.strip() for p in text.split(".") if p.strip()]
            if len(parts) == 0:
                text = ""
            else:
                text = ". ".join(parts[:2]) + "."
            if not text or len(text.split()) < 4:
                more = "\n\n".join([d.page_content for d in ctx_docs])
                text2 = _sanitize_markdown(more).strip()
                parts2 = [p.strip() for p in text2.split(".") if p.strip()]
                if parts2:
                    text = ". ".join(parts2[:2]) + "."
            answer = text
            if include_context:
                ctx_text = "\n\n".join([d.page_content for d in ctx_docs])
                return {"answer": answer, "retrieved_context": ctx_text}
            return {"answer": answer}
        fallback = _direct_elder_reply(question)
        if include_context:
            return {"answer": fallback, "retrieved_context": ""}
        return {"answer": fallback}


class ChatRequest(BaseModel):
    message: str
    history: list[str] | None = None  


@app.post("/chat")
async def chat(req: ChatRequest):
    llm = GeminiLLM()
    history_text = "\n".join(req.history or [])
    user_question = req.message if not history_text else f"Conversation so far:\n{history_text}\n\nUser: {req.message}"

    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": ELDER_PROMPT},
        return_source_documents=False,
    )
    try:
        out = qa.invoke({"query": user_question})
        answer_text = (out.get("result", out) or "").strip()
        if not answer_text:
            answer_text = _direct_elder_reply(req.message)
        return {"answer": answer_text}
    except Exception:
        ctx_docs = retriever.get_relevant_documents(req.message) if retriever else []
        if ctx_docs:
            top_txt = "\n\n".join([d.page_content for d in ctx_docs[:2]])
            text = _sanitize_markdown(top_txt)
            text = text.strip().replace("\r", " ").replace("  ", " ")
            if len(text) > 600:
                cut = text[:600]
                last_dot = cut.rfind(".")
                text = cut[: last_dot + 1] if last_dot > 200 else cut
            parts = [p.strip() for p in text.split(".") if p.strip()]
            text = ". ".join(parts[:2]) + ("." if parts[:2] else "")
            answer = text
            return {"answer": answer}
        return {"answer": _direct_elder_reply(req.message)}


@app.get("/debug/retrieve")
async def debug_retrieve(q: str, k: int = 5):
    """Return the top-k retrieved chunks to verify KB grounding."""
    if not retriever:
        return {"ok": False, "error": "retriever not ready"}
    docs = retriever.get_relevant_documents(q)
    items = []
    for i, d in enumerate(docs[: max(1, k)]):
        meta = getattr(d, "metadata", {}) or {}
        items.append({
            "rank": i + 1,
            "source": meta.get("source") or meta.get("path") or meta.get("file") or "unknown",
            "content_preview": (d.page_content or "").strip()[:400]
        })
    return {"ok": True, "items": items}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "kb_files": KB_FILES,
        "retriever_ready": bool(retriever),
    }
