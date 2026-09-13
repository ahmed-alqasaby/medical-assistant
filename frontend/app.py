"""Medical Assistant — practice chat UI (M4).

A thin Streamlit front that talks to the FastAPI backend via ``BackendClient``:
question -> grounded, cited answer with citations rendered on screen, not hidden.

URL comes ONLY from ``BACKEND_URL`` (env), never hard-coded. Loading and error
states are explicit so a dead backend is visible, not silent.
"""

from __future__ import annotations

import asyncio
import os

import streamlit as st

from frontend.client import BackendClient, BackendError

st.set_page_config(page_title="Medical Assistant", page_icon="🩺", layout="centered")


@st.cache_resource
def get_client() -> BackendClient:
    return BackendClient.from_env()


def run_async(coro):
    """Asyncio boundary: Streamlit is sync; the client is async."""
    return asyncio.run(coro)


def render_answer(payload: dict) -> None:
    """Render one QueryResponse: answer (or refusal) + visible citations."""
    if payload.get("refuse"):
        reason = payload.get("refuse_reason") or "لا أملك معلومات كافية"
        st.chat_message("assistant").write(f"🤔 {reason}")
        return

    st.chat_message("assistant").write(payload.get("answer", ""))

    citations = payload.get("citations") or []
    if citations:
        with st.expander(f"المصادر ({len(citations)})", expanded=True):
            for c in citations:
                st.markdown(
                    f"**`{c.get('doc_id')}`** — {c.get('section', '')} · "
                    f"score {c.get('score', 0):.3f}"
                )
                st.markdown(f"> {c.get('quoted', '')}")
                st.divider()


def main() -> None:
    try:
        client = get_client()
    except BackendError as exc:
        st.error(f"لا يمكن الاتصال بالخلفية: {exc}")
        st.info("Set `BACKEND_URL` (e.g. export BACKEND_URL=http://localhost:8000) and rerun.")
        return

    with st.sidebar:
        st.subheader("Medical Assistant")
        st.caption(f"BACKEND_URL = `{os.environ.get('BACKEND_URL', '')}`")
        if st.button("فحص الاتصال"):
            try:
                health = run_async(client.health())
                st.success(
                    f"الخلفية تعمل — {health['store']['chunks']:,} chunks "
                    f"({health['store']['collection']})"
                )
            except BackendError as exc:
                st.error(str(exc))

    st.title("🩺 مرشد صحي موثوق بالمصادر")
    st.caption("أجب بالعربية. كل إجابة تستند إلى مصادر مسترجعة مرئية أدناه.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("citations"):
                render_answer(msg)
            else:
                st.write(msg["content"])

    if prompt := st.chat_input("اسأل عن جرعة دواء، أو حمية، أو علاج..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("جارٍ البحث في المصادر واسترجاع الإجابة..."):
                try:
                    payload = run_async(client.query(prompt))
                except BackendError as exc:
                    st.error(str(exc))
                else:
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": payload.get("answer", ""),
                            "citations": payload.get("citations", []),
                        }
                    )
                    render_answer(payload)


if __name__ == "__main__":
    main()