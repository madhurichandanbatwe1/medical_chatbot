from dotenv import load_dotenv
from flask import Flask, redirect, render_template, request, session, url_for
import os
from pathlib import Path
import secrets

load_dotenv()

from app.common.chat_state import (
    clear_chat_history,
    get_qa_chain,
    load_chat_history,
    save_chat_message,
)
from app.common.logger import get_logger

logger = get_logger(__name__)

is_production = os.environ.get("APP_ENV", "").lower() == "production"
secret_key = os.environ.get("FLASK_SECRET_KEY")
if is_production and not secret_key:
    raise RuntimeError("FLASK_SECRET_KEY must be configured in production")
if is_production and not os.environ.get("DATABASE_URL"):
    raise RuntimeError("DATABASE_URL must be configured in production")

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
)
app.config.update(
    SECRET_KEY=secret_key or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=16 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=is_production,
)

MAX_PROMPT_LENGTH = 2000


def get_conversation_id() -> str:
    conversation_id = session.get("conversation_id")
    if conversation_id is None:
        conversation_id = secrets.token_urlsafe(32)
        session["conversation_id"] = conversation_id
    return conversation_id


@app.route("/", methods=["GET", "POST"])
def index():
    conversation_id = get_conversation_id()
    error = None
    try:
        messages = load_chat_history(conversation_id)
    except Exception:
        logger.exception("Failed to load chat history")
        return (
            render_template(
                "index.html",
                messages=[],
                error="Chat history is unavailable. Please check the database configuration.",
            ),
            503,
        )

    if request.method == "POST":
        user_input = request.form.get("prompt", "").strip()
        if not user_input:
            error = "Enter a question before sending."
        elif len(user_input) > MAX_PROMPT_LENGTH:
            error = f"Please keep your question under {MAX_PROMPT_LENGTH} characters."
        else:
            try:
                save_chat_message(conversation_id, "user", user_input)
                messages.append({"role": "user", "content": user_input})
                response = get_qa_chain().invoke({"input": user_input})
                answer = response.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("The RAG chain returned an empty answer")

                save_chat_message(conversation_id, "assistant", answer)
                messages.append({"role": "assistant", "content": answer})
                return redirect(url_for("index"))
            except Exception:
                logger.exception("Failed to process a chat question")
                error = "Sorry, I couldn't answer that just now. Please try again."

    return render_template("index.html", messages=messages, error=error)


@app.route("/clear", methods=["POST"])
def clear():
    conversation_id = session.get("conversation_id")
    if conversation_id is not None:
        try:
            clear_chat_history(conversation_id)
        except Exception:
            logger.exception("Failed to clear chat history")
            return (
                render_template(
                    "index.html",
                    messages=[],
                    error="Chat history could not be cleared. Please try again.",
                ),
                503,
            )
        session.pop("conversation_id", None)
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )


