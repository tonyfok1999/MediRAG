"""Telegram entrypoint. Given in full — this is boilerplate, not the learning.

Run:  python -m bot.main

python-telegram-bot v21+ (async, Python 3.10+). v20 was a breaking async
rewrite, so most tutorials online predate it and will not run. Check the
version on anything you copy from Stack Overflow.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
# httpx logs every Telegram poll at INFO. Without this your logs are unusable.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)

TELEGRAM_MAX = 4096

DISCLAIMER = (
    "⚕️ I provide general health information, not medical advice. "
    "I can't diagnose you. For anything urgent, contact emergency services."
)


def split_message(text: str, limit: int = TELEGRAM_MAX) -> list[str]:
    """Telegram hard-rejects messages over 4096 characters.

    Split on paragraph boundaries where possible, hard-split anything that is
    still too long (a single giant paragraph would otherwise slip through).
    """
    if len(text) <= limit:
        return [text]

    parts: list[str] = []
    current = ""
    for para in text.split("\n\n"):
        while len(para) > limit:              # single paragraph over the limit
            parts.append(para[:limit])
            para = para[limit:]
        if len(current) + len(para) + 2 > limit:
            if current:
                parts.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        parts.append(current)
    return parts


async def send(update: Update, text: str) -> None:
    """Send a reply, split if needed.

    HTML parse mode, not MarkdownV2: MarkdownV2 requires escaping a long list
    of special characters and will raise on raw LLM output.
    """
    for part in split_message(text):
        await update.message.reply_text(part, parse_mode="HTML")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send(
        update,
        f"Hi — describe what you're experiencing and I'll help you "
        f"understand it.\n\n{DISCLAIMER}",
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await send(
        update,
        "<b>Commands</b>\n"
        "/start — begin\n"
        "/reset — clear this conversation\n"
        "/scope — what I can and can't help with\n"
        "/help — this message",
    )


async def scope_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # TODO(day 7): render from scope.md so there's one source of truth.
    await send(update, "TODO: scope card")


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.clear()
    # TODO(day 8): also clear the SessionStore entry for this chat_id.
    await send(update, "Conversation reset. Tell me what's going on.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    text = (update.message.text or "").strip()

    if not text:
        await send(update, "Send me a description of what you're experiencing.")
        return

    # RAG takes 3-10 seconds. Without this, users assume the bot is broken.
    await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)

    try:
        # ── pipeline ────────────────────────────────────────────────────
        # session = store.get(chat_id)
        # session.add(Message(Role.USER, text))
        #
        # verdict, reply = safety.screen(text, session.history, cfg)
        # if verdict is not SafetyVerdict.OK:
        #     await send(update, reply)
        #     return
        #
        # session.slots |= agent.extract_slots(session, cfg)
        # decision = agent.decide(session, cfg)
        #
        # if decision.action is Action.ASK:
        #     reply = decision.question
        #     session.questions_asked += 1
        # else:
        #     query = rewriter.rewrite_query(session.history, cfg)
        #     retrieval = retriever.search(query)
        #     reply = generator.answer(session.history, retrieval, cfg)
        #
        # session.add(Message(Role.BOT, reply))
        # store.save(session)
        reply = f"(not wired up yet) you said: {text}"
        # ────────────────────────────────────────────────────────────────
        await send(update, reply)

    except Exception:
        # Never leak a stack trace to a user. Log it, apologise, stay running.
        log.exception("pipeline failure for chat_id=%s", chat_id)
        await send(update, "Something went wrong on my end. Try again in a moment.")


def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_TOKEN not set — copy .env.example to .env")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("scope", scope_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("bot starting (long polling)")
    app.run_polling()


if __name__ == "__main__":
    main()
