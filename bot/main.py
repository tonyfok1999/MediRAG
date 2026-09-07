"""Telegram entrypoint. Given in full — this is boilerplate, not the learning.

Run:  python -m bot.main

python-telegram-bot v21+ (async, Python 3.10+). v20 was a breaking async
rewrite, so most tutorials online predate it and will not run. Check the
version on anything you copy from Stack Overflow.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from config import Config
from rag.pipeline import MediRAG
from schema import Message, Role
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


@asynccontextmanager
async def keep_typing(bot, chat_id: int, interval: float = 4.0):
    """Hold the "typing…" indicator for as long as the block runs.

    send_chat_action sets the indicator for about five seconds, then Telegram
    drops it. Answers here take 19-46 seconds, so firing it once means the user
    watches typing stop two-thirds of the way through and concludes the bot
    died. Re-sending on a 4s cadence keeps it lit until the reply lands.

    Runs as a background task rather than inline because the work it is
    covering is awaited on the same loop; cancelled in `finally` so an
    exception in the pipeline cannot leave the task running forever.
    """

    async def loop() -> None:
        while True:
            try:
                await bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            except Exception:
                # A dropped chat action is cosmetic. Never let it take down the
                # request it is decorating.
                log.debug("chat action failed for chat_id=%s", chat_id, exc_info=True)
            await asyncio.sleep(interval)

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def send(update: Update, text: str) -> None:
    """Send a reply, split if needed.

    No parse mode at all. Both of Telegram's are traps for raw LLM output:
    MarkdownV2 requires escaping a long list of special characters, and HTML
    rejects the whole message with a 400 on any stray < or > — likely here,
    since the prompt is built out of <reference> blocks the model can echo.
    split_message can also slice a tag in half, producing a second failure on
    a message that would otherwise have been fine.

    The cost is that the model's **bold** and ### headers arrive as literal
    characters. That is a cosmetic problem; a rejected message is not.
    """
    for part in split_message(text):
        await update.message.reply_text(part)


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

    try:
        # ── pipeline ────────────────────────────────────────────────────
        # Still to wire in (Phases 2-4): SessionStore for multi-turn history,
        # safety.screen() before retrieval, and the agent ASK/ANSWER decision.
        #
        # to_thread is not optional here. MediRAG.answer() is fully synchronous
        # and slow — a torch forward pass, a Qdrant round-trip, and TWO LLM
        # calls (rewrite, then generate). Measured end-to-end: 19s, 36s, 46s.
        # Calling it directly from this coroutine would park the single asyncio
        # event loop for that whole time: no polling, no replies to any other
        # chat, not even the typing indicator above. One user would block
        # everyone. to_thread hands it to a worker so the loop stays free.
        system: MediRAG = context.application.bot_data["rag"]
        conversation = [Message(role=Role.USER, text=text)]
        async with keep_typing(context.bot, chat_id):
            reply = await asyncio.to_thread(system.answer, conversation)
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

    # Built once, here — never per message. MediRAG holds the Retriever,
    # whose first use loads MedCPT; constructing it per update would pay
    # that cost on every message and hold N copies of a transformer.
    app.bot_data["rag"] = MediRAG(Config())

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("scope", scope_cmd))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("bot starting (long polling)")
    app.run_polling()


if __name__ == "__main__":
    main()
