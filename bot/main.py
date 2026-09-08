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
import re
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
    Update,
)
from telegram.constants import ChatAction
from agent.session import SessionStore
from config import Config
from rag.pipeline import MediRAG
from schema import Message, Role
from telegram.ext import (
    CallbackQueryHandler,
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
# Split against a lower ceiling than Telegram's: the HTML tags are added
# after splitting, so a part sized to exactly 4096 would overflow once
# <b> and &amp; expansions land on it.
SPLIT_LIMIT = 3500

# An INLINE keyboard, not a reply keyboard. A reply keyboard is an input
# surface: it occupies the same space as the system keyboard, so it is only
# visible while the user is not typing — is_persistent only promises to show it
# "when the regular keyboard is hidden". An inline keyboard lives inside a
# message bubble in the chat scroll instead, so attached to the newest message
# it sits directly above the text field and stays put while typing.
#
# callback_data is stateless: a tap on an old message's button still starts a
# new session correctly, so nothing has to be cleaned up as the chat grows.
NEW_SESSION_INLINE = InlineKeyboardMarkup(
    [[InlineKeyboardButton("🔄 New session", callback_data="new_session")]]
)


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


# Telegram's HTML subset: <b> <i> <u> <s> <code> <pre> <a> <blockquote>. No
# headers, no lists — so a header becomes bold and a bullet becomes "• ".
_CODE = re.compile(r"`([^`\n]+)`")
_HEADER = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*(.+?)[ \t]*#*[ \t]*$")
_BULLET = re.compile(r"(?m)^([ \t]*)[-*+][ \t]+")
_BOLD = re.compile(r"\*\*([^*\n]+?)\*\*")
_ITALIC = re.compile(r"(?<!\*)\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\*)")


def md_to_telegram_html(text: str) -> str:
    """Render the model's Markdown as Telegram HTML.

    Escape FIRST, then insert tags. Everything added after the escape is our
    own markup, so a <reference> block echoed out of the prompt arrives as
    inert &lt;reference&gt; text rather than taking the whole message down with
    a 400. Clinical prose like "under <30 minutes" is safe for the same reason.
    """
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Park code spans so the emphasis passes cannot reach inside them. Building
    # <code> first and then running the italic rule over the whole string
    # produces <code>x_<i>y</i></code> — nested tags Telegram rejects.
    spans: list[str] = []

    def _park(match: re.Match) -> str:
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE.sub(_park, text)

    # [ \t], never \s: \s matches newlines, so \s{0,3} silently swallows the
    # blank line above a header and collapses the paragraph break.
    text = _HEADER.sub(r"<b>\1</b>", text)
    text = _BULLET.sub(r"\1• ", text)   # before italics, or a leading "* " opens a span
    text = _BOLD.sub(r"<b>\1</b>", text)
    text = _ITALIC.sub(r"<i>\1</i>", text)

    for i, code in enumerate(spans):
        text = text.replace(f"\x00{i}\x00", f"<code>{code}</code>")

    return re.sub(r"\n{3,}", "\n\n", text).strip()


async def send(update: Update, text: str, reply_markup=None) -> None:
    """Send a reply, split if needed, formatted as Telegram HTML.

    Split BEFORE formatting, so every part is independently well-formed and a
    <b> span can never straddle a message boundary — that would 400 both
    halves. Splitting on the raw text also means the limit is applied to
    something shorter than what is finally sent, hence the headroom in
    SPLIT_LIMIT.

    reply_markup rides on the LAST part only, so the button lands at the very
    bottom of a multi-part answer rather than repeating between the parts.

    effective_message, not update.message: a callback query (an inline button
    tap) carries no update.message, and reading it there would raise.
    """
    parts = split_message(text, limit=SPLIT_LIMIT)
    for i, part in enumerate(parts):
        await update.effective_message.reply_text(
            md_to_telegram_html(part),
            parse_mode="HTML",
            reply_markup=reply_markup if i == len(parts) - 1 else None,
        )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Greeting, and the one place the old docked keyboard gets cleared.

    ReplyKeyboardRemove and an inline keyboard are both reply_markup, so they
    cannot share a message. This is the natural home for the removal: it is a
    one-time UI reset for anyone left holding the previous build's reply
    keyboard, and there is no session to restart yet, so no button is needed.
    """
    await send(
        update,
        f"Hi — describe what you're experiencing and I'll help you "
        f"understand it.\n\n{DISCLAIMER}",
        reply_markup=ReplyKeyboardRemove(),
    )


def chat_lock(context: ContextTypes.DEFAULT_TYPE, chat_id: int) -> asyncio.Lock:
    """One lock per chat, created on demand.

    Held across the whole read-modify-write, not just the store call: the gap
    between get() and save() spans a 20-45 second LLM call, so two quick
    messages from the same chat would otherwise both read the same history and
    the second save would clobber the first. Per-chat rather than global, so
    one user's slow answer never blocks anybody else.

    No await between the check and the insert, so this is safe on a single
    event loop without a lock of its own.
    """
    locks: dict[int, asyncio.Lock] = context.application.bot_data["locks"]
    if chat_id not in locks:
        locks[chat_id] = asyncio.Lock()
    return locks[chat_id]


async def new_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Start over: transcript, slots and demographics all discarded.

    Deliberately total. A session is one complaint, and carrying anything
    across the boundary means the next answer is shaped by the last illness.

    Serves both the /new command and the inline button. A callback query must be
    answered within a few seconds or the client leaves the button spinning, so
    that happens before the lock — clearing a dict is fast, but the lock may be
    held by an in-flight answer for the better part of a minute.
    """
    if update.callback_query is not None:
        await update.callback_query.answer("Starting a new session")

    chat_id = update.effective_chat.id
    async with chat_lock(context, chat_id):
        context.application.bot_data["sessions"].clear(chat_id)

    await send(
        update,
        f"Started a new session. Tell me what's going on.\n\n{DISCLAIMER}",
        reply_markup=NEW_SESSION_INLINE,
    )


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
        store: SessionStore = context.application.bot_data["sessions"]

        async with chat_lock(context, chat_id):
            session = store.get(chat_id)
            session.add(Message(role=Role.USER, text=text))
            # Copy before handing to the worker thread: the pipeline reads the
            # transcript off-loop, and a later /new must not mutate the list
            # underneath it.
            conversation = list(session.history)

            async with keep_typing(context.bot, chat_id):
                reply = await asyncio.to_thread(system.answer, conversation)

            session.add(Message(role=Role.BOT, text=reply))
            store.save(session)
        # ────────────────────────────────────────────────────────────────
        await send(update, reply, reply_markup=NEW_SESSION_INLINE)

    except Exception:
        # Never leak a stack trace to a user. Log it, apologise, stay running.
        log.exception("pipeline failure for chat_id=%s", chat_id)
        await send(update, "Something went wrong on my end. Try again in a moment.")


async def post_init(app: Application) -> None:
    """Publish no command list at all.

    Deleting rather than simply not registering: the list lives on Telegram's
    servers, so a bot that once had commands keeps showing the ☰ menu forever
    unless something clears it. This makes a fresh deploy converge on the same
    state as a clean one.

    /start and /new still work as typed commands, and Telegram's native START
    button for first-time users is independent of this list. Discovery is the
    reply keyboard's job.
    """
    await app.bot.delete_my_commands()
    log.info("cleared Telegram command menu")


def main() -> None:
    token = os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise SystemExit("TELEGRAM_TOKEN not set — copy .env.example to .env")

    app = Application.builder().token(token).post_init(post_init).build()

    # Built once, here — never per message. MediRAG holds the Retriever,
    # whose first use loads MedCPT; constructing it per update would pay
    # that cost on every message and hold N copies of a transformer.
    app.bot_data["rag"] = MediRAG(Config())
    app.bot_data["sessions"] = SessionStore()
    app.bot_data["locks"] = {}

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("new", new_session))
    # An inline tap arrives as a callback query, not a text message, so it
    # cannot collide with the catch-all handler the way the old reply-keyboard
    # button did — no ordering constraint and no interception needed.
    app.add_handler(CallbackQueryHandler(new_session, pattern="^new_session$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("bot starting (long polling)")
    app.run_polling()


if __name__ == "__main__":
    main()
