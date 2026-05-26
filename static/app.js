const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#messages");

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function scrollToBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, content, options = {}) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  if (role === "bot") {
    const avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.textContent = "N";
    article.appendChild(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = content;

  if (options.meta) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = options.meta;
    bubble.appendChild(meta);
  }

  if (options.sources?.length) {
    const details = document.createElement("details");
    details.className = "sources";
    details.innerHTML = `<summary>View sources (${options.sources.length})</summary>`;

    options.sources.forEach((source) => {
      const item = document.createElement("div");
      item.className = "source-item";
      item.textContent = source;
      details.appendChild(item);
    });

    bubble.appendChild(details);
  }

  article.appendChild(bubble);
  messages.appendChild(article);
  scrollToBottom();
  return article;
}

function addTypingMessage() {
  return addMessage(
    "bot",
    '<div class="typing" aria-label="Thinking"><span></span><span></span><span></span></div>'
  );
}

function setBusy(isBusy) {
  input.disabled = isBusy;
  sendButton.disabled = isBusy;
  sendButton.querySelector("span").textContent = isBusy ? "Wait" : "Send";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const question = input.value.trim();
  if (!question) return;

  addMessage("user", escapeHtml(question));
  input.value = "";
  setBusy(true);

  const typingMessage = addTypingMessage();
  const started = performance.now();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    const browserMs = Math.round(performance.now() - started);
    typingMessage.remove();
    addMessage("bot", escapeHtml(data.answer), {
      sources: data.sources || [],
      meta: `Answered in ${data.response_time_ms} ms locally`,
    });

    if (browserMs > 10000) {
      addMessage(
        "bot",
        "This response took more than 10 seconds in the browser. The local model or first-time embedding load may still be warming up."
      );
    }
  } catch (error) {
    typingMessage.remove();
    addMessage(
      "bot",
      `Unable to reach the local FastAPI backend. Start it with: <strong>uvicorn app:app --reload</strong>`
    );
  } finally {
    setBusy(false);
    input.focus();
  }
});

input.focus();
