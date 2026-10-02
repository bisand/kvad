// Clipboard, with the fallback for a page not served over https.

export async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    const el = Object.assign(document.createElement("textarea"), { value: text });
    el.style.position = "fixed";
    el.style.opacity = "0";
    document.body.append(el);
    el.select();
    const ok = document.execCommand("copy");
    el.remove();
    return ok;
  }
}

// `use:codeCopy` on rendered Markdown: a Copy button on every code block.
export function codeCopy(node) {
  for (const block of node.querySelectorAll(".code")) {
    const b = document.createElement("button");
    b.className = "copy";
    b.type = "button";
    b.textContent = "Copy";
    b.addEventListener("click", async () => {
      await copy(block.querySelector("pre").innerText.replace(/\n$/, ""));
      b.textContent = "Copied";
      setTimeout(() => (b.textContent = "Copy"), 1400);
    });
    block.append(b);
  }
}
