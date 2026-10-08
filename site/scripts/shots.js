// Screenshots of the web UI, for the site.
//
//   node scripts/shots.js [http://127.0.0.1:5823] [name...]
//
// Takes each page in SHOTS in both themes and writes
// static/shots/NAME-dark.webp and NAME-light.webp. It drives the Chrome that
// is installed, so nothing is downloaded. Point it at a server whose data
// you are happy to publish: what is on its pages is what ends up here.

import { chromium } from "playwright";
import sharp from "sharp";
import { mkdirSync } from "node:fs";

const [base = "http://127.0.0.1:5823", ...only] = process.argv.slice(2);

// `prepare` runs in the page before the picture is taken.
const SHOTS = [
  { name: "dashboard", path: "/" },
  { name: "models", path: "/models" },
  {
    name: "chat",
    path: "/chat",
    // The newest conversation.
    prepare: async (p) => {
      await p.getByText(/^What is a KV cache/).first().click();
      await p.waitForTimeout(600);
    },
  },
  {
    name: "playground",
    path: "/playground",
    // A completion with the five candidates kept at each step, and one
    // token opened to show them.
    prepare: async (p) => {
      await p.locator('input[type="range"][max="20"]').fill("5");
      await p.locator('input[type="range"][max="512"]').fill("48");
      await p.getByRole("button", { name: "Continue" }).click();
      await p.getByRole("button", { name: "Stop" }).waitFor({ state: "detached", timeout: 120000 });
      await p.locator("button", { hasText: /^\s*(Vaswani|Attention|attention|2017)/ }).first().click().catch(() => {});
    },
  },
  {
    name: "videos",
    path: "/videos",
    // The form as the clip in the gallery was asked for. It wants the video
    // model loaded: the pipeline and the epilogue are a loaded model's to offer.
    prepare: async (p) => {
      await p.locator("textarea").first().fill("a fox trots through fresh snow towards the camera, snow falling, photograph");
      await p.locator("select", { hasText: "Size preset" }).selectOption({ label: "Large landscape · 1536×1024" });
      await p.locator('input[placeholder="auto"]').fill("3");
      await p.locator('input[type="number"]').nth(3).fill("48");
      await p.getByText("Fixed seed", { exact: true }).click();
      await p.locator('input[placeholder="random"]').fill("3");
      await p.getByText("Epilogue", { exact: true }).click();
    },
  },
  {
    name: "training",
    path: "/training",
    prepare: (p) => p.locator("tr.cursor-pointer").first().click({ timeout: 5000 }),
  },
  {
    name: "benchmarks",
    path: "/benchmarks",
    prepare: (p) => p.locator("tr.cursor-pointer").first().click({ timeout: 5000 }),
  },
];

mkdirSync("static/shots", { recursive: true });
const browser = await chromium.launch({ channel: "chrome" });

for (const theme of ["dark", "light"]) {
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    deviceScaleFactor: 2,
    colorScheme: theme,
  });
  const page = await context.newPage();
  for (const shot of SHOTS) {
    if (only.length && !only.includes(shot.name)) continue;
    await page.goto(base + shot.path, { waitUntil: "networkidle" });
    // A page that could not be prepared is still taken, and says so.
    if (shot.prepare) await shot.prepare(page).catch((e) => console.error(`${shot.name}: ${e.message.split("\n")[0]}`));
    await page.waitForTimeout(700);
    const png = await page.screenshot();
    await sharp(png).resize({ width: 2240 }).webp({ quality: 84 }).toFile(`static/shots/${shot.name}-${theme}.webp`);
    console.log(`${shot.name}-${theme}`);
  }
  await context.close();
}
await browser.close();
