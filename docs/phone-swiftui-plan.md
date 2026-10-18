# Plan: Kvad for iPhone on SwiftUI, over the Rust core

Written 2026-10-18. It replaces *The choices* and *How it is built* in
[phone-plan.md](phone-plan.md). The rest of that plan stands: the screens,
the design brief, Security, and the phases from Speak on.

## Why the move

Slint draws every pixel itself. On the iPhone that meant rebuilding by hand
what UIKit gives every app, and each rebuild met a quirk. This is what one
day on a real phone found:

- **The keyboard.** Its frame is sampled frame by frame, and the samples
  change size while it slides. A keyboard dismissed mid-slide was never
  reported as gone.
- **Text measuring.** A list measured a long wrapped reply at a quarter of
  its height, so the reply would not scroll.
- **Scrolling.** There was no rubber band, no momentum and no keyboard
  dismissal, so a hand-written `Scroller` stood in for them. It flickers on a
  120 Hz phone.
- **The composer.** It stayed one line tall whatever was typed.

Each was fixed or patched, and the next one was always close behind.
SwiftUI on UIKit has none of them: `ScrollView`, `List`, the keyboard
avoidance and `TextField(axis: .vertical)` are Apple's own.

The owner asked for Rust, and for an app that looks like a phone app. Rust
stays where it does the work: the tunnel, pairing, the client and the
Keychain. Swift does only what the screen shows.

## The split

| Rust (`apps/phone/core`) | Swift (`apps/phone/ios/Kvad`) |
|---|---|
| Connect: a LAN address, a `kvad://` name through the tunnel, a pairing link that pairs first | Every screen: Connect, Chat, Models, More, Network test, the model sheet |
| `api.rs` as it is: health, models, the chat as a stream, unpair, vouchers | The keyboard, scrolling, the multi-line composer, all from SwiftUI |
| `pairings.rs` and `secrets.rs` as they are: the same Keychain item, so the phone's pairing survives the move | `kvad://` links (`.onOpenURL`) and coming back to the foreground (`scenePhase`), which `ios.rs` patched into Slint's delegate |
| The chat's turns and its stop flag | The pasteboard, the idle timer and the device name, from UIKit directly |
| The Network test's run: relay-probe's dialler on its own tokio runtime | Later: scanning the QR code in the app (VisionKit), not only from the Camera |

None of today's Rust is UI except `main.rs`'s wiring, `probe.rs`'s
`set_*` calls and `ios.rs`. Those three are what change.

## The boundary: messages, not bindings

The core has three C functions, and every message is one line of JSON:

```c
// The core starts its threads; every event comes back through `on_event`,
// on any thread, as a NUL-terminated JSON object the core frees afterwards.
void kvad_core_start(void *context, void (*on_event)(void *context, const char *json));
// A command, as a JSON object. It returns at once; the answers are events.
void kvad_core_send(const char *json);
// The core's version, for the About screen.
const char *kvad_core_version(void);
```

Commands, from Swift:

```json
{"connect": {"url": "kvad://bdbc…", "key": ""}}
{"chat": {"model": "HuggingFaceTB/SmolLM2-135M-Instruct", "text": "Hi"}}
{"stop": {}}            {"new_chat": {}}        {"models": {}}
{"disconnect": {}}      {"unpair": {}}          {"network_changed": {}}
{"probe_start": {"ticket": "kvadprobe:1:…", "secs": 60}}   {"probe_stop": {}}
```

Events, from Rust:

```json
{"connected": {"name": "kvad://bdbc…", "version": "0.20.0", "who": "andre", "role": "admin", "path": "lan", "paired": true}}
{"connect_failed": {"error": "This phone is not paired with …"}}
{"models": [{"id": "…", "kind": "chat", "resident": true}]}
{"piece": {"text": " the"}}     {"reply_done": {}}      {"chat_failed": {"error": "…"}}
{"path": {"now": "relay"}}      {"probe": {…}}          {"probe_done": {…}}
```

**Why not UniFFI.** I suggested UniFFI before writing this down. It
generates typed Swift for a Rust API, and it is the industry choice for
that. The surface here is small, though: a dozen commands, and events that
are already JSON-shaped. With one JSON line each way, the core needs no new
crate (`serde_json` is in already), and Xcode's build needs no code
generator. The `unsafe` code is three functions and one callback. A
Rust test can drive the core by those messages alone, with no simulator.
The cost is that the messages are not type-checked across the line. Two
things keep them honest: Swift's `Codable` types decode exactly what the
core's `serde` types encode, and one fixture file holds every message,
which both sides' tests read. If the API outgrows this, UniFFI can take over
behind the same Swift types.

**Threads.** The core runs each command on a worker thread, as `main.rs`
does today, and holds the tunnel's runtime. The callback can come from any
thread. Swift moves each event to the main actor before an `@Observable`
model changes, and SwiftUI redraws from that.

## What the app will look like in code

```
apps/phone/
  core/                 the Rust: a staticlib, its own workspace as now
    Cargo.toml          ureq, serde_json, kvad-tunnel, relay-probe, iroh, tokio
    src/lib.rs          the three C functions; commands in, events out
    src/session.rs      today's main.rs logic, without Slint: open, join, through, chat
    src/api.rs          unchanged
    src/pairings.rs     unchanged
    src/secrets.rs      unchanged
    src/probe.rs        the run, with events for the screen's set_* calls
    tests/messages.rs   every command and event against messages.json
  ios/
    project.yml         a SwiftUI app; links libkvad_phone_core.a
    build.sh            builds the staticlib for the simulator or the device
    Kvad/KvadApp.swift  the App, scenePhase, onOpenURL
    Kvad/Core.swift     the bridge: start, send, events onto the main actor
    Kvad/Model.swift    @Observable: connection, chat, models, probe
    Kvad/ConnectView.swift  ChatView.swift  ModelsView.swift  MoreView.swift  NetworkTestView.swift
  core/kvad_core.h      the three C functions, for Swift's bridging header
  messages.json         every message, read by the tests on both sides
```

What goes away: `ui/`, `build.rs`, Slint and its renderer, `ios.rs` and the
ObjC runtime calls in it, and the Slint scene manifest in `project.yml`.
The About screen's Slint attribution goes too. Kvad stays MIT.

## Steps

1. **The core.** Move `main.rs`'s connect, pair, chat and unpair logic into
   `core/src/session.rs`, and `probe.rs`'s screen updates into events. Add
   the three C functions and the message tests. Slint still builds from the
   same modules until step 3, so the phone keeps working in between.
   *Done, 2026-10-18:* `apps/phone/core`, a member of apps/phone's
   workspace. The Slint app calls its `session` functions; its Network test
   screen still runs the probe itself, until step 3 deletes it. Eight
   message tests, and one that chats with a running kvad-serve
   (`--ignored`). A debug build of the static library is 430 MB, nearly all
   of it debug info, which the app's link drops.
2. **The SwiftUI app.** Bring the screens to parity with today, and keep the
   design brief: tab bar, large titles, grouped lists, bubbles, the reply
   streaming in, light and dark, and Dynamic Type for free. Use the SwiftUI
   tools in place of the patches:
   - `.scrollDismissesKeyboard(.interactively)`;
   - `defaultScrollAnchor(.bottom)` for the chat's follow;
   - `TextField(axis: .vertical).lineLimit(1...5)` for the composer.
3. **Slint out.** Delete the Slint UI, update phone-plan.md and the README,
   and close [#271](https://github.com/bisand/kvad/pull/271) unmerged. Its
   findings are written down above.
4. **On the phone.** In the simulator first, then on the iPhone 14 Pro Max:
   - scrolling and the keyboard, against the list of what went wrong;
   - pairing from the Camera;
   - LAN to relay when Wi-Fi goes off;
   - a reconnect on mobile data;
   - the two field tests still open: pairing on mobile data alone, and a
     network change while the app is in the background.

Each step is its own PR, merged on a green check.

## What is given up

- **Running the UI on the Mac** with `cargo run`. Xcode's SwiftUI previews
  replace it, along with the simulator. The core still builds and is tested
  on the Mac.
- **Android from the same UI code.** The core carries over to an Android
  app, behind the same messages over JNI. The UI would be Kotlin, as this
  one is Swift.
- **"All Rust."** The UI is Swift, roughly the size `app.slint` is now.

## Open questions

- Keep the Keychain in Rust (`security-framework`), or move it to Swift?
  Rust, for now: the same item, and the existing pairing survives.
- Scan the QR code in the app at step 2, or later? Later, since the Camera
  already opens `kvad://` links.
