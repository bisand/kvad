# Plan: Speak, on the phone

Written 2026-10-19. It is Phase 2 of [phone-plan.md](phone-plan.md), and the
phone's end of *Live, from the phone* in [speech-plan.md](speech-plan.md).
The server's end is built: `/v1/realtime` takes audio as it is recorded and
sends settled words and a replaceable partial back, and it holds a session
for a minute after the link drops. On small at gpu-q8 a word shows about
0.4 s after it is spoken and settles about 1.6 s after. What is left is the
phone's half: the microphone, the socket, the recording kept on the phone,
the resume, and the screen.

## What it does

- **Tap, talk, read.** One big button on the Speak tab. While it records,
  the words come up under it: settled text as it is, the partial greyed
  after it, and how far behind the Mac is when that is more than a couple
  of seconds. Tap again to stop. The commit settles the rest, and the take
  is done.
- **The screen can go off.** Recording carries on with the phone locked or
  the app in the background (the `audio` background mode). iOS shows its
  own orange microphone dot and the Dynamic Island the whole time. A phone
  call pauses the take, and it carries on after.
- **Every take is kept on the phone**, the audio and its transcript, in a
  list under the button: copy, share, delete. The audio is what makes the
  live path safe. Whatever happens to the link, nothing said is lost
  (*When the link drops*).
- **Again, with a better model.** A take can be sent whole to
  `POST /v1/audio/transcriptions` with any speech model the Mac has.
  Large is too slow to be live but writes better text than small, so a
  meeting can be followed on small and read back on large.
- **The model and the language** come from a sheet like Chat's: the speech
  models the server lists, small first when it is there, and the language
  as Whisper's code, or found by the model.

## The split

As in [phone-swiftui-plan.md](phone-swiftui-plan.md): Rust does the work,
Swift does what only the phone can.

| Rust (`apps/phone/core`) | Swift (`apps/phone/ios/Kvad`) |
|---|---|
| The realtime session: the WebSocket, the events, the offsets | The microphone: `AVAudioSession`, `AVAudioEngine`'s input tap, and `AVAudioConverter` to 16 kHz mono 16-bit |
| The take's file: writing the audio as it comes, and reading it back to resend | Where the file lives, and its protection class, set when it is made |
| The resume after a drop, and the file endpoint after the minute | Interruptions (a call, Siri) and route changes (AirPods in or out) |
| The take's transcript, saved beside the audio | The Speak tab, the list of takes, the share sheet |

**The audio crosses as bytes, not JSON.** A fourth C function,
`kvad_core_audio(const int16_t *samples, size_t count)`, takes 100 ms of
samples at a time from the input tap's thread. JSON would mean base64 ten
times a second for nothing. Everything else stays a message:

- Commands: `speak_start {model, language, folder}`, `speak_stop`,
  `speak_cancel`, `takes {folder}`, `take_delete {folder, id}`,
  `retranscribe {folder, id, model, language}`.
- Events: `speak_started {id}`; `speak_settled {text, lag}`, settled text
  to add, and `speak_partial {text, lag}`, the guess after it, which
  replaces the last; `speak_link {state, why}`, the link as `live`,
  `reconnecting`, `offline` or `sending`; `speak_done {take}`,
  `speak_failed {error, kept}`, `speak_cancelled {id}`; `takes {takes}`,
  `take_deleted {id}`; `retranscribe_piece {id, text}`,
  `retranscribe_done {take}`, `retranscribe_failed {id, error}`.
- A take, in `speak_done`, `takes` and its JSON file: `id` (the
  millisecond it began), `started`, `seconds`, `model`, `language`,
  `text`, `how` (`live`, `file`, or empty when there is no transcript yet)
  and `server`.

`messages.json` lists them, and both sides' tests read it, as they do now.

**Why Swift owns the microphone.** `cpal` was the phase plan's first guess.
But the work on iOS is not reading samples: it is the audio session's
category and mode, its interruptions, the route changes and the background
mode, and those are `AVAudioSession`'s and nothing else's. In Swift that is
a page of code against Apple's own types. Through `cpal` it would be the
same calls made by hand through `objc2`, with `cpal` adding nothing.

## The socket

**`tungstenite` 0.29, client only, default features off.** It is the crate
the server's own `axum` WebSocket runs on, already in kvad's lockfile, and
the Rust ecosystem's standard. It runs a handshake over any `Read + Write`
stream. For a `kvad://` server that is the tunnel's `open()`, which gives
one QUIC stream as a `UnixStream`. For a LAN address it is a `TcpStream`.
Written by hand, a client is the handshake, SHA-1 to check the accept key,
base64, masking and framing. That is not much, but it is protocol code with
an RFC behind it, which a standard crate gets right for free.

**One thread, never blocked for long.** The socket has a 20 ms read timeout,
and a loop sends any audio that is waiting and reads any event that has
come. The two directions do not wait on each other, which is what
relay-plan asks of the return channel.

**What goes up**, in OpenAI's shapes with Kvad's two additions, as the
server reads them:

1. `session.update`: the model, the language, `input_audio_format: pcm16`
   at 16 kHz.
2. `kvad.input_audio_buffer.offset {offset}` once a connection, then binary
   frames of PCM.
3. `input_audio_buffer.commit` on stop, then a close once `completed` is in.

## When the link drops

Built on what the server already does (speech-plan, *The session held when
the link drops*):

- The take is written to its file before anything is sent, so the file is
  always ahead of the socket.
- A socket that fails is opened again with `?resume=<id>&offset=<n>`, `n`
  being the offset of the last settled event the phone has. The server says
  how many samples it has, and the phone resends from there out of the
  file. Settled events that come again are known by their `event_id` and
  dropped. The screen says *Reconnecting…*, and recording never stops.
- A network change already reaches the tunnel at once (`network_changed`).
  Mostly QUIC carries the stream across. The resume is for when it does not.
- After the minute, or with the Mac asleep, the take goes on to the file
  and nothing else. On stop it is sent whole to `/v1/audio/transcriptions`
  once the Mac answers, and its transcript replaces the live one.
- A take that never reaches a Mac stays in the list as *Not transcribed*,
  with a button to send it.

## Security and privacy

- **The microphone is on only while a take is being recorded.** The audio
  session is set active on start and inactive on stop, so iOS's dot goes
  out with it.
- **The takes are under `Application Support/Takes`**, with the protection
  class `completeUnlessOpen`. That keeps them out of reach while the phone
  is locked, yet lets a file opened while it was unlocked go on being
  written after the screen goes off. `complete`, which the phase plan named,
  would stop a locked phone's recording mid-sentence.
- **Not in iCloud or computer backups.** `isExcludedFromBackup` is set on
  the folder. A recording leaves the phone only for the Mac it is paired
  with, and only through the tunnel or the LAN.
- **No other speech recogniser.** Not Apple's `SFSpeechRecognizer`, and no
  dictation as a fallback: either would send audio somewhere other than the
  owner's Mac.
- **Nothing is kept on the server**, which is already true of both
  endpoints.
- **Least privilege** (relay-plan): a phone paired *for transcription*
  reaches these two endpoints and nothing else. Speak works with today's
  full pairing. The scoped pairing is relay-plan's to build, and Speak
  will not need to change for it.

## Steps

Each is a PR, merged on a green check, as the SwiftUI move was.

1. **The core's session.** *Built 2026-10-19.* `speak.rs` in `apps/phone/core`: the socket over
   both kinds of stream, the take's file (a WAV, so it opens anywhere), the
   resume, the file endpoint, the takes list, and the messages. The tests
   use a scripted realtime server in the test, written on `tungstenite`'s
   server half, that settles a word a second. They check:
   - A take played in gives its words, each once.
   - A socket cut mid-word and resumed gives the take of a session never
     cut.
   - A server gone past the minute leads to the file endpoint.
   - Stop during a reconnect loses nothing.

   An `#[ignore]`d test plays a recording into a real server at real time,
   then sends the same take to the file endpoint, and compares the two.
   A test in `crates/tunnel` runs a WebSocket through the tunnel, direct
   and relayed, which relay-plan listed as not yet done.

   *As built*, beyond the above:
   - The take's thread reads its audio back out of the file, through a
     handle opened when the take was made, so live and resent audio are one
     path.
   - A server that answers the upgrade with anything but 101, 401, 403,
     429 or 5xx has no `/v1/realtime`. kvad-serve from before it answers
     with its web page. The take goes offline at once, not after a minute
     of retries. An `https://` LAN address goes offline too: the socket is
     plain `ws://`, and TLS for it waits until someone runs one.
   - A socket that owes an answer (audio or a commit sent) and hears
     nothing for 15 s is taken for dead and opened again. That covers a
     tunnel stream left hanging when the network goes.
   - A resumed session on an item past the take's first means its commit
     landed, so it is not sent twice.
   - A WAV with no JSON beside it, from an app stopped mid-take, has its
     header put right when the takes are listed, and is listed as not
     transcribed.
   - Measured on an M5 Pro with nb-whisper-small at gpu-q8, against a
     kvad-serve built from master: 11.7 s of Norwegian from `say -v Nora`,
     played in at real time, came back whole 0.9 s after the stop. All 41
     words were the file endpoint's.
2. **The Speak tab.** The microphone, the audio session, the background
   mode, interruptions, the live text, the list of takes, copy, share and
   delete, the model and language sheet. Then on the iPhone:
   - A take on Wi-Fi.
   - One on 5G alone.
   - One that starts on Wi-Fi and ends on 5G.
   - One with the screen locked halfway.
   - One with the Mac's server stopped and started again inside the minute.
3. **Files in, and again with a better model.** "Transcribe again" with a
   chosen model on the file endpoint, streamed. Taking audio from elsewhere:
   the Files picker, and the share sheet's *Open in Kvad* for a Voice Memo.
   Subtitles (SRT, VTT) from the segments, as the web UI writes them.

## Open questions

- **Opus on the uplink.** 16-bit PCM at 16 kHz is 256 kbit/s, which 5G and
  the relay carry easily. A weak 4G signal might not. It waits, as
  speech-plan says, until a measured take on a bad uplink says it is needed.
- **How long takes are kept.** Until deleted, to begin with. An automatic
  clear-out after so many days can come later, once there is a Settings
  screen to put it on.
