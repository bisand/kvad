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
  to add, after a blank line when it begins a paragraph, and `speak_partial {text, lag}`, the guess after it, which
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

## Hearing the far end of the table

A 19-minute meeting recorded with the phone on the table had stretches of
17 to 24 s with nothing settled, and in them the guess was empty: small
wrote nothing while a quieter voice across the table spoke, then the whole
sentence at once. The file endpoint left those passages out altogether.
The take was quiet. Its 50 ms blocks were at -40 dBFS at the median, -26
at the 90th percentile (the voices near the phone) and -54 at the 10th
(the room). Whisper reads loudness as it comes: its log-mel has no gain of
its own, and a voice 15 dB under the near ones was too faint for small.

**Meeting**, the default in the Speak sheet, brings quieter voices up
before the take is written: `apps/phone/core/src/level.rs`. It measures
50 ms at a time, takes a block as voice when it is 8 dB over the quietest
block of the last 3 s, and follows the voice's level in decibels, up to a
louder one in a block or two, down to a quieter one over about a second.
The gain brings that level toward -26 dBFS, by at most 18 dB (8 times). It
never turns a voice down, holds between voices so that the room is raised
no more than the last voice was, and a block's peak caps it so that nothing
clips. **Close by** is the sound as the microphone gives it, as before. The
gain is the core's, on the take's file, so live, the file endpoint after a
dropped link and *Transcribe again* all hear the same audio. The pauses
that begin paragraphs are found before it.

Measured on that meeting with `live_session` (`+level` runs the take
through the same `level.rs`), nb-whisper-small at gpu-q8 on an M5 Pro, at
real time with an update a second. The reference is nb-whisper-large's file
transcript of the take as recorded, 2023 words. Large hears the far voices
either way: with the gain it writes 1983 words, 1826 of them the same.

| small, gpu-q8 | words | of large's 2023 | settled, median / p90 / max | longest with nothing settled |
|---|---|---|---|---|
| live, no gain, run 1 | 1791 | 1248 | 3.79 / 10.45 / 57.0 s | 57.1 s |
| live, no gain, run 2 | 1782 | 1258 | 3.72 / 9.65 / 96.8 s | 36.9 s |
| live, Meeting (gain at most 8) | 2012 | **1429** | 3.35 / 9.45 / 38.3 s | 41.0 s |
| live, gain at most 16 | 1957 | 1341 | 3.57 / 11.85 / 32.9 s | 30.6 s |
| file, no gain | 1332 | 1008 | | |
| file, gain at most 8 | 1630 | 1158 | | |
| file, gain at most 16 | 1830 | 1308 | | |

The two runs with no gain agree within 1%. Meeting's live text has 14%
more of large's words, and they settle sooner. A file transcript is one
decode, and it swings by ±10% with any change to the audio: a plain gain of
2, 4 and 8 times gave 1163, 1015 and 1202. So the file rows say only that
any gain helps. At most 16 times did best on the file, but not live, so the
cap is 8. The share of 4-grams written twice stayed at 0.2 to 0.5% in
every case: the gain does not make small invent text over the room's
noise. With the Mac's `say -v Nora`, every other sentence 24 dB under the
rest over a noise floor, the live text kept 135 of the file's words in
place of 97 to 104.

*Not measured:* iOS's own ways of hearing. A debug build has a *Trial*
picker in the Speak sheet, and `KVAD_PHONE_MIC` for a launch from the Mac,
for `measurement` (none of iOS's processing), `voice` (voice processing:
its own gain and noise suppression, made for one near voice), and the
`front`, `back` and `bottom` microphones with no direction. Each records
with no gain, so `+level` can be tried on the same take afterwards.
`KVAD_PHONE_RECORD=<seconds>` records that long and stops, for a room test
driven by `devicectl`. One run of radio has been recorded and not yet
measured. Measuring these needs a room test that is the same every time:
the same recording played from the same place, and the phone not moved.

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
2. **The Speak tab.** *Built 2026-10-19.* The microphone, the audio session, the background
   mode, interruptions, the live text, the list of takes, copy, share and
   delete, the model and language sheet. Then on the iPhone:
   - A take on Wi-Fi.
   - One on 5G alone.
   - One that starts on Wi-Fi and ends on 5G.
   - One with the screen locked halfway.
   - One with the Mac's server stopped and started again inside the minute.
   - One with the phone in Airplane Mode for 20 s mid-take.

   *As built:*
   - `Microphone.swift`: `AVAudioSession` set to `.record` only while a
     take is being recorded, `AVAudioEngine`'s input tap, and
     `AVAudioConverter` to 16 kHz mono 16-bit, made again when the input
     changes.
   - The take's folder is made by the app with `completeUnlessOpen` and
     `isExcludedFromBackup`.
   - Copy, *Transcribe again* (with the model chosen for Speak), *Share the
     recording* and Delete are in the take's top bar. A bottom bar does not
     show under a tab bar.
   - A debug launch can record from a WAV in place of the microphone
     (`KVAD_PHONE_SPEAK`), for the simulator.

   *Found by the field runs.* The first four went as they should. The fifth
   showed three things:
   - A server restarted is a session lost, not resumed: the server holds
     sessions in memory, and the phone is told `session_not_found`. At
     first the phone recorded on and sent the take whole when it stopped,
     which lost nothing, but left the screen still for the rest of the take.
     It now begins a new session at the last settled word, sends what was
     said since from the file, and the words come live again. A Mac out of
     reach for longer than the minute is tried on and on, the same way. Only
     a take stopped while the Mac is still out of reach is sent whole.
     Resuming is for a link that drops while the server stays up, which is
     the Airplane Mode run.
   - The tunnel never dialled again. `blocking::Tunnel` held its one QUIC
     connection for good, so after a restart every stream failed until the
     phone was connected again by hand: Chat, and the take's file, as well
     as live. It now dials again on the next `open` once the connection has
     ended. `a_client_outlives_a_server_restart` in `crates/tunnel` restarts
     a server with nothing said. The client gets through again 31.6 s later,
     which is QUIC's idle timeout. On the iPhone, it dialled again 22 s
     after the server came back.

   *Asked for after the field runs:*
   - **Meeting and Close by.** In a meeting, the quieter voices farther
     from the phone came out late or not at all. Meeting, the default,
     brings them up before the take is written (*Hearing the far end of
     the table*).
   - **Paragraphs.** A pause of 1.5 s or more begins a new one. The core
     finds pauses in the audio as it comes (`speak::Quiet`: loudness every
     50 ms against a noise floor that a voice cannot raise), and a word
     settled after one ends begins its paragraph. A take keeps its pauses
     in its record (`pauses`, seconds, heard before any gain), and a take
     sent whole, or transcribed again, begins a paragraph at the gap between
     Whisper's segments nearest each one, within a second
     (`api::paragraphs`). The file endpoint is asked for timestamps for it:
     without them each segment is a whole 30-second window and the windows
     meet, so a take transcribed again came back as one paragraph (field
     run, 2026-10-20). With them the segments are sentences, but their times
     are a guess: on a test recording with 3 s of silence between two
     sentences, nb-whisper-small's two segments met exactly, at 6.62 s. A
     take kept before its pauses were has them found again in its file. A new speaker
     would begin one too, once kvad can tell speakers apart
     (docs/diarization-plan.md, *Speakers*, not built yet).
   - **Conversations kept, as takes are.** Chat's conversations are kept on
     the phone (`chats.rs`, `Application Support/Chats`, the same protection
     and no backups), listed from the top left of the Chat tab, opened again
     with the model they were had with, and deleted with a swipe.
3. **Files in, and again with a better model.** "Transcribe again" with a
   chosen model on the file endpoint, streamed. Taking audio from elsewhere:
   the Files picker, and the share sheet's *Open in Kvad* for a Voice Memo.
   Subtitles (SRT, VTT) from the segments, as the web UI writes them.
   **Takes stored smaller.** A take is 16 kHz 16-bit PCM, 1.9 MB a minute
   and 115 MB an hour; a 19-minute meeting was 36 MB. It has to be PCM
   while it is recorded, since the core reads it back to resend what a
   dropped link lost. Once it ends, it is converted on the phone with
   Apple's own encoder to AAC in `.m4a` at about 32 kbit/s, about 0.24 MB a
   minute, which the server's file endpoint reads through ffmpeg as it reads
   any recording. Opus would be smaller still at the same quality for
   speech, if the server is shown to read it. The WAV goes once the
   compressed file is written and read back.
   *Not MP3.* AAC is the later and more efficient codec: at a low rate
   such as 32 kbit/s it keeps more of a voice than MP3 does with the same
   bits. iOS has no MP3 encoder either (it reads MP3, it does not write
   it), so MP3 would mean bringing in LAME, a C library under the LGPL,
   against what kvad lets in. AAC is in `AVAudioConverter`, and `.m4a` plays
   and shares everywhere a Voice Memo does. Before the WAV is let go, the
   19-minute meeting is transcribed from both and compared, as the gain
   was: that is the check that the format costs Whisper nothing.

## Open questions

- **Opus on the uplink.** 16-bit PCM at 16 kHz is 256 kbit/s, which 5G and
  the relay carry easily. A weak 4G signal might not. It waits, as
  speech-plan says, until a measured take on a bad uplink says it is needed.
- **How long takes are kept.** Until deleted, to begin with. An automatic
  clear-out after so many days can come later, once there is a Settings
  screen to put it on.
