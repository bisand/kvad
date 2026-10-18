//! `/v1/audio/transcriptions` and `/v1/audio/translations`: a recording in,
//! what is said in it out, from a Whisper ([`kvad::speech`]).
//!
//! OpenAI's shape, for the reason the other `/v1` routes are: a client that
//! already sends recordings to OpenAI sends them here by changing a URL. A
//! form, with the recording as `file` and `model`, `language`, `prompt`,
//! `response_format` and `temperature` beside it. All five formats are
//! [`kvad::speech::Transcript::render`]'s, so the server and `kvad
//! transcribe --local` write the same bytes for the same recording.
//!
//! `timestamp_granularities[]=word`, with `verbose_json`, times every word
//! as well ([`kvad::speech::align`]): a `words` array beside `segments`, as
//! OpenAI's. It costs one more pass of the decoder over each window's text.
//!
//! What OpenAI has no field for: `seed`, and `timestamps`, which asks for
//! timestamp tokens whatever the format. They change what is written, not
//! only how it is laid out — a Whisper decodes a window differently when it
//! is timing it — and `srt`, `vtt` and `verbose_json` always ask for them,
//! since their segments are timed.
//!
//! `stream: true` answers with OpenAI's events for a streamed transcription,
//! `transcript.text.delta` as each 30-second window is done and
//! `transcript.text.done` at the end. An hour of audio is minutes on the CPU,
//! and a client watching it come is a client that knows it works. Each delta
//! carries the window's timed segments under `kvad`, which is what lets the
//! command line print them as it does here.
//!
//! Nothing is kept: neither the recording nor what was said in it. An image
//! or a video is something somebody made and comes back for; a transcript is
//! the answer to a request.
//!
//! `temperature` is OpenAI's: 0, or none, is greedy with the fallback to
//! warmer temperatures for a window that fails, and anything above is the
//! one temperature tried. `kvad transcribe --local` reads it the same way.

use crate::api::{blocking_or, Fail};
use crate::auth::{Identity, State};
use crate::models::{sse, stream};
use crate::scheduler::{Entry, Heard, Kind};
use crate::videos::Fields;
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, State as St};
use axum::http::{header, HeaderMap};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::Router;
use kvad::speech::{Format, Request, Task};
use serde_json::json;

pub fn routes() -> Router<State> {
    Router::new()
        .route("/v1/audio/transcriptions", post(transcriptions))
        .route("/v1/audio/translations", post(translations))
        .layer(DefaultBodyLimit::max(MAX_FILE + (1 << 16)))
}

/// The largest recording a request may carry. OpenAI's is 25 MB, which is
/// under half an hour of MP3 at 128 kbit/s, and a meeting is longer than
/// that. 200 MiB is three and a half hours of it; the body is held whole
/// while it is read, and then once more as the file `ffmpeg` opens.
const MAX_FILE: usize = 200 << 20;

pub async fn transcriptions(who: Identity, headers: HeaderMap, St(state): St<State>, body: Bytes) -> Result<Response, Fail> {
    listen(who, &headers, state, &body, Task::Transcribe).await
}

/// Into English, whatever is spoken. OpenAI's takes no `language`; this one
/// does, as the language the recording is in, for a Whisper that guesses it
/// wrong.
pub async fn translations(who: Identity, headers: HeaderMap, St(state): St<State>, body: Bytes) -> Result<Response, Fail> {
    listen(who, &headers, state, &body, Task::Translate).await
}

/// A request, read and checked before any of it is decoded.
#[derive(Debug, PartialEq)]
struct Asked {
    model: Option<String>,
    format: Format,
    request: Request,
    stream: bool,
    /// The recording, and the name it was sent under.
    file: (String, Vec<u8>),
}

impl Asked {
    fn read(f: &Fields, task: Task) -> Result<Asked, Fail> {
        if let Some((name, _)) = f.files.iter().find(|(n, _)| n != "file") {
            return Err(Fail::bad(format!("{name}: the only file this takes is `file`, the recording")));
        }
        let Some((_, bytes)) = f.files.iter().find(|(n, _)| n == "file") else {
            return Err(Fail::bad("no recording: send it as the form's `file`, multipart/form-data"));
        };
        if bytes.is_empty() {
            return Err(Fail::bad("file is empty"));
        }
        let filename = f.filenames.iter().find(|(n, _)| n == "file").map(|(_, f)| f.clone()).unwrap_or_default();
        let format = match f.text(&["response_format"]) {
            None => Format::Json,
            Some(name) => Format::parse(name.trim())
                .ok_or_else(|| Fail::bad(format!("response_format is json, text, srt, verbose_json or vtt, not {name}")))?,
        };
        let temperature: Option<f32> = f.number(&["temperature"])?;
        if let Some(t) = temperature {
            if !(0.0..=1.0).contains(&t) {
                return Err(Fail::bad(format!("temperature is between 0 and 1, not {t}")));
            }
        }
        let timed = matches!(format, Format::Srt | Format::Vtt | Format::VerboseJson);
        let mut words = false;
        for g in f.texts(&["timestamp_granularities[]", "timestamp_granularities"]) {
            match g.as_str() {
                "word" => words = true,
                "segment" => {}
                other => return Err(Fail::bad(format!("timestamp_granularities are word and segment, not {other}"))),
            }
        }
        if words && format != Format::VerboseJson {
            return Err(Fail::bad("word timestamps are written in verbose_json: set response_format to it"));
        }
        let request = Request {
            language: f.text(&["language"]).map(|l| l.trim().to_ascii_lowercase()),
            task,
            timestamps: timed || f.flag("timestamps")?.unwrap_or(false),
            prompt: f.text(&["prompt"]),
            // 0 is where OpenAI's fallback starts; see the module docs.
            temperature: temperature.filter(|&t| t > 0.0),
            seed: f.number(&["seed"])?,
            words,
            ..Request::default()
        };
        Ok(Asked {
            model: f.text(&["model"]),
            format,
            request,
            stream: f.flag("stream")?.unwrap_or(false),
            file: (filename, bytes.clone()),
        })
    }
}

async fn listen(who: Identity, headers: &HeaderMap, state: State, body: &[u8], task: Task) -> Result<Response, Fail> {
    let asked = Asked::read(&Fields::read(headers, body)?, task)?;

    // Decoded before the model is found, which may mean loaded: a file that
    // is not a recording is a 400 now, and not after gigabytes of Whisper
    // have been read in to listen to it.
    let ffmpeg = crate::videos::ffmpeg();
    let (name, bytes) = asked.file;
    let audio = blocking_or(move || {
        kvad::speech::audio::from_bytes(&bytes, &name, ffmpeg.as_deref()).map_err(|e| match ffmpeg {
            None => Fail::bad(
                "file: only a WAV is read without ffmpeg, and this server has none; `[videos] ffmpeg` in kvad.toml",
            ),
            // ffmpeg's own words name the file it was given, which is a path
            // on this machine: they go to the log, not the client.
            Some(_) => {
                tracing::info!("a recording refused: {e}");
                Fail::bad("file is not a recording this server can read: try WAV, MP3, M4A, FLAC, OGG or a video's sound")
            }
        })
    })
    .await?;
    if audio.is_empty() {
        return Err(Fail::bad("file has no sound in it"));
    }
    let resident = crate::openai::resident_for(&state, asked.model.as_deref(), Kind::Speech).await?;

    let (_, mut heard) = state.engine.transcribe(who.id, Entry::Request, &resident.key, audio, asked.request)?;
    if asked.stream {
        return Ok(streamed(heard).into_response());
    }
    let transcript = loop {
        match heard.recv().await {
            Some(Heard::Window(_)) => continue,
            Some(Heard::Done(t)) => break *t,
            Some(Heard::Refused(why)) => return Err(Fail::bad(why)),
            Some(Heard::Failed(why)) => return Err(Fail::internal(why)),
            None => return Err(Fail::internal("the engine stopped before it answered")),
        }
    };
    Ok(([(header::CONTENT_TYPE, asked.format.content_type())], transcript.render(asked.format, task)).into_response())
}

/// OpenAI's streamed transcription: a delta a window, then the whole text.
///
/// Dropping `heard`, which a client that hangs up does, is what tells the
/// scheduler to stop after the window it is on.
fn streamed(mut heard: tokio::sync::mpsc::Receiver<Heard>) -> impl IntoResponse {
    let (events, rx) = tokio::sync::mpsc::channel(16);
    tokio::spawn(async move {
        while let Some(h) = heard.recv().await {
            let (event, last) = match h {
                Heard::Window(p) => {
                    let delta = json!({
                        "type": "transcript.text.delta",
                        "delta": p.segments.iter().map(|s| s.text.as_str()).collect::<String>(),
                        "kvad": {
                            "done": p.done,
                            "total": p.total,
                            "segments": p.segments.iter().map(segment).collect::<Vec<_>>(),
                        },
                    });
                    (sse("transcript.text.delta", &delta), false)
                }
                Heard::Done(t) => {
                    let done = json!({
                        "type": "transcript.text.done",
                        "text": t.text.trim(),
                        "kvad": { "language": t.language, "duration": t.duration },
                    });
                    (sse("transcript.text.done", &done), true)
                }
                Heard::Refused(why) | Heard::Failed(why) => (sse("error", &json!({ "error": why })), true),
            };
            if events.send(event).await.is_err() || last {
                return;
            }
        }
    });
    stream(rx)
}

/// A segment in a delta, with its words when they were asked for.
fn segment(s: &kvad::speech::Segment) -> serde_json::Value {
    let mut v = json!({ "start": s.start, "end": s.end, "text": s.text });
    if !s.words.is_empty() {
        v["words"] = s.words.iter().map(|w| json!({ "word": w.word.trim(), "start": w.start, "end": w.end })).collect::<Vec<_>>().into();
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A form as a client sends one, read the way the route reads it.
    fn form(parts: &[(&str, Option<&str>, &[u8])]) -> Fields {
        let mut body = Vec::new();
        for (name, filename, data) in parts {
            body.extend_from_slice(b"--XyZ\r\n");
            match filename {
                Some(f) => body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{f}\"\r\nContent-Type: audio/mpeg\r\n\r\n").as_bytes()),
                None => body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes()),
            }
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(b"--XyZ--\r\n");
        let mut h = HeaderMap::new();
        h.insert(header::CONTENT_TYPE, "multipart/form-data; boundary=XyZ".parse().unwrap());
        Fields::read(&h, &body).unwrap()
    }

    #[test]
    fn a_bare_request_is_json_greedy_with_fallback_and_untimed() {
        let a = Asked::read(&form(&[("file", Some("talk.mp3"), b"ID3"), ("model", None, b"NbAiLab/nb-whisper-tiny")]), Task::Transcribe).unwrap();
        assert_eq!(a.model.as_deref(), Some("NbAiLab/nb-whisper-tiny"));
        assert_eq!(a.format, Format::Json);
        assert_eq!(a.file, ("talk.mp3".to_string(), b"ID3".to_vec()));
        assert_eq!(a.request, Request { task: Task::Transcribe, ..Request::default() });
        assert!(!a.stream);
    }

    /// The timed formats decode with timestamps, as `kvad transcribe` does
    /// for them, and `timestamps` asks for it for any other.
    #[test]
    fn the_formats_with_times_ask_for_timestamps() {
        for (format, timed) in [("json", false), ("text", false), ("srt", true), ("vtt", true), ("verbose_json", true)] {
            let a = Asked::read(&form(&[("file", Some("a.wav"), b"x"), ("response_format", None, format.as_bytes())]), Task::Transcribe).unwrap();
            assert_eq!(a.request.timestamps, timed, "{format}");
        }
        let a = Asked::read(&form(&[("file", Some("a.wav"), b"x"), ("timestamps", None, b"true")]), Task::Translate).unwrap();
        assert!(a.request.timestamps);
        assert_eq!(a.request.task, Task::Translate);
    }

    #[test]
    fn the_other_fields_are_read_as_they_are_meant() {
        let a = Asked::read(
            &form(&[
                ("file", Some("a.wav"), b"x"),
                ("language", None, b"NO"),
                ("prompt", None, b"Grieg, Kygo"),
                ("temperature", None, b"0.4"),
                ("seed", None, b"7"),
                ("stream", None, b"true"),
            ]),
            Task::Transcribe,
        )
        .unwrap();
        assert_eq!(a.request.language.as_deref(), Some("no"));
        assert_eq!(a.request.prompt.as_deref(), Some("Grieg, Kygo"));
        assert_eq!(a.request.temperature, Some(0.4));
        assert_eq!(a.request.seed, Some(7));
        assert!(a.stream);
        // 0 is OpenAI's start of the fallback, not a temperature to keep to.
        let zero = Asked::read(&form(&[("file", Some("a.wav"), b"x"), ("temperature", None, b"0")]), Task::Transcribe).unwrap();
        assert_eq!(zero.request.temperature, None);
    }

    /// OpenAI's `timestamp_granularities[]`, a part a value, or JSON's
    /// list: `word` times every word, and only `verbose_json` has room for
    /// them.
    #[test]
    fn word_timestamps_are_asked_for_as_openai_asks() {
        let words = |parts: &[(&str, Option<&str>, &[u8])]| Asked::read(&form(parts), Task::Transcribe).map(|a| a.request.words);
        let file: (&str, Option<&str>, &[u8]) = ("file", Some("a.wav"), b"x");
        let verbose: (&str, Option<&str>, &[u8]) = ("response_format", None, b"verbose_json");
        assert_eq!(words(&[file, verbose]).unwrap(), false);
        assert_eq!(words(&[file, verbose, ("timestamp_granularities[]", None, b"word")]).unwrap(), true);
        assert_eq!(words(&[file, verbose, ("timestamp_granularities[]", None, b"segment"), ("timestamp_granularities[]", None, b"word")]).unwrap(), true);
        assert_eq!(words(&[file, verbose, ("timestamp_granularities[]", None, b"segment")]).unwrap(), false);
        assert!(words(&[file, ("timestamp_granularities[]", None, b"word")]).unwrap_err().1.contains("verbose_json"));
        assert!(words(&[file, verbose, ("timestamp_granularities[]", None, b"letter")]).unwrap_err().1.contains("letter"));
    }

    #[test]
    fn what_cannot_be_transcribed_is_refused_by_name() {
        let why = |parts: &[(&str, Option<&str>, &[u8])]| Asked::read(&form(parts), Task::Transcribe).unwrap_err().1;
        assert!(why(&[("model", None, b"m")]).contains("`file`"));
        assert!(why(&[("file", Some("a.wav"), b"")]).contains("empty"));
        assert!(why(&[("file", Some("a.wav"), b"x"), ("response_format", None, b"docx")]).contains("docx"));
        assert!(why(&[("file", Some("a.wav"), b"x"), ("temperature", None, b"2")]).contains("between 0 and 1"));
        assert!(why(&[("file", Some("a.wav"), b"x"), ("image", Some("a.png"), b"x")]).contains("only file"));
    }
}
