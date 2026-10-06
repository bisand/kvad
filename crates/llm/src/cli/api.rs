//! The commands only a server can answer.
//!
//! Each group reads like the page of the web UI it matches, and prints a
//! table where the page has one. `--json` prints what the server sent
//! instead, for anything that wants to take it further.

use super::{needs, out, split, unknown};
use crate::Args;
use kvad::client::{self, Auth, Body, Remote};
use serde_json::{json, Value};
use std::collections::HashSet;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

pub const CONVERSATIONS: &str = "usage: kvad conversations [ls]
       kvad conversations show ID
       kvad conversations edit ID [--title T] [--system S]
       kvad conversations rm ID

The ones started with `kvad chat --save`, and in the web UI. Carry one on with
`kvad chat --conversation ID`.";

pub const IMAGES: &str = "usage: kvad images [ls]
       kvad images make PROMPT [--out FILE] [--model MODEL] [--size WxH] [--steps N]
                               [--guidance F] [--negative TEXT] [--seed N]
                               [--lora NAME[:SCALE]]...
       kvad images edit PROMPT --image PICTURE [--mask FILE] [--strength F]
                               [and what `make` takes]
       kvad images rm ID

Pictures made by an image model on the server — SDXL, SD 1.5, FLUX, Qwen-Image.
Every one is kept there, with the settings that made it; `make` also writes it
here, to --out or to image-ID.png. Anything left out is the model's own
default. --lora applies a LoRA fetched with `kvad pull`, at the strength after
a colon (1 if none), to Qwen-Image, FLUX, SDXL or SD 1.5.

`edit` makes a picture from PICTURE: it is noised part of the way and drawn
over from there, so PROMPT says what the result should be, not what to change.
--strength is how far, above 0 and at most 1 (0.75 unless said): near 0 the
picture comes back as it was, and at 1 nothing of it is left. With --mask only
part of it is drawn anew, at strength 1 unless said: where the mask is
transparent, or white if it has no transparency. Without --size the result is
the picture's own shape. SDXL and SD 1.5 edit.";

pub const VIDEOS: &str = "usage: kvad videos [ls]
       kvad videos make PROMPT [--out FILE] [--model MODEL] [--size WxH]
                               [--seconds S | --frames N] [--fps N] [--seed N] [--silent]
                               [--image PICTURE] [--steps N] [--guidance G] [--negative TEXT]
                               [--decoder diffusion|conv] [--pipeline fast|dfr]
                               [--lora NAME[:SCALE]]...
       kvad videos show ID
       kvad videos watch ID    follow it until it ends
       kvad videos get ID [--out FILE]
       kvad videos rm ID

Clips made by a video model on the server — LTX-2.5 — with their sound. A
video takes minutes, and is the server's job from the moment it is asked for:
`make` follows it and writes it here, to --out or to video-ID.mp4, but
stopping the wait does not stop the video. `show` says how far along one is,
`watch` follows it again, `get` fetches it when it is done, and `rm` deletes
it, stopping it if it is still being made. Anything left out is the model's own default;
with no --seconds or --frames, LTX-2.5 chooses the length from the prompt.

--steps, --guidance or --negative runs LTX-2.5's dev model, guided, rather than
its distilled one: 30 steps at guidance 3 unless given, about four times as long.
Its files come from `kvad pull Lightricks/LTX-2.5 --dev` (51 GB).

--lora applies a LoRA fetched with `kvad pull` to every DiT the video's
pipeline runs, at the strength after a colon (1 if none).

--decoder picks how LTX-2.5's latents become frames: diffusion, the default and
the reference's, or conv, lighter and about twice as fast to decode.

--pipeline dfr runs LTX-2.5's production pipeline, DFR: generated keyframes, a
detailing pass and a keyframe-aware decode, slower and finer. --fps above 30
runs it without being asked, doubling a clip at half or a quarter of the rate
(48, 50, 60, 96, 100, 120 fps). Its detailing LoRA is gated on Hugging Face,
and the first DFR video fetches it.

--image starts the video from a picture, which becomes its first frame, scaled
to cover the video's size and cut from the middle. Any format the server's
ffmpeg reads, 20 MiB at most; the server needs ffmpeg for it.";

pub const JOBS: &str = "usage: kvad jobs [ls] [--limit N]
       kvad jobs show ID
       kvad jobs watch ID      follow it until it ends
       kvad jobs cancel ID     ask it to stop
       kvad jobs pictures ID [--out DIR]
                               fetch what a LoRA run drew: the samples of
                               `kvad tune --sample`, into DIR (default: here)

Downloads, training runs, evals and benchmarks, in one history.";

pub const DATASETS: &str = "usage: kvad datasets [ls]
       kvad datasets add FILE [--name NAME]   a text, to train a language model on
       kvad datasets add DIR [--name NAME]    a folder of pictures, for `kvad tune`
       kvad datasets crawl URL --name NAME [--pages N] [--mb F] [--pause MS]
                                           [--drop-rare N] [--same-host]
       kvad datasets show ID
       kvad datasets check ID --model MODEL   would it fit that model's vocabulary
       kvad datasets search ID QUESTION [--k N]
       kvad datasets put ID FILE...           pictures and captions into a folder of them
       kvad datasets get ID FILE [--out PATH] one of them back
       kvad datasets rm ID FILE...            take those out of it
       kvad datasets rm ID

An ID can also be a dataset's name. A picture's caption is the .txt of the same
name: `dog.jpg` and `dog.txt`. A file's name may hold letters, digits, spaces
and `-`, `_` or `.`; anything else in it is sent as `_`. Adding a folder under
the name of one already there adds to it.";

pub const TUNE: &str = "usage: kvad tune --data DIR --name NAME [options]
       kvad tune --dataset ID|NAME --name NAME [options]
       kvad tune options                      what a run can be asked for there

Train a LoRA for SDXL on the server, from a folder of pictures, each with its
caption in a .txt of the same name. --data uploads the folder as a dataset
first; --dataset names one the server has. The LoRA is the server's when it is
done: `kvad images make PROMPT --lora NAME` draws with it.

  --model MODEL       the model it is for (default: SDXL's base)
  --caption TEXT      the caption of every picture that has none
  --size N            pixels a side: 512, 768 or 1024 (default 1024)
  --rank N            (default 16)
  --steps N           (default 1000)
  --lr F              (default 1e-4)
  --eval-every N      steps between measurements (default 100)
  --seed N            (default 1337)
  --sample TEXT       a prompt to draw before the first step and at every
                      measurement; up to four. `kvad jobs pictures ID`
                      fetches them
  --sample-size N     pixels a side of a sample (default 512)
  --sample-steps N    its denoising steps (default 20)

The run is a job: Ctrl-C stops the watching, `kvad jobs cancel ID` the run.
`kvad-gpu tune` trains the same LoRA in the terminal that asks, with no
server, and has --from, --alpha and --holdout, which this does not.";

pub const EVALS: &str = "usage: kvad evals [runs]
       kvad evals show ID
       kvad evals suites
       kvad evals add FILE [--name NAME]      a suite: {\"name\", \"cases\": [{\"prompt\", \"expect\", \"match\"?}]}
       kvad evals edit ID FILE
       kvad evals rm ID
       kvad evals run SUITE MODEL[@BACKEND]... [--max-tokens N] [--seed N]
       kvad evals perplexity DATASET MODEL[@BACKEND]... [--window N]

A model with no @BACKEND is run on --backend, else on the server's default.";

pub const BENCH: &str = "usage: kvad bench [runs]
       kvad bench show ID
       kvad bench run MODEL[@BACKEND]... [--prompt T] [--rounds N] [--tokens N] [--seed N]

A round visits every model once and a run is several rounds, so a machine that
warms up shows as a trend rather than as a winner.";

pub const METRICS: &str = "usage: kvad metrics             what the machine is doing
       kvad metrics requests    [--limit N]  recent requests, and what each route costs
       kvad metrics log         [--limit N]  the server's log tail

For this machine's service, `kvad service logs -f` follows the whole log.";

pub const AUTH: &str = "usage: kvad auth status
       kvad auth login [--name NAME]    sign in, and keep an API key for this server
       kvad auth login --key            keep a key made elsewhere: the web UI's Account
                                        page, or an identity provider's sign-in
       kvad auth logout                 revoke the kept key, and forget it
       kvad auth setup TOKEN            make the first account, with the token the
                                        server printed when it started with none
       kvad auth password               change your password

Keys are kept in credentials.json in the config directory, readable by you only.
KVAD_API_KEY, when it is set, is used instead.";

pub const USERS: &str = "usage: kvad users [ls]
       kvad users add NAME [--role admin|user]
       kvad users edit ID [--role admin|user] [--password]
       kvad users rm ID

An ID can also be a name. Passwords are asked for, never taken as flags.";

pub const SESSIONS: &str = "usage: kvad sessions [ls]
       kvad sessions rm HASH       the start of one is enough";

pub const KEYS: &str = "usage: kvad keys [ls]
       kvad keys add NAME          the key is shown once
       kvad keys rm ID";

pub const API: &str = "usage: kvad api                           every route, from /api/openapi.json
       kvad api [METHOD] PATH [BODY]         call one, e.g.
                                             kvad api /api/health
                                             kvad api post /api/keys '{\"name\": \"ci\"}'

BODY is JSON, `-` for stdin, or anything else as text. A streaming route prints
its events one JSON object to a line.";

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

pub fn jobs(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let limit = args.limit.unwrap_or(50);
            let list = remote.get(&format!("/api/jobs?limit={limit}"))?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            job_table(out::items(&list));
            Ok(())
        }
        "show" => {
            let id = super::id(needs(words, "a job's id", JOBS))?;
            let job = remote.get(&format!("/api/jobs/{id}"))?;
            if args.json {
                out::json(&job);
                return Ok(());
            }
            describe_job(&job);
            let metrics = out::items(&job["metrics"]);
            if !metrics.is_empty() {
                println!();
                let rows: Vec<Vec<String>> = metrics.iter().map(metric_row).collect();
                out::table(&["STEP", "TRAIN", "VALIDATION", "CHARS/S", "KEPT"], &rows);
            }
            if let Some(last) = out::items(&job["samples"]).last() {
                println!("\nsample at step {}:\n{}", out::s(&last["step"]), out::s(&last["text"]));
            }
            Ok(())
        }
        "watch" => {
            let id = super::id(needs(words, "a job's id", JOBS))?;
            watch(remote, id, args.json)
        }
        "cancel" => {
            let id = super::id(needs(words, "a job's id", JOBS))?;
            let asked = remote.delete(&format!("/api/jobs/{id}"))?;
            match (args.json, asked["asked"] == true) {
                (true, _) => out::json(&asked),
                (false, true) => println!("asked job {id} to stop; it stops at its next step or file"),
                (false, false) => println!("job {id} is not running"),
            }
            Ok(())
        }
        "pictures" => {
            let id = super::id(needs(words, "a job's id", JOBS))?;
            let job = remote.get(&format!("/api/jobs/{id}"))?;
            let drawn = out::items(&job["pictures"]);
            if args.json {
                out::json(&job["pictures"]);
                return Ok(());
            }
            if drawn.is_empty() {
                println!("job {id} drew no pictures; a LoRA run draws them when it is given `--sample`");
                return Ok(());
            }
            let dir = std::path::PathBuf::from(args.out.as_deref().unwrap_or("."));
            std::fs::create_dir_all(&dir).map_err(|e| format!("could not make {}: {e}", dir.display()))?;
            for p in drawn {
                let file = out::s(&p["file"]);
                // The name is the server's, and it becomes a path here.
                if file.contains(['/', '\\']) || file.starts_with('.') {
                    return Err(format!("the server named a picture `{file}`, which is not a file's name").into());
                }
                let to = dir.join(&file);
                let file = segment(&file);
                remote.download(&format!("/api/jobs/{id}/pictures/{file}"), &to)?;
            }
            println!("wrote {} pictures to {}", drawn.len(), dir.display());
            Ok(())
        }
        other => unknown("jobs", other, JOBS),
    }
}

fn job_table(jobs: &[Value]) {
    if jobs.is_empty() {
        println!("no jobs yet");
        return;
    }
    let rows: Vec<Vec<String>> = jobs
        .iter()
        .map(|j| {
            vec![
                out::s(&j["id"]),
                out::s(&j["kind"]),
                out::s(&j["state"]),
                out::s(&j["created_at"]),
                out::cut(&out::s(&j["label"]), 60),
            ]
        })
        .collect();
    out::table(&["ID", "KIND", "STATE", "STARTED", "WHAT"], &rows);
}

fn describe_job(job: &Value) {
    println!("job {} — {} ({})", out::s(&job["id"]), out::s(&job["label"]), out::s(&job["kind"]));
    println!("  {} · created {}", out::s(&job["state"]), out::s(&job["created_at"]));
    if let Some(ended) = job["ended_at"].as_str() {
        println!("  ended {ended}");
    }
    if let Some(error) = job["error"].as_str() {
        println!("  error: {error}");
    }
}

fn metric_row(m: &Value) -> Vec<String> {
    vec![
        out::s(&m["step"]),
        format!("{:.3}", m["train_loss"].as_f64().unwrap_or(f64::NAN)),
        format!("{:.3}", m["val_loss"].as_f64().unwrap_or(f64::NAN)),
        format!("{:.0}", m["chars_per_sec"].as_f64().unwrap_or(0.0)),
        if m["saved"] == true { "saved".into() } else { String::new() },
    ]
}

/// Follow a job a command just started, until it ends.
///
/// Said first, because it is the thing people get wrong: the job is the
/// server's, and Ctrl-C here stops the watching, not the work.
pub fn follow(remote: &Remote, job: &Value, args: &Args) -> Res<()> {
    let id = job["id"].as_i64().ok_or("the server started a job and did not say which")?;
    if !args.json {
        eprintln!(
            "job {id}: {}\n  Ctrl-C stops watching, not the job. `kvad jobs cancel {id}` stops the job.",
            out::s(&job["label"])
        );
    }
    watch(remote, id, args.json)
}

/// Print a job's updates as they arrive, and return how it ended.
///
/// The server sends the history and then what is live, and an update landing
/// between the two can arrive twice; each is keyed, and a repeat is dropped
/// here rather than printed as a second checkpoint at the same step.
pub fn watch(remote: &Remote, id: i64, raw: bool) -> Res<()> {
    let mut progress = out::Progress::new();
    let mut seen: HashSet<String> = HashSet::new();
    // A download reports each file done twice, once from each of the two
    // places that notice. A line identical to the one before it says
    // nothing new.
    let mut said = String::new();
    for event in remote.stream("get", &format!("/api/jobs/{id}/events"), None)? {
        let event = event?;
        if raw {
            println!("{}", event.to_json());
        }
        let u = event.json()?;
        let kind = u["kind"].as_str().unwrap_or("");
        if kind == "ended" {
            progress.done();
            let state = out::s(&u["state"]);
            return match (state.as_str(), u["error"].as_str()) {
                ("done", _) => {
                    if !raw {
                        println!("job {id} done");
                    }
                    Ok(())
                }
                (_, Some(error)) => Err(format!("job {id} {state}: {error}").into()),
                _ => Err(format!("job {id} {state}").into()),
            };
        }
        if raw {
            continue;
        }
        let key = match kind {
            "case" => format!("case/{}/{}", u["variant"], u["idx"]),
            "scored" => format!("scored/{}", u["variant"]),
            "timing" => format!("timing/{}/{}", u["variant"], u["round"]),
            _ => String::new(),
        };
        if !key.is_empty() && !seen.insert(key) {
            continue;
        }
        match kind {
            "status" => {
                let message = out::s(&u["message"]);
                if message != std::mem::replace(&mut said, message.clone()) {
                    progress.done();
                    println!("  {message}");
                }
            }
            "download" => progress.show(format!(
                "  {}  {} of {}",
                out::s(&u["file"]),
                out::bytes(&u["bytes"]),
                out::bytes(&u["total"])
            )),
            "progress" => progress.show(format!("  {} of {}", out::s(&u["done"]), out::s(&u["total"]))),
            "pace" => {
                progress.done();
                println!(
                    "  {:.0} characters a second here; about {} to go",
                    u["chars_per_sec"].as_f64().unwrap_or(0.0),
                    duration(u["remaining_secs"].as_f64().unwrap_or(0.0))
                );
            }
            // A training run says each checkpoint twice: as a `metric` for the
            // chart and in its own words as a `status`, which is what `kvad
            // train` prints in this process. The words are printed; the chart
            // is `kvad jobs show`.
            "metric" | "sample" => {}
            "picture" => {
                progress.done();
                println!("  drew {}", out::s(&u["file"]));
            }
            "case" => {
                progress.done();
                println!(
                    "  {}  {}  #{}  {}  → {}",
                    if u["passed"] == true { "pass" } else { "FAIL" },
                    out::s(&u["variant"]),
                    out::s(&u["idx"]),
                    out::cut(&out::s(&u["prompt"]), 40),
                    out::cut(&out::s(&u["got"]), 40),
                );
            }
            "scored" => {
                progress.done();
                println!(
                    "  {}  perplexity {:.2} · {:.3} bits a token · {} tokens in {:.1}s",
                    out::s(&u["variant"]),
                    u["perplexity"].as_f64().unwrap_or(f64::NAN),
                    u["bits_per_token"].as_f64().unwrap_or(f64::NAN),
                    out::s(&u["tokens"]),
                    u["took_secs"].as_f64().unwrap_or(0.0),
                );
            }
            "timing" => {
                progress.done();
                println!(
                    "  round {}  {}  {:.1} tok/s decode · {:.0} tok/s prefill · first token {:.0} ms",
                    out::s(&u["round"]),
                    out::s(&u["variant"]),
                    u["decode_per_sec"].as_f64().unwrap_or(0.0),
                    u["prefill_per_sec"].as_f64().unwrap_or(0.0),
                    u["ttft_millis"].as_f64().unwrap_or(0.0),
                );
            }
            _ => {}
        }
    }
    progress.done();
    Err(format!("the server stopped sending before job {id} ended; `kvad jobs watch {id}` picks it up again").into())
}

fn duration(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    match s {
        s if s < 90 => format!("{s}s"),
        s if s < 5400 => format!("{} minutes", s / 60),
        s => format!("{:.1} hours", s as f64 / 3600.0),
    }
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

/// Whether a dataset is a folder of pictures, not a text.
fn pictures(dataset: &Value) -> bool {
    dataset["kind"] == "pictures"
}

/// A dataset of pictures in a line.
fn pictures_line(d: &Value) -> String {
    let bare = match d["uncaptioned"].as_u64().unwrap_or(0) {
        0 => String::new(),
        n => format!(", {n} with no caption"),
    };
    format!("dataset {} `{}`: {} pictures{bare}, {}", out::s(&d["id"]), out::s(&d["name"]), out::s(&d["pictures"]), out::bytes(&d["bytes"]))
}

/// A file's name as one segment of a path. The server takes letters, digits,
/// spaces and `-`, `_` or `.` in one, so a space is all there is to escape.
fn segment(file: &str) -> String {
    file.replace(' ', "%20")
}

/// What a file is sent as: its own name, with anything the server would
/// refuse in one as `_`, which is what the web UI's upload does too.
fn sent_as(path: &std::path::Path) -> String {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let name: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ') { c } else { '_' }).collect();
    match name.strip_prefix('.') {
        Some(rest) => format!("_{rest}"),
        None => name,
    }
}

/// What a folder's dataset is called when nothing says: the folder's name,
/// in the characters a dataset's name may have.
fn folder_name(dir: &std::path::Path) -> String {
    let name = dir.canonicalize().ok().and_then(|d| d.file_name().map(|n| n.to_string_lossy().into_owned())).unwrap_or_default();
    let name: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '-' }).collect();
    match name.trim_matches(['-', '.']) {
        "" => "pictures".into(),
        n => n.to_string(),
    }
}

/// The pictures and captions straight inside `dir`, by name.
fn picture_files(dir: &std::path::Path) -> Res<Vec<std::path::PathBuf>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| format!("could not read {}: {e}", dir.display()))? {
        let path = entry?.path();
        let ending = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
        let hidden = path.file_name().is_some_and(|n| n.to_string_lossy().starts_with('.'));
        if path.is_file() && !hidden && matches!(ending.as_str(), "jpg" | "jpeg" | "png" | "webp" | "bmp" | "txt") {
            found.push(path);
        }
    }
    found.sort();
    Ok(found)
}

/// Send files into a dataset of pictures, one request each, and fail at the
/// end with every refusal if any was refused: the rest have arrived.
fn put_files(remote: &Remote, id: i64, files: &[std::path::PathBuf], quiet: bool) -> Res<()> {
    let mut progress = out::Progress::new();
    let mut refused = Vec::new();
    for (i, path) in files.iter().enumerate() {
        let name = sent_as(path);
        if !quiet {
            progress.show(format!("  {} of {}  {name}", i + 1, files.len()));
        }
        let sent = std::fs::read(path)
            .map_err(|e| format!("could not read {}: {e}", path.display()).into())
            .and_then(|bytes| {
                let file = segment(&name);
                remote.call("put", &format!("/api/datasets/{id}/files/{file}"), Some(Body::Bytes(bytes)))
            });
        if let Err(why) = sent {
            refused.push(format!("  {}: {why}", path.display()));
        }
    }
    progress.done();
    match refused.len() {
        0 => Ok(()),
        n => Err(format!("{n} of {} files did not go in:\n{}", files.len(), refused.join("\n")).into()),
    }
}

/// Make the dataset of pictures called `name`, or find it, and send it the
/// pictures and captions in `dir`. The dataset as it then is comes back.
pub fn add_pictures(remote: &Remote, dir: &std::path::Path, name: &str, quiet: bool) -> Res<Value> {
    let files = picture_files(dir)?;
    if files.is_empty() {
        return Err(format!("{} holds no pictures: files ending in jpg, jpeg, png, webp or bmp, each with its caption in a .txt of the same name", dir.display()).into());
    }
    let made = remote.call("post", &format!("/api/datasets/pictures?name={}", super::enc(name)), None)?;
    let id = made["id"].as_i64().ok_or("the server made a dataset and did not say which")?;
    if !quiet {
        eprintln!("sending {} files from {} to dataset `{name}`", files.len(), dir.display());
    }
    put_files(remote, id, &files, quiet)?;
    Ok(remote.get(&format!("/api/datasets/{id}/pictures"))?["dataset"].take())
}

// ---------------------------------------------------------------------------
// A LoRA for an image model
// ---------------------------------------------------------------------------

pub fn tune(remote: &Remote, args: &Args) -> Res<()> {
    if args.words.first().map(String::as_str) == Some("options") {
        return tune_options(remote, args);
    }
    if let Some(word) = args.words.first() {
        unknown("tune", word, TUNE);
    }
    let Some(name) = &args.name else {
        eprintln!("a LoRA needs a name.\n\n{TUNE}");
        std::process::exit(2);
    };
    let size = match args.size.as_deref().map(str::parse::<usize>) {
        None => None,
        Some(Ok(n)) => Some(n),
        Some(Err(_)) => return Err(format!("--size is pixels a side, 512, 768 or 1024, not `{}`", args.size.as_deref().unwrap_or("")).into()),
    };
    let dataset = match (&args.dataset, &args.data) {
        (Some(d), _) => dataset_id(remote, d)?,
        (None, Some(dir)) => {
            let dir = std::path::Path::new(dir);
            if !dir.is_dir() {
                return Err(format!("{} is not a folder; --data is a folder of pictures and their captions", dir.display()).into());
            }
            let made = add_pictures(remote, dir, &folder_name(dir), args.json)?;
            made["id"].as_i64().ok_or("the server made a dataset and did not say which")?
        }
        (None, None) => {
            eprintln!("train it on what?\n\n{TUNE}");
            std::process::exit(2);
        }
    };

    let mut body = json!({ "dataset": dataset, "name": name });
    for (key, value) in [
        ("model", args.model.clone().map(Value::from)),
        ("size", size.map(Value::from)),
        ("rank", args.rank.map(Value::from)),
        ("steps", args.steps.map(Value::from)),
        // Through its text, so that 1e-4 arrives as 0.0001 and not as the
        // nearest f32 to it written out as an f64.
        ("lr", args.lr.and_then(|lr| lr.to_string().parse::<f64>().ok()).map(Value::from)),
        ("eval_every", args.eval_every.map(Value::from)),
        ("seed", args.seed_given.then_some(args.seed).map(Value::from)),
        ("caption", args.caption.clone().map(Value::from)),
        ("sample_size", args.sample_size.map(Value::from)),
        ("sample_steps", args.sample_steps.map(Value::from)),
    ] {
        if let Some(v) = value {
            body[key] = v;
        }
    }
    if !args.samples.is_empty() {
        body["samples"] = json!(args.samples);
    }
    let job = remote.post("/api/tune", &body)?;
    let id = job["id"].as_i64().unwrap_or(0);
    follow(remote, &job, args)?;
    if args.json {
        return Ok(());
    }
    let done = remote.get(&format!("/api/jobs/{id}"))?;
    let r = &done["result"];
    match (r["lora"].is_string(), r["improved"] == true) {
        (false, _) => println!("\nno LoRA was written: the run ended before it measured one"),
        (true, improved) => {
            println!(
                "\n{name}: validation {:.4} at step {}, {:.4} without it{}",
                r["best_val"].as_f64().unwrap_or(f64::NAN),
                out::s(&r["best_step"]),
                r["base_val"].as_f64().unwrap_or(f64::NAN),
                if improved { "" } else { ". It did not beat the model without it" },
            );
            println!("draw with it:  kvad images make \"...\" --lora {name}");
        }
    }
    if !args.samples.is_empty() {
        println!("its samples:   kvad jobs pictures {id} --out DIR");
    }
    Ok(())
}

/// `kvad tune options`: what a LoRA run can be asked for there.
fn tune_options(remote: &Remote, args: &Args) -> Res<()> {
    let o = remote.get("/api/tune/options")?;
    if args.json {
        out::json(&o);
        return Ok(());
    }
    if let Some(why) = o["unavailable"].as_str() {
        println!("no LoRA can be trained there: {why}\n");
    }
    let models = out::items(&o["models"]);
    match models.is_empty() {
        true => println!("models: none on that machine yet. kvad pull {}", out::s(&o["base"])),
        false => println!("models: {}", models.iter().map(out::s).collect::<Vec<_>>().join(", ")),
    }
    let gb = |v: &Value| v.as_f64().unwrap_or(0.0) / 1e9;
    println!("\nmemory set aside for a run, of {:.1} GB nothing holds:", gb(&o["left"]));
    let rows: Vec<Vec<String>> = out::items(&o["sides"])
        .iter()
        .map(|s| vec![format!("{}²", out::s(&s["side"])), format!("{:.1} GB", gb(&s["takes"])), format!("{:.1} GB", gb(&s["takes_sampling"]))])
        .collect();
    out::table(&["SIZE", "RUN", "WITH SAMPLES"], &rows);
    let d = &o["defaults"];
    println!(
        "\ndefaults: --size {} --rank {} --steps {} --lr {} --eval-every {} --seed {} --sample-size {} --sample-steps {}",
        out::s(&d["size"]),
        out::s(&d["rank"]),
        out::s(&d["steps"]),
        out::s(&d["lr"]),
        out::s(&d["eval_every"]),
        out::s(&d["seed"]),
        out::s(&d["sample_size"]),
        out::s(&d["sample_steps"]),
    );
    println!("at most {} --sample prompts a run", out::s(&o["max_samples"]));
    if o["training"] == true {
        println!("a training run is going there now; a second waits for it or is refused");
    }
    Ok(())
}

/// A dataset by id, or by name.
pub fn dataset_id(remote: &Remote, word: &str) -> Res<i64> {
    if let Ok(id) = word.parse() {
        return Ok(id);
    }
    let list = remote.get("/api/datasets")?;
    let all = out::items(&list);
    all.iter()
        .find(|d| d["name"] == word)
        .and_then(|d| d["id"].as_i64())
        .ok_or_else(|| {
            let names: Vec<String> = all.iter().map(|d| out::s(&d["name"])).collect();
            match names.is_empty() {
                true => format!("there is no dataset `{word}`, and no datasets at all").into(),
                false => format!("there is no dataset `{word}`; there are {}", names.join(", ")).into(),
            }
        })
}

pub fn datasets(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/datasets")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let all = out::items(&list);
            if all.is_empty() {
                println!("no datasets yet. add one:  kvad datasets add FILE");
                return Ok(());
            }
            let rows: Vec<Vec<String>> = all
                .iter()
                .map(|d| {
                    vec![
                        out::s(&d["id"]),
                        out::s(&d["name"]),
                        out::bytes(&d["bytes"]),
                        match pictures(d) {
                            true => format!("{} pictures", out::s(&d["pictures"])),
                            false => out::s(&d["characters"]),
                        },
                        match (pictures(d), d["uncaptioned"].as_u64().unwrap_or(0)) {
                            (true, 0) => String::new(),
                            (true, n) => format!("{n} with no caption"),
                            (false, _) => out::s(&d["distinct"]),
                        },
                        match d["present"] == false {
                            true => "its file is gone".into(),
                            false => out::s(&d["source"]),
                        },
                    ]
                })
                .collect();
            out::table(&["ID", "NAME", "SIZE", "CHARACTERS", "DISTINCT", "FROM"], &rows);
            Ok(())
        }
        "add" if std::path::Path::new(needs(words, "a file, or a folder of pictures", DATASETS)).is_dir() => {
            let dir = std::path::Path::new(&words[0]);
            let name = args.name.clone().unwrap_or_else(|| folder_name(dir));
            let made = add_pictures(remote, dir, &name, args.json)?;
            match args.json {
                true => out::json(&made),
                false => println!("{}", pictures_line(&made)),
            }
            Ok(())
        }
        "put" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            if words.len() < 2 {
                eprintln!("put which files into it?  kvad datasets put {id} FILE...");
                std::process::exit(2);
            }
            let files: Vec<std::path::PathBuf> = words[1..].iter().map(std::path::PathBuf::from).collect();
            put_files(remote, id, &files, args.json)?;
            let held = remote.get(&format!("/api/datasets/{id}/pictures"))?;
            match args.json {
                true => out::json(&held["dataset"]),
                false => println!("{}", pictures_line(&held["dataset"])),
            }
            Ok(())
        }
        "get" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            let Some(file) = words.get(1) else {
                eprintln!("get which file from it?  kvad datasets get {id} FILE");
                std::process::exit(2);
            };
            let to = args.out.clone().unwrap_or_else(|| file.clone());
            let file = segment(file);
            let bytes = remote.download(&format!("/api/datasets/{id}/files/{file}"), std::path::Path::new(&to))?;
            println!("wrote {to} ({})", kvad::hub::human_bytes(bytes));
            Ok(())
        }
        "rm" if words.len() > 1 => {
            let id = dataset_id(remote, &words[0])?;
            let files = &words[1..];
            if !out::confirm(&format!("take {} out of dataset {id}?", files.join(", ")), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            for name in files {
                let file = segment(name);
                let gone = remote.delete(&format!("/api/datasets/{id}/files/{file}"))?;
                match args.json {
                    true => out::json(&gone),
                    false => println!("took out {name}"),
                }
            }
            Ok(())
        }
        "add" => {
            let file = needs(words, "a file", DATASETS);
            let text = std::fs::read_to_string(file).map_err(|e| format!("could not read {file}: {e}"))?;
            let name = args.name.clone().unwrap_or_else(|| {
                std::path::Path::new(file).file_stem().map_or("corpus".into(), |s| s.to_string_lossy().into_owned())
            });
            let made = remote.call("post", &format!("/api/datasets?name={}", super::enc(&name)), Some(Body::Text(text)))?;
            match args.json {
                true => out::json(&made),
                false => println!(
                    "dataset {} `{}`: {}, {} characters",
                    out::s(&made["id"]),
                    out::s(&made["name"]),
                    out::bytes(&made["bytes"]),
                    out::s(&made["characters"])
                ),
            }
            Ok(())
        }
        "crawl" => {
            let url = needs(words, "an address to start from", DATASETS);
            let Some(name) = &args.name else {
                eprintln!("a dataset needs a name: kvad datasets crawl {url} --name NAME");
                std::process::exit(2);
            };
            let mut body = json!({ "url": url, "name": name, "same_host": args.same_host });
            if let Some(n) = args.pages {
                body["max_pages"] = json!(n);
            }
            if let Some(mb) = args.megabytes {
                body["max_bytes"] = json!((mb * 1024.0 * 1024.0) as u64);
            }
            if let Some(ms) = args.pause {
                body["delay_ms"] = json!(ms);
            }
            if let Some(n) = args.drop_rare {
                body["drop_rare"] = json!(n);
            }
            let job = remote.post("/api/datasets/crawl", &body)?;
            follow(remote, &job, args)
        }
        "show" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            // A folder of pictures is shown by its own route; the text
            // one refuses it.
            let listed = remote.get("/api/datasets")?;
            if out::items(&listed).iter().any(|d| d["id"] == id && pictures(d)) {
                let held = remote.get(&format!("/api/datasets/{id}/pictures"))?;
                if args.json {
                    out::json(&held);
                    return Ok(());
                }
                let d = &held["dataset"];
                println!("{}, added {}\n", pictures_line(d), out::s(&d["created_at"]));
                let rows: Vec<Vec<String>> = out::items(&held["pictures"])
                    .iter()
                    .map(|p| vec![out::s(&p["file"]), out::bytes(&p["bytes"]), p["caption"].as_str().map_or("(no caption)".into(), |c| out::cut(c, 80))])
                    .collect();
                out::table(&["FILE", "SIZE", "CAPTION"], &rows);
                return Ok(());
            }
            let d = remote.get(&format!("/api/datasets/{id}"))?;
            if args.json {
                out::json(&d);
                return Ok(());
            }
            println!("dataset {} `{}`", out::s(&d["id"]), out::s(&d["name"]));
            println!(
                "  {} · {} characters, {} distinct · added {}",
                out::bytes(&d["bytes"]),
                out::s(&d["characters"]),
                out::s(&d["distinct"]),
                out::s(&d["created_at"])
            );
            if let Some(source) = d["source"].as_str() {
                println!("  from {source}");
            }
            println!("\n{}", out::s(&d["preview"]));
            Ok(())
        }
        "check" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            let Some(model) = &args.model else {
                eprintln!("check against which model?  kvad datasets check {id} --model NAME");
                std::process::exit(2);
            };
            let answer = remote.get(&format!("/api/datasets/{id}/check?model={}", super::enc(model)))?;
            if args.json {
                out::json(&answer);
                return Ok(());
            }
            match (answer["continuable"] == true, answer["unseen_count"].as_u64().unwrap_or(0)) {
                (false, _) => println!("{model} is not a model trained here with a character tokeniser, so it cannot be trained further"),
                (true, 0) => println!("every character in `{}` is one {model} has a token for", out::s(&answer["dataset"])),
                (true, n) => println!(
                    "{model} has no token for {n} of the characters in `{}`: {:?}",
                    out::s(&answer["dataset"]),
                    out::s(&answer["unseen"])
                ),
            }
            Ok(())
        }
        "search" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            let question = words[1..].join(" ");
            if question.is_empty() {
                eprintln!("search it for what?  kvad datasets search {id} QUESTION");
                std::process::exit(2);
            }
            let k = args.k.unwrap_or(5);
            let found = remote.get(&format!("/api/datasets/{id}/search?q={}&k={k}", super::enc(&question)))?;
            if args.json {
                out::json(&found);
                return Ok(());
            }
            let passages = out::items(&found["passages"]);
            if passages.is_empty() {
                println!("nothing in it bears on that");
            }
            for p in passages {
                let because: Vec<String> = out::items(&p["because"])
                    .iter()
                    .map(|w| format!("{} {:.2}", out::s(&w[0]), w[1].as_f64().unwrap_or(0.0)))
                    .collect();
                println!("{:.2}  {}", p["score"].as_f64().unwrap_or(0.0), out::s(&p["heading"]));
                if let Some(source) = p["source"].as_str() {
                    println!("      {source}");
                }
                println!("      because: {}", because.join(", "));
                println!("      {}\n", out::cut(&out::s(&p["text"]), 300));
            }
            println!("{} chunks, {} words in its vocabulary", out::s(&found["chunks"]), out::s(&found["vocabulary"]));
            Ok(())
        }
        "rm" => {
            let id = dataset_id(remote, needs(words, "a dataset", DATASETS))?;
            if !out::confirm(&format!("delete dataset {id} and its file?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/api/datasets/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted dataset {id}"),
            }
            Ok(())
        }
        other => unknown("datasets", other, DATASETS),
    }
}

// ---------------------------------------------------------------------------
// Evals and benchmarks
// ---------------------------------------------------------------------------

/// `MODEL[@BACKEND]` words as the variants a comparison runs.
///
/// A backend the word does not name is `--backend`, else the one the server
/// would load that model on by default — asked of the server rather than
/// guessed here, because the default is a fact about the server's build.
fn variants(remote: &Remote, words: &[String], args: &Args) -> Res<Vec<Value>> {
    if words.is_empty() {
        return Err("name at least one model to run it on, as MODEL or MODEL@BACKEND".into());
    }
    let mut fallback = args.backend.clone();
    let mut out = Vec::new();
    for word in words {
        let (model, backend) = match word.rsplit_once('@') {
            Some((m, b)) => (m.to_string(), b.to_string()),
            None => {
                if fallback.is_none() {
                    fallback = remote.get("/api/models")?["backend"].as_str().map(str::to_string);
                }
                let b = fallback.clone().ok_or("the server did not say which backend it loads on by default; give one with --backend")?;
                (word.clone(), b)
            }
        };
        out.push(json!({ "model": model, "backend": backend }));
    }
    Ok(out)
}

/// A suite by id, or by name.
fn suite_id(remote: &Remote, word: &str) -> Res<i64> {
    if let Ok(id) = word.parse() {
        return Ok(id);
    }
    let list = remote.get("/api/evals/suites")?;
    out::items(&list)
        .iter()
        .find(|s| s["name"] == word)
        .and_then(|s| s["id"].as_i64())
        .ok_or_else(|| format!("there is no suite `{word}`; `kvad evals suites` lists them").into())
}

/// A suite from a JSON file, with `--name` over whatever name it has.
fn suite_file(file: &str, args: &Args) -> Res<Value> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("could not read {file}: {e}"))?;
    let mut suite: Value = serde_json::from_str(&text).map_err(|e| format!("{file} is not JSON: {e}"))?;
    if let Some(name) = &args.name {
        suite["name"] = json!(name);
    }
    Ok(suite)
}

pub fn evals(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "runs");
    match sub {
        "runs" | "ls" => {
            let runs = remote.get("/api/evals/runs")?;
            match args.json {
                true => out::json(&runs),
                false => job_table(out::items(&runs)),
            }
            Ok(())
        }
        "show" => {
            let id = super::id(needs(words, "a run's id", EVALS))?;
            let run = remote.get(&format!("/api/evals/runs/{id}"))?;
            if args.json {
                out::json(&run);
                return Ok(());
            }
            describe_job(&run);
            let cases = out::items(&run["cases"]);
            if !cases.is_empty() {
                println!();
                let rows: Vec<Vec<String>> = cases
                    .iter()
                    .map(|c| {
                        vec![
                            if c["passed"] == true { "pass".into() } else { "FAIL".into() },
                            out::s(&c["variant"]),
                            out::s(&c["idx"]),
                            out::cut(&out::s(&c["prompt"]), 30),
                            out::cut(&out::s(&c["expect"]), 20),
                            out::cut(&out::s(&c["got"]), 40),
                        ]
                    })
                    .collect();
                out::table(&["", "MODEL", "#", "PROMPT", "EXPECTED", "GOT"], &rows);
                let passed = cases.iter().filter(|c| c["passed"] == true).count();
                println!("\n{passed} of {} passed", cases.len());
            }
            let scores = out::items(&run["scores"]);
            if !scores.is_empty() {
                println!();
                let rows: Vec<Vec<String>> = scores
                    .iter()
                    .map(|s| {
                        vec![
                            out::s(&s["variant"]),
                            format!("{:.2}", s["perplexity"].as_f64().unwrap_or(f64::NAN)),
                            format!("{:.3}", s["bits_per_token"].as_f64().unwrap_or(f64::NAN)),
                            out::s(&s["tokens"]),
                        ]
                    })
                    .collect();
                out::table(&["MODEL", "PERPLEXITY", "BITS/TOKEN", "TOKENS"], &rows);
            }
            Ok(())
        }
        "suites" => {
            let list = remote.get("/api/evals/suites")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|s| {
                    vec![
                        out::s(&s["id"]),
                        out::s(&s["name"]),
                        out::items(&s["cases"]).len().to_string(),
                        out::s(&s["updated_at"]),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no suites yet. add one:  kvad evals add FILE"),
                false => out::table(&["ID", "NAME", "CASES", "CHANGED"], &rows),
            }
            Ok(())
        }
        "add" => {
            let suite = suite_file(needs(words, "a suite file", EVALS), args)?;
            let made = remote.post("/api/evals/suites", &suite)?;
            match args.json {
                true => out::json(&made),
                false => println!("suite {} `{}`", out::s(&made["id"]), out::s(&made["name"])),
            }
            Ok(())
        }
        "edit" => {
            let id = suite_id(remote, needs(words, "a suite", EVALS))?;
            let Some(file) = words.get(1) else {
                eprintln!("edit it with what?  kvad evals edit {id} FILE");
                std::process::exit(2);
            };
            let changed = remote.patch(&format!("/api/evals/suites/{id}"), &suite_file(file, args)?)?;
            match args.json {
                true => out::json(&changed),
                false => println!("suite {id} changed"),
            }
            Ok(())
        }
        "rm" => {
            let id = suite_id(remote, needs(words, "a suite", EVALS))?;
            if !out::confirm(&format!("delete suite {id}?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/api/evals/suites/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted suite {id}"),
            }
            Ok(())
        }
        "run" => {
            let suite = suite_id(remote, needs(words, "a suite", EVALS))?;
            let mut body = json!({ "suite": suite, "variants": variants(remote, &words[1..], args)?, "seed": args.seed });
            body["max_tokens"] = json!(args.max_tokens);
            let job = remote.post("/api/evals/run", &body)?;
            follow(remote, &job, args)
        }
        "perplexity" => {
            let dataset = dataset_id(remote, needs(words, "a dataset", EVALS))?;
            let mut body = json!({ "dataset": dataset, "variants": variants(remote, &words[1..], args)? });
            if let Some(w) = args.window {
                body["window"] = json!(w);
            }
            let job = remote.post("/api/evals/perplexity", &body)?;
            follow(remote, &job, args)
        }
        other => unknown("evals", other, EVALS),
    }
}

pub fn bench(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "runs");
    match sub {
        "runs" | "ls" => {
            let runs = remote.get("/api/bench/runs")?;
            match args.json {
                true => out::json(&runs),
                false => job_table(out::items(&runs)),
            }
            Ok(())
        }
        "show" => {
            let id = super::id(needs(words, "a run's id", BENCH))?;
            let run = remote.get(&format!("/api/bench/runs/{id}"))?;
            if args.json {
                out::json(&run);
                return Ok(());
            }
            describe_job(&run);
            let rows: Vec<Vec<String>> = out::items(&run["summary"])
                .iter()
                .map(|s| {
                    let n = |k: &str| s[k].as_f64().map_or("-".into(), |v| format!("{v:.1}"));
                    vec![
                        out::s(&s["variant"]),
                        out::s(&s["runs"]),
                        format!("{} ({}–{})", n("decode_median"), n("decode_low"), n("decode_high")),
                        format!("{} ({}–{})", n("ttft_median"), n("ttft_low"), n("ttft_high")),
                    ]
                })
                .collect();
            if !rows.is_empty() {
                println!();
                out::table(&["MODEL", "RUNS", "DECODE TOK/S, MEDIAN (RANGE)", "FIRST TOKEN MS"], &rows);
            }
            Ok(())
        }
        "run" => {
            let mut body = json!({ "variants": variants(remote, words, args)?, "seed": args.seed });
            for (key, value) in [
                ("prompt", args.prompt.clone().map(Value::from)),
                ("rounds", args.rounds.map(Value::from)),
                ("tokens", args.tokens.map(Value::from)),
            ] {
                if let Some(v) = value {
                    body[key] = v;
                }
            }
            let job = remote.post("/api/bench/run", &body)?;
            follow(remote, &job, args)
        }
        other => unknown("bench", other, BENCH),
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

pub fn metrics(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, _) = split(args, "overview");
    match sub {
        "overview" => {
            let m = remote.get("/api/metrics")?;
            if args.json {
                out::json(&m);
                return Ok(());
            }
            let residents: Vec<String> = out::items(&m["residents"]).iter().map(|r| out::s(&r["id"])).collect();
            println!("up         {}s", out::s(&m["uptime_secs"]));
            println!("in memory  {}", if residents.is_empty() { "nothing".into() } else { residents.join(", ") });
            println!(
                "memory     {} of {} left for models; the process holds {}",
                out::bytes(&m["memory"]["left"]),
                out::bytes(&m["memory"]["total"]),
                out::bytes(&m["resident_bytes"])
            );
            let kv = &m["kv"];
            if kv.is_object() {
                println!(
                    "KV cache   {} tokens, {} of {} at full context",
                    out::s(&kv["cached_tokens"]),
                    out::bytes(&kv["cached_bytes"]),
                    out::bytes(&kv["max_bytes"])
                );
            }
            let median = |k: &str| {
                let mut v: Vec<f64> = out::items(&m[k]).iter().filter_map(Value::as_f64).collect();
                v.sort_by(f64::total_cmp);
                v.get(v.len() / 2).copied()
            };
            if let (Some(decode), Some(ttft)) = (median("decode_per_sec"), median("ttft_millis")) {
                println!("replies    {decode:.1} tok/s decode, first token in {ttft:.0} ms (medians)");
            }
            let l = &m["latency"];
            println!(
                "requests   {} recent, {} failed · median {:.0} ms, p95 {:.0} ms",
                out::s(&m["requests"]),
                out::s(&m["errors"]),
                l["median"].as_f64().unwrap_or(0.0),
                l["p95"].as_f64().unwrap_or(0.0)
            );
            println!("queue      {}", out::s(&m["queue_depth"]));
            let running = out::items(&m["running"]);
            if !running.is_empty() {
                println!();
                job_table(running);
            }
            Ok(())
        }
        "requests" => {
            let limit = args.limit.unwrap_or(20);
            let r = remote.get(&format!("/api/metrics/requests?limit={limit}"))?;
            if args.json {
                out::json(&r);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&r["by_route"])
                .iter()
                .map(|b| {
                    vec![
                        out::s(&b["method"]),
                        out::s(&b["path"]),
                        out::s(&b["count"]),
                        format!("{:.0}", b["median"].as_f64().unwrap_or(0.0)),
                        format!("{:.0}", b["p95"].as_f64().unwrap_or(0.0)),
                        out::s(&b["errors"]),
                    ]
                })
                .collect();
            out::table(&["", "ROUTE", "COUNT", "MEDIAN MS", "P95 MS", "ERRORS"], &rows);
            println!();
            let rows: Vec<Vec<String>> = out::items(&r["recent"])
                .iter()
                .map(|q| {
                    vec![
                        out::s(&q["method"]),
                        out::s(&q["path"]),
                        out::s(&q["status"]),
                        format!("{:.0}", q["millis"].as_f64().unwrap_or(0.0)),
                    ]
                })
                .collect();
            out::table(&["", "RECENT", "STATUS", "MS"], &rows);
            Ok(())
        }
        "log" => {
            let limit = args.limit.unwrap_or(200);
            let lines = remote.get(&format!("/api/metrics/log?limit={limit}"))?;
            match args.json {
                true => out::json(&lines),
                false => {
                    for line in out::items(&lines) {
                        println!("{}", out::s(line).trim_end());
                    }
                }
            }
            Ok(())
        }
        other => unknown("metrics", other, METRICS),
    }
}

// ---------------------------------------------------------------------------
// Conversations
// ---------------------------------------------------------------------------

pub fn conversations(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/conversations")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|c| {
                    vec![
                        out::s(&c["id"]),
                        out::s(&c["messages"]),
                        out::s(&c["updated_at"]),
                        out::s(&c["model"]),
                        out::cut(&out::s(&c["title"]), 60),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no conversations yet. keep one:  kvad chat --save"),
                false => out::table(&["ID", "MESSAGES", "CHANGED", "MODEL", "TITLE"], &rows),
            }
            Ok(())
        }
        "show" => {
            let id = super::id(needs(words, "a conversation's id", CONVERSATIONS))?;
            let c = remote.get(&format!("/api/conversations/{id}"))?;
            if args.json {
                out::json(&c);
                return Ok(());
            }
            println!("{} — {}", out::s(&c["title"]), out::s(&c["updated_at"]));
            if let Some(system) = c["system"].as_str() {
                println!("\nsystem: {system}");
            }
            for m in out::items(&c["messages"]) {
                let who = out::s(&m["role"]);
                let text = out::s(&m["content"]);
                match (who.as_str(), out::tty()) {
                    ("user", true) => println!("\n\x1b[1m> {text}\x1b[0m"),
                    ("user", false) => println!("\n> {text}"),
                    _ => println!("\n{text}"),
                }
                let s = &m["stats"];
                if s.is_object() {
                    println!(
                        "  [{} tok, {} cached, {}]",
                        out::s(&s["generated_tokens"]),
                        out::s(&s["cached_tokens"]),
                        out::s(&s["model"])
                    );
                }
            }
            Ok(())
        }
        "edit" => {
            let id = super::id(needs(words, "a conversation's id", CONVERSATIONS))?;
            let mut body = json!({});
            if let Some(t) = &args.title {
                body["title"] = json!(t);
            }
            if let Some(s) = &args.system {
                // An empty system prompt clears it, rather than setting one
                // that says nothing.
                body["system"] = if s.is_empty() { Value::Null } else { json!(s) };
            }
            if body.as_object().is_some_and(|o| o.is_empty()) {
                eprintln!("change what?  kvad conversations edit {id} --title T --system S");
                std::process::exit(2);
            }
            let changed = remote.patch(&format!("/api/conversations/{id}"), &body)?;
            match args.json {
                true => out::json(&changed),
                false => println!("conversation {id} changed"),
            }
            Ok(())
        }
        "rm" => {
            let id = super::id(needs(words, "a conversation's id", CONVERSATIONS))?;
            if !out::confirm(&format!("delete conversation {id} and its messages?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/api/conversations/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted conversation {id}"),
            }
            Ok(())
        }
        other => unknown("conversations", other, CONVERSATIONS),
    }
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

pub fn images(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/images")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|i| {
                    vec![
                        out::s(&i["id"]),
                        out::s(&i["created_at"]),
                        format!("{}×{}", out::s(&i["width"]), out::s(&i["height"])),
                        out::s(&i["seed"]),
                        out::s(&i["model"]),
                        out::cut(&out::s(&i["prompt"]), 50),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no images yet. make one:  kvad images make \"a lighthouse at dusk\""),
                false => out::table(&["ID", "MADE", "SIZE", "SEED", "MODEL", "PROMPT"], &rows),
            }
            Ok(())
        }
        "make" | "edit" => {
            let prompt = match (&args.prompt, words.is_empty()) {
                (Some(p), _) => p.clone(),
                (None, false) => words.join(" "),
                (None, true) => {
                    eprintln!("{sub} what?\n\n{IMAGES}");
                    std::process::exit(2);
                }
            };
            let mut body = json!({ "prompt": prompt, "stream": true, "response_format": "b64_json" });
            // A file as a `data:` URL: the server reads a picture by its
            // bytes, not its name, so the media type is only a label.
            let file = |path: &String| -> Res<Value> {
                let bytes = std::fs::read(path).map_err(|e| format!("could not read {path}: {e}"))?;
                Ok(json!(format!("data:application/octet-stream;base64,{}", client::encode_base64(&bytes))))
            };
            let editing = match (sub, &args.image) {
                ("edit", Some(picture)) => {
                    body["image"] = file(picture)?;
                    if let Some(mask) = &args.mask {
                        body["mask"] = file(mask)?;
                    }
                    if let Some(s) = args.strength {
                        body["strength"] = json!(s);
                    }
                    true
                }
                ("edit", None) => return Err(format!("edit which picture? --image PICTURE\n\n{IMAGES}").into()),
                (_, None) if args.mask.is_none() && args.strength.is_none() => false,
                _ => return Err("--image, --mask and --strength are `kvad images edit`'s; `make` draws from a prompt alone".into()),
            };
            if let Some(m) = &args.model {
                body["model"] = json!(m);
            }
            if let Some(size) = &args.size {
                body["size"] = json!(size);
            }
            if let Some(n) = args.steps {
                body["steps"] = json!(n);
            }
            if let Some(g) = args.guidance {
                body["guidance_scale"] = json!(g);
            }
            if let Some(n) = &args.negative {
                body["negative_prompt"] = json!(n);
            }
            if args.seed_given {
                body["seed"] = json!(args.seed);
            }
            if !args.loras.is_empty() {
                body["loras"] = json!(args.loras.iter().map(|l| kvad::image::Lora::parse(l)).collect::<Vec<_>>());
            }

            let mut progress = out::Progress::new();
            let started = std::time::Instant::now();
            // The same events from either route.
            let events = match editing {
                true => remote.stream("post", "/v1/images/edits", Some(Body::Json(&body)))?,
                false => remote.stream("post", "/v1/images/generations", Some(Body::Json(&body)))?,
            };
            for event in events {
                let event = event?;
                let data = event.json()?;
                match event.name.as_str() {
                    "image_generation.step" => progress.show(format!(
                        "  step {} of {}  {:.0} s",
                        out::s(&data["step"]),
                        out::s(&data["total"]),
                        started.elapsed().as_secs_f64()
                    )),
                    "image_generation.completed" => {
                        progress.done();
                        let k = &data["kvad"];
                        let png = client::decode_base64(data["b64_json"].as_str().unwrap_or(""))?;
                        let path = args.out.clone().unwrap_or_else(|| format!("image-{}.png", out::s(&k["id"])));
                        std::fs::write(&path, &png).map_err(|e| format!("could not write {path}: {e}"))?;
                        match args.json {
                            true => out::json(k),
                            false => eprintln!(
                                "{path}: {}×{}, {}{} steps, guidance {}, seed {} — image {} on the server\n  \
                                 denoise {:.1} s, decode {:.1} s",
                                out::s(&k["width"]),
                                out::s(&k["height"]),
                                // An edit runs the last `strength` of them.
                                k["strength"].as_f64().map(|s| format!("strength {s:.2} of ")).unwrap_or_default(),
                                out::s(&k["steps"]),
                                out::s(&k["guidance"]),
                                out::s(&k["seed"]),
                                out::s(&k["id"]),
                                k["denoise_secs"].as_f64().unwrap_or(0.0),
                                k["decode_secs"].as_f64().unwrap_or(0.0),
                            ),
                        }
                    }
                    "error" => {
                        progress.done();
                        return Err(out::s(&data["error"]).into());
                    }
                    _ => {}
                }
            }
            Ok(())
        }
        "rm" => {
            let id = super::id(needs(words, "an image's id", IMAGES))?;
            if !out::confirm(&format!("delete image {id}?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/api/images/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted image {id}"),
            }
            Ok(())
        }
        other => unknown("images", other, IMAGES),
    }
}

// ---------------------------------------------------------------------------
// Videos
// ---------------------------------------------------------------------------

pub fn videos(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/v1/videos?limit=100")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = list["data"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or_default()
                .iter()
                .map(|v| {
                    vec![
                        out::s(&v["kvad"]["id"]),
                        state(v),
                        out::s(&v["size"]),
                        length(v),
                        out::s(&v["kvad"]["seed"]),
                        out::s(&v["model"]),
                        out::cut(&out::s(&v["prompt"]), 40),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no videos yet. make one:  kvad videos make \"a lighthouse at dusk, waves breaking\""),
                false => out::table(&["ID", "STATUS", "SIZE", "LENGTH", "SEED", "MODEL", "PROMPT"], &rows),
            }
            Ok(())
        }
        "make" => {
            let prompt = match (&args.prompt, words.is_empty()) {
                (Some(p), _) => p.clone(),
                (None, false) => words.join(" "),
                (None, true) => {
                    eprintln!("make what?\n\n{VIDEOS}");
                    std::process::exit(2);
                }
            };
            let mut body = json!({ "prompt": prompt });
            if let Some(m) = &args.model {
                body["model"] = json!(m);
            }
            if let Some(size) = &args.size {
                body["size"] = json!(size);
            }
            if let Some(s) = args.seconds {
                body["seconds"] = json!(s);
            }
            if let Some(n) = args.frames {
                body["frames"] = json!(n);
            }
            if let Some(n) = args.fps {
                body["fps"] = json!(n);
            }
            if args.seed_given {
                body["seed"] = json!(args.seed);
            }
            if args.silent {
                body["audio"] = json!(false);
            }
            // Any of these asks for the guided (dev) pipeline.
            if let Some(n) = args.steps {
                body["steps"] = json!(n);
            }
            if let Some(g) = args.guidance {
                body["guidance_scale"] = json!(g);
            }
            if let Some(n) = &args.negative {
                body["negative_prompt"] = json!(n);
            }
            if let Some(d) = &args.decoder {
                body["decoder"] = json!(d);
            }
            if let Some(p) = &args.pipeline {
                body["pipeline"] = json!(p);
            }
            if !args.loras.is_empty() {
                body["loras"] = json!(args.loras.iter().map(|l| kvad::image::Lora::parse(l)).collect::<Vec<_>>());
            }
            if let Some(path) = &args.image {
                let bytes = std::fs::read(path).map_err(|e| format!("could not read {path}: {e}"))?;
                // The server reads the picture by its bytes, not its name,
                // so the media type in the URL is only a label.
                body["input_reference"] = json!({ "image_url": format!("data:application/octet-stream;base64,{}", client::encode_base64(&bytes)) });
            }
            let video = remote.post("/v1/videos", &body)?;
            let id = out::s(&video["id"]);
            eprintln!(
                "{id}: {}, {} — the server keeps making it if this stops waiting; `kvad videos get {}` fetches it",
                out::s(&video["size"]),
                length(&video),
                out::s(&video["kvad"]["id"]),
            );
            let video = follow_video(remote, &id)?;
            let path = args.out.clone().unwrap_or_else(|| format!("video-{}.mp4", out::s(&video["kvad"]["id"])));
            fetch(remote, &video, &path)?;
            let k = &video["kvad"];
            match args.json {
                true => out::json(&video),
                false => eprintln!(
                    "{path}: {}, {} frames at {} fps, seed {}{}{}\n  \
                     {} {:.1} s, denoise {:.1} s, decode{} {:.1} s",
                    out::s(&video["size"]),
                    out::s(&k["frames"]),
                    out::s(&k["fps"]),
                    out::s(&k["seed"]),
                    if k["pipeline"] == json!("dfr") { ", DFR" } else { "" },
                    if k["audio"] == json!(false) { ", no sound" } else { "" },
                    if k["picture_url"].is_string() { "picture and text" } else { "text" },
                    k["encode_secs"].as_f64().unwrap_or(0.0),
                    k["denoise_secs"].as_f64().unwrap_or(0.0),
                    k["decoder"].as_str().map(|d| format!(" ({d})")).unwrap_or_default(),
                    k["decode_secs"].as_f64().unwrap_or(0.0),
                ),
            }
            Ok(())
        }
        "watch" => {
            let id = super::id(needs(words, "a video's id", VIDEOS))?;
            let video = follow_video(remote, &format!("video_{id}"))?;
            match args.json {
                true => out::json(&video),
                false => println!("video {id} is done: `kvad videos get {id}` fetches it"),
            }
            Ok(())
        }
        "show" => {
            let id = super::id(needs(words, "a video's id", VIDEOS))?;
            let v = remote.get(&format!("/v1/videos/{id}"))?;
            match args.json {
                true => out::json(&v),
                false => {
                    println!("{}  {}", out::s(&v["id"]), state(&v));
                    let (size, seed) = (out::s(&v["size"]), out::s(&v["kvad"]["seed"]));
                    let decoder = v["kvad"]["decoder"].as_str().map(|d| format!(", {d} decoder")).unwrap_or_default();
                    let decoder = match v["kvad"]["pipeline"].as_str() {
                        Some("dfr") => format!(", DFR{decoder}"),
                        _ => decoder,
                    };
                    println!("  {size}, {}, seed {seed}{decoder}, {}", length(&v), out::s(&v["model"]));
                    println!("  {}", out::s(&v["prompt"]));
                }
            }
            Ok(())
        }
        "get" => {
            let id = super::id(needs(words, "a video's id", VIDEOS))?;
            let v = remote.get(&format!("/v1/videos/{id}"))?;
            if v["status"] != "completed" {
                return Err(format!("video {id} is {}", state(&v)).into());
            }
            let path = args.out.clone().unwrap_or_else(|| format!("video-{id}.mp4"));
            fetch(remote, &v, &path)
        }
        "rm" => {
            let id = super::id(needs(words, "a video's id", VIDEOS))?;
            if !out::confirm(&format!("delete video {id}?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/v1/videos/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted video {id}"),
            }
            Ok(())
        }
        other => unknown("videos", other, VIDEOS),
    }
}

/// Where a video is, in a few words: `completed`, `stage 2, 61%, about 40 s
/// left`, `queued`.
/// A video's length: `4.04 s`, marked when the model chose it, or what it
/// is waiting on when it has not yet.
fn length(v: &Value) -> String {
    let chosen = v["kvad"]["length_chosen"] == json!(true);
    match (v["seconds"].as_str(), chosen) {
        (Some(s), false) => format!("{s} s"),
        (Some(s), true) => format!("{s} s, from the prompt"),
        (None, _) => "as long as the prompt wants".into(),
    }
}

fn state(v: &Value) -> String {
    let k = &v["kvad"];
    match v["status"].as_str().unwrap_or("?") {
        "in_progress" => {
            let p = k["progress"].as_f64().unwrap_or(0.0);
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0);
            // From the share done so far: rough early on, and not shown
            // until there is something to go on.
            let left = match (k["started_at"].as_f64(), p > 0.05) {
                (Some(t), true) => format!(", about {:.0} s left", (now - t) * (1.0 - p) / p),
                _ => String::new(),
            };
            format!("{}, {:.0}%{left}", out::s(&k["phase"]), p * 100.0)
        }
        "failed" => format!("failed: {}", out::s(&v["error"]["message"])),
        other => other.to_string(),
    }
}

/// Follow video `id` on its stream of events until it ends, showing where it
/// is on one line; the finished video comes back.
fn follow_video(remote: &Remote, id: &str) -> Res<Value> {
    let mut progress = out::Progress::new();
    for event in remote.stream("get", &format!("/v1/videos/{id}/events"), None)? {
        let event = event?;
        let v = event.json()?;
        match event.name.as_str() {
            "video.updated" => progress.show(format!("  {}", state(&v))),
            "video.completed" => {
                progress.done();
                return Ok(v);
            }
            "video.failed" => {
                progress.done();
                return Err(out::s(&v["error"]["message"]).into());
            }
            "video.deleted" => {
                progress.done();
                return Err(format!("{id} was deleted").into());
            }
            _ => {}
        }
    }
    progress.done();
    Err(format!("the server stopped telling us about {id} before it ended; `kvad videos show` says where it is").into())
}

/// Write a finished video's file to `path`.
fn fetch(remote: &Remote, video: &Value, path: &str) -> Res<()> {
    let id = out::s(&video["id"]);
    let bytes = remote.download(&format!("/v1/videos/{id}/content"), std::path::Path::new(path))?;
    eprintln!("wrote {path} ({:.1} MB)", bytes as f64 / 1e6);
    Ok(())
}

// ---------------------------------------------------------------------------
// Accounts
// ---------------------------------------------------------------------------

/// This machine's name, for naming the key a sign-in makes: "kvad CLI on
/// laptop" is a line somebody can recognise on the Account page and revoke.
fn hostname() -> String {
    let mut buf = [0u8; 256];
    // SAFETY: the buffer is as long as we say it is.
    let ok = unsafe { libc::gethostname(buf.as_mut_ptr().cast(), buf.len()) } == 0;
    let end = buf.iter().position(|&b| b == 0).unwrap_or(0);
    match ok && end > 0 {
        true => String::from_utf8_lossy(&buf[..end]).split('.').next().unwrap_or("").to_string(),
        false => "this machine".into(),
    }
}

/// The same server, proving itself a different way.
fn with(remote: &Remote, auth: Auth) -> Res<Remote> {
    let mut other = Remote::new(&remote.base, remote.why.clone())?;
    other.auth = auth;
    Ok(other)
}

/// A new password, asked twice.
fn new_password(question: &str) -> Res<String> {
    let first = out::secret(question)?;
    let again = out::secret("the same again")?;
    if first != again {
        return Err("those were not the same; nothing was changed".into());
    }
    Ok(first)
}

/// Make a key with an identity that has just signed in, and keep it.
fn keep_key(remote: &Remote, signed_in: &Remote, user: &str) -> Res<()> {
    let made = signed_in.post("/api/keys", &json!({ "name": format!("kvad CLI on {}", hostname()) }))?;
    let token = made["token"].as_str().ok_or("the server made a key and did not return it")?;
    let credential = client::Credential {
        key: token.to_string(),
        id: made["key"]["id"].as_i64(),
        user: Some(user.to_string()),
    };
    let path = client::remember(&remote.base, &credential)?;
    println!("signed in to {} as {user}", remote.base);
    println!("  key {} kept in {}, readable by you only", out::s(&made["key"]["prefix"]), path.display());
    Ok(())
}

pub fn auth(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "status");
    match sub {
        "status" => {
            let situation = remote.get("/api/auth")?;
            if args.json {
                out::json(&situation);
                return Ok(());
            }
            println!("server     {}", remote.base);
            println!("auth mode  {}", out::s(&situation["mode"]));
            if situation["needs_setup"] == true {
                println!("           no accounts yet: make the first with `kvad auth setup TOKEN`");
            }
            match situation["signed_in"].as_object() {
                Some(who) => println!("you are    {} ({})", out::s(&who["name"]), out::s(&who["role"])),
                None => println!("you are    not signed in: kvad auth login"),
            }
            match (std::env::var(client::KEY_VAR).is_ok(), client::credential(&remote.base)) {
                (true, _) => println!("key        from {}", client::KEY_VAR),
                (false, Some(_)) => println!("key        kept in {}", client::credentials_path().display()),
                (false, None) => {}
            }
            Ok(())
        }
        "login" => {
            let situation = remote.get("/api/auth")?;
            let mode = out::s(&situation["mode"]);
            if mode == "none" {
                println!("{} has no accounts (auth.mode = \"none\"), so there is nothing to sign in to", remote.base);
                return Ok(());
            }
            if args.key || mode == "oidc" {
                if !args.key {
                    eprintln!(
                        "{} signs in through an identity provider, which wants a browser.\n\
                         Sign in there, make a key on the Account page, and paste it here.",
                        remote.base
                    );
                }
                let key = out::secret("API key")?;
                let trying = with(remote, Auth::Bearer(key.clone()))?;
                let who = trying.get("/api/auth")?;
                let Some(name) = who["signed_in"]["name"].as_str() else {
                    return Err(format!("{} does not know that key", remote.base).into());
                };
                let path = client::remember(
                    &remote.base,
                    &client::Credential { key, id: None, user: Some(name.to_string()) },
                )?;
                println!("signed in to {} as {name}\n  key kept in {}, readable by you only", remote.base, path.display());
                return Ok(());
            }

            let name = match &args.name {
                Some(n) => n.clone(),
                None => out::ask("name")?,
            };
            let password = out::secret("password")?;
            match mode.as_str() {
                // HTTP Basic is a credential on every request, so it can make
                // the key directly.
                "basic" => keep_key(remote, &with(remote, Auth::Basic { name: name.clone(), password })?, &name),
                // A session: sign in for a cookie, make the key with it, and
                // end the session, which nothing needs once the key exists.
                _ => {
                    let (_, cookie) = remote.call_for_cookie(
                        "post",
                        "/api/auth/login",
                        Some(Body::Json(&json!({ "name": name, "password": password }))),
                    )?;
                    let cookie = cookie.ok_or("the server signed you in and set no session")?;
                    let session = with(remote, Auth::Cookie(cookie))?;
                    let kept = keep_key(remote, &session, &name);
                    let _ = session.post("/api/auth/logout", &json!({}));
                    kept
                }
            }
        }
        "logout" => {
            let Some(credential) = client::forget(&remote.base)? else {
                println!("no key kept for {}", remote.base);
                return Ok(());
            };
            // Revoked as well as forgotten: a key left alive on the server
            // is a key that works for whoever finds a copy.
            if let Some(id) = credential.id {
                let keyed = with(remote, Auth::Bearer(credential.key.clone()))?;
                match keyed.delete(&format!("/api/keys/{id}")) {
                    Ok(_) => println!("revoked key {id} on {}", remote.base),
                    Err(e) => eprintln!("could not revoke key {id} on {}: {e}\n  revoke it on the Account page", remote.base),
                }
            }
            println!("forgot the key for {}", remote.base);
            Ok(())
        }
        "setup" => {
            let token = needs(words, "the setup token the server printed", AUTH);
            let name = match &args.name {
                Some(n) => n.clone(),
                None => out::ask("name for the first account (an administrator)")?,
            };
            let password = new_password("password")?;
            let (_, cookie) = remote.call_for_cookie(
                "post",
                "/api/auth/setup",
                Some(Body::Json(&json!({ "token": token, "name": name, "password": password }))),
            )?;
            let cookie = cookie.ok_or("the server made the account and set no session")?;
            let session = with(remote, Auth::Cookie(cookie))?;
            let kept = keep_key(remote, &session, &name);
            let _ = session.post("/api/auth/logout", &json!({}));
            kept
        }
        "password" => {
            let current = out::secret("current password")?;
            let new = new_password("new password")?;
            let done = remote.post("/api/auth/password", &json!({ "current": current, "new": new }))?;
            match args.json {
                true => out::json(&done),
                false => println!("password changed; every other session of yours has ended"),
            }
            Ok(())
        }
        other => unknown("auth", other, AUTH),
    }
}

/// A user by id, or by name.
fn user_id(remote: &Remote, word: &str) -> Res<i64> {
    if let Ok(id) = word.parse() {
        return Ok(id);
    }
    let list = remote.get("/api/users")?;
    out::items(&list)
        .iter()
        .find(|u| u["name"] == word)
        .and_then(|u| u["id"].as_i64())
        .ok_or_else(|| format!("there is no account `{word}`").into())
}

pub fn users(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/users")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|u| {
                    vec![
                        out::s(&u["id"]),
                        out::s(&u["name"]),
                        out::s(&u["role"]),
                        out::s(&u["email"]),
                        out::s(&u["last_seen_at"]),
                    ]
                })
                .collect();
            out::table(&["ID", "NAME", "ROLE", "EMAIL", "LAST SEEN"], &rows);
            Ok(())
        }
        "add" => {
            let name = needs(words, "a name", USERS);
            let password = new_password(&format!("password for {name}"))?;
            let role = args.role.clone().unwrap_or_else(|| "user".into());
            let made = remote.post("/api/users", &json!({ "name": name, "password": password, "role": role }))?;
            match args.json {
                true => out::json(&made),
                false => println!("account {} `{}` ({})", out::s(&made["id"]), out::s(&made["name"]), out::s(&made["role"])),
            }
            Ok(())
        }
        "edit" => {
            let id = user_id(remote, needs(words, "an account", USERS))?;
            let mut body = json!({});
            if let Some(role) = &args.role {
                body["role"] = json!(role);
            }
            if args.password {
                body["password"] = json!(new_password("new password")?);
            }
            if body.as_object().is_some_and(|o| o.is_empty()) {
                eprintln!("change what?  kvad users edit {id} --role admin|user, or --password");
                std::process::exit(2);
            }
            let changed = remote.patch(&format!("/api/users/{id}"), &body)?;
            match args.json {
                true => out::json(&changed),
                false => println!("account {id} changed"),
            }
            Ok(())
        }
        "rm" => {
            let id = user_id(remote, needs(words, "an account", USERS))?;
            if !out::confirm(&format!("delete account {id}, with its sessions, keys and conversations?"), args.yes)? {
                println!("cancelled");
                return Ok(());
            }
            let gone = remote.delete(&format!("/api/users/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("deleted account {id}"),
            }
            Ok(())
        }
        other => unknown("users", other, USERS),
    }
}

pub fn sessions(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/sessions")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|s| {
                    vec![
                        out::cut(&out::s(&s["token_hash"]), 13),
                        out::s(&s["created_at"]),
                        out::s(&s["expires_at"]),
                        out::cut(&out::s(&s["user_agent"]), 50),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no sessions"),
                false => out::table(&["HASH", "SIGNED IN", "EXPIRES", "FROM"], &rows),
            }
            Ok(())
        }
        "rm" => {
            let start = needs(words, "a session's hash", SESSIONS).trim_end_matches('…');
            let list = remote.get("/api/sessions")?;
            let found: Vec<String> = out::items(&list)
                .iter()
                .filter_map(|s| s["token_hash"].as_str())
                .filter(|h| h.starts_with(start))
                .map(str::to_string)
                .collect();
            let hash = match found.as_slice() {
                [one] => one,
                [] => return Err(format!("no session of yours starts {start}").into()),
                _ => return Err(format!("{} sessions start {start}; give more of it", found.len()).into()),
            };
            let gone = remote.delete(&format!("/api/sessions/{hash}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("ended session {}", out::cut(hash, 13)),
            }
            Ok(())
        }
        other => unknown("sessions", other, SESSIONS),
    }
}

pub fn keys(remote: &Remote, args: &Args) -> Res<()> {
    let (sub, words) = split(args, "ls");
    match sub {
        "ls" => {
            let list = remote.get("/api/keys")?;
            if args.json {
                out::json(&list);
                return Ok(());
            }
            let rows: Vec<Vec<String>> = out::items(&list)
                .iter()
                .map(|k| {
                    vec![
                        out::s(&k["id"]),
                        out::s(&k["name"]),
                        format!("{}…", out::s(&k["prefix"])),
                        out::s(&k["created_at"]),
                        out::s(&k["last_used_at"]),
                    ]
                })
                .collect();
            match rows.is_empty() {
                true => println!("no keys. make one:  kvad keys add NAME"),
                false => out::table(&["ID", "NAME", "STARTS", "MADE", "LAST USED"], &rows),
            }
            Ok(())
        }
        "add" => {
            let name = words.join(" ");
            if name.is_empty() {
                eprintln!("{KEYS}");
                std::process::exit(2);
            }
            let made = remote.post("/api/keys", &json!({ "name": name }))?;
            if args.json {
                out::json(&made);
                return Ok(());
            }
            println!("key {} `{name}`:\n\n    {}\n", out::s(&made["key"]["id"]), out::s(&made["token"]));
            println!("This is the only time it is shown: only its hash is kept.");
            Ok(())
        }
        "rm" => {
            let id = super::id(needs(words, "a key's id", KEYS))?;
            let gone = remote.delete(&format!("/api/keys/{id}"))?;
            match args.json {
                true => out::json(&gone),
                false => println!("revoked key {id}"),
            }
            Ok(())
        }
        other => unknown("keys", other, KEYS),
    }
}

// ---------------------------------------------------------------------------
// Any route
// ---------------------------------------------------------------------------

/// `kvad api` — the escape hatch, and the index.
///
/// With no arguments it lists every route from the server's own description,
/// which is the answer to "what can this server do" that cannot go stale.
/// With a path it sends one request and prints what came back.
pub fn raw(remote: &Remote, args: &Args) -> Res<()> {
    let words = &args.words;
    if words.is_empty() {
        let doc = remote.get("/api/openapi.json")?;
        if args.json {
            out::json(&doc);
            return Ok(());
        }
        let mut rows = Vec::new();
        for (path, operations) in doc["paths"].as_object().into_iter().flatten() {
            for (method, op) in operations.as_object().into_iter().flatten() {
                rows.push(vec![method.to_uppercase(), path.clone(), out::s(&op["summary"])]);
            }
        }
        out::table(&["", "PATH", "WHAT IT DOES"], &rows);
        return Ok(());
    }

    let (method, path, body) = match words.as_slice() {
        [path] if path.starts_with('/') => ("get".to_string(), path.clone(), None),
        [method, path] => (method.to_ascii_lowercase(), path.clone(), None),
        [method, path, body] => (method.to_ascii_lowercase(), path.clone(), Some(body.clone())),
        _ => {
            eprintln!("{API}");
            std::process::exit(2);
        }
    };
    if !path.starts_with('/') {
        return Err(format!("`{path}` is not a path; paths start with /, like /api/health").into());
    }
    let body = match body.as_deref() {
        Some("-") => {
            let mut all = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut all)?;
            Some(all)
        }
        other => other.map(str::to_string),
    };
    let json_body = body.as_deref().and_then(|b| serde_json::from_str::<Value>(b).ok());
    let body = match (&json_body, body) {
        (Some(v), _) => Some(Body::Json(v)),
        (None, Some(text)) => Some(Body::Text(text)),
        (None, None) => None,
    };
    match remote.exchange(&method, &path, body)? {
        client::Reply::Json(value) => out::json(&value),
        client::Reply::Events(events) => {
            for event in events {
                println!("{}", event?.to_json());
            }
        }
    }
    Ok(())
}
