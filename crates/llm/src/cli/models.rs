//! The commands that also run in this process, answered by a server instead.
//!
//! Each prints what its local twin in `main.rs` prints, as nearly as the
//! server's answer allows, so that output does not change shape depending
//! on whether a service happened to be running. What a server knows that
//! this process does not — which models are in memory, which can be offered
//! tools — is added, never swapped in for something that was there.
//!
//! Three commands have no local twin, because residency is a service
//! concept: in this process a model is loaded by the command that uses it
//! and let go when that command ends. `ps`, `load` and `unload` are here
//! because they are about models all the same.

use super::out;
use crate::Args;
use kvad::client::{Body, Remote};
use serde_json::{json, Value};
use std::io::{BufRead, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// The commands with a local twin, which run here when no server answers.
pub const LOCAL_TOO: &[&str] = &["ls", "search", "info", "pull", "use", "rm", "cache", "run", "chat", "train"];

/// `kvad ls`.
pub fn ls(remote: &Remote, args: &Args) -> Res<()> {
    let listing = remote.get("/api/models")?;
    if args.json {
        out::json(&listing);
        return Ok(());
    }
    // Which models can be offered tools is a fact about their templates that
    // only the OpenAI listing carries. Worth a second request: it is the
    // question somebody picking a model for an agent is asking.
    let tools: Vec<String> = remote
        .get("/v1/models")
        .map(|v| {
            out::items(&v["data"])
                .iter()
                .filter(|m| m["kvad"]["tools"] == true)
                .filter_map(|m| m["id"].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let resident: Vec<&str> =
        out::items(&listing["residents"]).iter().filter_map(|r| r["repo"].as_str()).collect();

    let (downloaded, trained) = (out::items(&listing["downloaded"]), out::items(&listing["trained"]));
    if downloaded.is_empty() && trained.is_empty() {
        println!("nothing on {} yet. try:  kvad pull HuggingFaceTB/SmolLM2-135M-Instruct", remote.base);
        return Ok(());
    }

    let active = listing["active"].as_str();
    let mut total = 0;
    let mut row = |m: &Value, unfinished: &str| {
        total += m["bytes"].as_u64().unwrap_or(0);
        let id = out::s(&m["id"]);
        let marker = if active == Some(id.as_str()) { " *" } else { "" };
        let mut notes = String::new();
        if m["complete"] == false {
            notes.push_str(unfinished);
        }
        if resident.iter().any(|r| r.eq_ignore_ascii_case(&id)) {
            notes.push_str("  in memory");
        }
        if tools.contains(&id) {
            notes.push_str("  tools");
        }
        let arch = m["arch"].as_str().unwrap_or("?");
        println!("{:<46} {arch:<11} {:>9}{marker}{notes}", out::cut(&id, 46), out::bytes(&m["bytes"]));
    };

    if !downloaded.is_empty() {
        println!("{:<46} {:<11} {:>9}", "DOWNLOADED", "ARCH", "SIZE");
        for m in downloaded {
            row(m, "  (config only)");
        }
    }
    if !trained.is_empty() {
        if !downloaded.is_empty() {
            println!();
        }
        println!("{:<46} {:<11} {:>9}", "TRAINED HERE", "ARCH", "SIZE");
        for m in trained {
            row(m, "  (unfinished — no weights)");
        }
    }

    println!("\n{} models, {}", downloaded.len() + trained.len(), kvad::hub::human_bytes(total));
    match active {
        Some(a) => println!("* active: {a}"),
        None => println!("no active model set (using {})", crate::DEFAULT_MODEL),
    }
    println!("on: {}", remote.base);
    Ok(())
}

/// `kvad ps` — what is in memory, and what is left for more.
pub fn ps(remote: &Remote, args: &Args) -> Res<()> {
    let listing = remote.get("/api/models")?;
    if args.json {
        out::json(&json!({
            "residents": listing["residents"],
            "memory": listing["memory"],
            "queue_depth": listing["queue_depth"],
        }));
        return Ok(());
    }
    let residents = out::items(&listing["residents"]);
    if residents.is_empty() {
        println!("nothing in memory. load one:  kvad load MODEL");
    } else {
        let rows: Vec<Vec<String>> = residents
            .iter()
            .map(|r| {
                let mut notes = Vec::new();
                if r["instruct"] == false {
                    notes.push("base model");
                }
                if r["tools"] == true {
                    notes.push("tools");
                }
                if r["streams"] == true {
                    notes.push("streams its experts from disk");
                }
                vec![
                    out::s(&r["id"]),
                    out::bytes(&r["commit"]),
                    out::s(&r["n_ctx"]),
                    out::s(&r["cached_tokens"]),
                    notes.join(", "),
                ]
            })
            .collect();
        out::table(&["MODEL", "CHARGED", "CONTEXT", "CACHED", ""], &rows);
    }
    let m = &listing["memory"];
    println!(
        "\nmemory: {} of {} left · each charged for {} tokens of KV cache",
        out::bytes(&m["left"]),
        out::bytes(&m["total"]),
        out::s(&m["context"]),
    );
    if let Some(depth) = listing["queue_depth"].as_u64().filter(|&d| d > 0) {
        println!("queue: {depth} waiting");
    }
    Ok(())
}

/// `kvad search`.
pub fn search(remote: &Remote, args: &Args) -> Res<()> {
    let Some(query) = &args.target else {
        eprintln!("usage: kvad search QUERY");
        std::process::exit(2);
    };
    let results = remote.get(&format!("/api/models/search?q={}&limit=20", super::enc(query)))?;
    if args.json {
        out::json(&results);
        return Ok(());
    }
    let results = out::items(&results);
    if results.is_empty() {
        println!("no models matched `{query}`");
        return Ok(());
    }
    println!("{:<40} {:>10} {:>9}  {:<11} {}", "MODEL", "DOWNLOADS", "SIZE", "ARCH", "STATUS");
    for m in results {
        let status = match (&m["blocker"], m["local"] == true) {
            (Value::String(reason), _) => reason.clone(),
            (_, true) => "downloaded".into(),
            _ if m["looks_instruct"] == true => format!("chat · {}", fit(m)),
            _ => format!("completion · {}", fit(m)),
        };
        // Names come from the Hub; print them, never act on them.
        println!(
            "{:<40} {:>10} {:>9}  {:<11} {status}",
            out::cut(&out::s(&m["id"]), 40),
            out::s(&m["downloads"]),
            out::bytes(&m["bytes"]),
            m["arch"].as_str().unwrap_or("-"),
        );
    }
    let runnable = results.iter().filter(|m| m["runnable"] == true).count();
    println!("\n{runnable} of {} runnable on {}.", results.len(), remote.base);
    Ok(())
}

/// What `kvad search` says about whether a model fits, from the fields a
/// search result carries.
fn fit(m: &Value) -> String {
    let per_token = || out::bytes(&m["disk_per_token"]);
    match (m["crawls"] == true, m["streams"] == true, m["fits_at"].as_str()) {
        (true, _, _) => format!("crawls — ~{} a token from disk", per_token()),
        (_, true, Some(p)) => format!("streams at {p} — ~{} a token from disk", per_token()),
        (_, _, Some(p)) => format!("fits at {p}"),
        _ => "?".into(),
    }
}

/// `kvad info`.
///
/// A model on the server's disk is described from its own config, which the
/// listing carries; one that is not is asked of the Hub, as the local command
/// does, but by the server.
pub fn info(remote: &Remote, args: &Args) -> Res<()> {
    let listing = remote.get("/api/models")?;
    let repo = match args.target.clone().or_else(|| args.model.clone()) {
        Some(repo) => repo,
        None => listing["active"].as_str().unwrap_or(crate::DEFAULT_MODEL).to_string(),
    };
    let here = on_disk(&listing)
        .find(|m| m["id"].as_str().is_some_and(|id| id.eq_ignore_ascii_case(&repo)))
        .map(|m| m["detail"].clone())
        .filter(Value::is_object);
    let detail = match here {
        Some(detail) => detail,
        None => remote.get(&format!("/api/models/detail?repo={}", super::enc(&repo)))?,
    };
    if args.json {
        out::json(&detail);
        return Ok(());
    }
    println!("{}", out::s(&detail["summary"]));
    if let Some(params) = detail["params"].as_u64() {
        println!("  {:.1}M parameters", params as f64 / 1e6);
    }
    let e = &detail["experts"];
    if e.is_object() {
        println!(
            "  {} experts, {} a token, in {} layers: {} read before one repeats",
            out::s(&e["count"]),
            out::s(&e["per_token"]),
            out::s(&e["layers"]),
            out::s(&e["working_set"]),
        );
    }
    if let Some(stored) = detail["stored_as"].as_str() {
        println!("  stored as {stored}");
    }
    if let Some(line) = detail["fit"]["line"].as_str() {
        println!("  on {}: {line}", remote.base);
    }
    Ok(())
}

/// `kvad pull` — a job on the server, followed until it ends.
pub fn pull(remote: &Remote, args: &Args) -> Res<()> {
    let Some(repo) = args.target.clone().or_else(|| args.model.clone()) else {
        eprintln!("usage: kvad pull REPO [--dev]");
        std::process::exit(2);
    };
    // `--dev`: LTX-2.5's dev model and distilled LoRA too, 51 GB, which a
    // video asks for by giving steps, guidance or a negative prompt.
    let job = remote.post("/api/models/pull", &json!({ "repo": repo, "dev": args.dev }))?;
    super::api::follow(remote, &job, args)?;
    match (args.json, args.dev) {
        (true, _) => {}
        (false, true) => println!("\nguided videos, 30 steps unless --steps says:  kvad videos make \"…\" --guidance 3"),
        (false, false) => println!("\nrun it with:  kvad run --model {repo}"),
    }
    Ok(())
}

/// Every model the server has on disk, by id.
fn on_disk(listing: &Value) -> impl Iterator<Item = &Value> {
    out::items(&listing["trained"]).iter().chain(out::items(&listing["downloaded"]))
}

/// `kvad use`.
pub fn use_model(remote: &Remote, args: &Args) -> Res<()> {
    let Some(repo) = &args.target else {
        eprintln!("usage: kvad use REPO");
        std::process::exit(2);
    };
    let listing = remote.get("/api/models")?;
    if !on_disk(&listing).any(|m| m["id"].as_str().is_some_and(|id| id.eq_ignore_ascii_case(repo))) {
        match repo.contains('/') {
            true => eprintln!("`{repo}` is not downloaded on {}. Run:  kvad pull {repo}", remote.base),
            false => eprintln!("`{repo}` is not a model on {}. Run `kvad ls` to see what is.", remote.base),
        }
        std::process::exit(1);
    }
    let set = remote.post("/api/models/active", &json!({ "repo": repo }))?;
    match args.json {
        true => out::json(&set),
        false => println!("active model is now {repo}"),
    }
    Ok(())
}

/// `kvad rm`.
pub fn remove(remote: &Remote, args: &Args) -> Res<()> {
    let Some(repo) = &args.target else {
        eprintln!("usage: kvad rm REPO");
        std::process::exit(2);
    };
    let listing = remote.get("/api/models")?;
    let Some(model) =
        on_disk(&listing).find(|m| m["id"].as_str().is_some_and(|id| id.eq_ignore_ascii_case(repo)))
    else {
        eprintln!("`{repo}` is not a model on {}. Run `kvad ls` to see what is.", remote.base);
        std::process::exit(1);
    };
    let id = out::s(&model["id"]);

    // Irreversible, and the download may have been slow: say exactly what
    // will go, and want a typed yes.
    println!("about to delete, on {}:", remote.base);
    println!("  {id} ({})", out::bytes(&model["bytes"]));
    let question = match model["trained"] == true {
        true => "\nit was trained there and is not on the Hub; deleting it is final. delete?",
        false => "\nre-download would be needed to use it again. delete?",
    };
    if !out::confirm(question, args.yes)? {
        println!("cancelled");
        return Ok(());
    }
    let gone = remote.delete(&format!("/api/models?id={}", super::enc(&id)))?;
    if args.json {
        out::json(&gone);
        return Ok(());
    }
    match gone["qcache_files"].as_u64().unwrap_or(0) {
        0 => {}
        n => println!("also deleted {n} pre-quantised file(s)"),
    }
    if listing["active"].as_str() == Some(id.as_str()) {
        println!("(was the active model; cleared)");
    }
    println!("deleted {id}");
    Ok(())
}

/// `kvad cache`.
pub fn cache(remote: &Remote, args: &Args) -> Res<()> {
    let listing = remote.get("/api/models")?;
    let entries = out::items(&listing["qcache"]);
    let forget = |which: Vec<&Value>| -> Res<u64> {
        let mut files = 0;
        for e in which {
            let (repo, precision) = (super::enc(&out::s(&e["repo"])), super::enc(&out::s(&e["precision"])));
            files += remote.delete(&format!("/api/qcache?repo={repo}&precision={precision}"))?["files"]
                .as_u64()
                .unwrap_or(0);
        }
        Ok(files)
    };
    match args.target.as_deref() {
        Some("clear") => {
            println!("deleted {} file(s)", forget(entries.iter().collect())?);
            return Ok(());
        }
        Some(repo) => {
            let n = forget(entries.iter().filter(|e| e["repo"] == repo).collect())?;
            println!("deleted {n} file(s) for {repo}");
            return Ok(());
        }
        None => {}
    }
    if args.json {
        out::json(&listing["qcache"]);
        return Ok(());
    }
    if entries.is_empty() {
        println!("nothing pre-quantised on {} yet.", remote.base);
        return Ok(());
    }
    println!("{:<44} {:<8} {:>10}", "MODEL", "QUANT", "SIZE");
    let mut total = 0;
    for e in entries {
        total += e["bytes"].as_u64().unwrap_or(0);
        println!(
            "{:<44} {:<8} {:>10}",
            out::cut(&out::s(&e["repo"]), 43),
            out::s(&e["precision"]),
            out::bytes(&e["bytes"])
        );
    }
    println!("\n{} file(s), {}", entries.len(), kvad::hub::human_bytes(total));
    Ok(())
}

/// Which backend a load asks for: `--backend`, else the CPU at `--quant`,
/// else nothing, which leaves the choice to the server.
fn backend(args: &Args) -> Option<String> {
    args.backend.clone().or_else(|| args.quant_given.then(|| format!("cpu-{}", args.quant)))
}

/// `kvad load`.
pub fn load(remote: &Remote, args: &Args) -> Res<Value> {
    let Some(repo) = args.target.clone().or_else(|| args.model.clone()) else {
        eprintln!("usage: kvad load MODEL [--backend ID]");
        std::process::exit(2);
    };
    load_named(remote, &repo, backend(args), args.json)
}

/// Load a model on the server, showing what it is doing, and return what it
/// loaded.
fn load_named(remote: &Remote, repo: &str, backend: Option<String>, raw: bool) -> Res<Value> {
    let mut body = json!({ "repo": repo });
    if let Some(b) = &backend {
        body["backend"] = json!(b);
    }
    let mut progress = out::Progress::new();
    for event in remote.stream("post", "/api/models/load", Some(Body::Json(&body)))? {
        let event = event?;
        if raw {
            println!("{}", event.to_json());
        }
        let data = event.json()?;
        match event.name.as_str() {
            "progress" => match data["kind"].as_str() {
                Some("download") => progress.show(format!(
                    "  {}  {} of {}",
                    out::s(&data["file"]),
                    out::bytes(&data["bytes"]),
                    out::bytes(&data["total"])
                )),
                Some("status") if !raw => {
                    progress.done();
                    eprintln!("  {}", out::s(&data["message"]));
                }
                _ => {}
            },
            "loaded" => {
                progress.done();
                if !raw {
                    eprintln!("loaded {} on {}", out::s(&data["repo"]), out::s(&data["backend"]));
                    eprintln!("  {}", out::s(&data["summary"]));
                    eprintln!(
                        "  {:.1}M parameters · weights {} · context {} · {}",
                        data["params"].as_f64().unwrap_or(0.0) / 1e6,
                        out::bytes(&data["weight_bytes"]),
                        out::s(&data["n_ctx"]),
                        if data["instruct"] == true { "instruction-tuned" } else { "base model (completion only)" }
                    );
                }
                return Ok(data);
            }
            "error" => {
                progress.done();
                let mut why = out::s(&data["error"]);
                if data["full"] == true {
                    why.push_str("\n  Nothing is unloaded to make room. See what is in memory with `kvad ps`,\n  and give some back with `kvad unload ID`.");
                }
                return Err(why.into());
            }
            _ => {}
        }
    }
    Err("the server stopped answering before the model loaded".into())
}

/// `kvad unload [ID]`.
pub fn unload(remote: &Remote, args: &Args) -> Res<()> {
    let body = match &args.target {
        Some(id) => json!({ "id": id }),
        None => json!({}),
    };
    let gone = remote.post("/api/models/unload", &body)?;
    if args.json {
        out::json(&gone);
        return Ok(());
    }
    let ids: Vec<String> = out::items(&gone["unloaded"]).iter().map(out::s).collect();
    match ids.is_empty() {
        true => println!("nothing was in memory"),
        false => println!("unloaded {}", ids.join(", ")),
    }
    Ok(())
}

/// `kvad cancel`.
pub fn cancel(remote: &Remote, args: &Args) -> Res<()> {
    let answer = remote.delete("/api/generation")?;
    match args.json {
        true => out::json(&answer),
        false => println!("asked whatever is generating on {} to stop", remote.base),
    }
    Ok(())
}

/// `kvad tokenize TEXT`.
pub fn tokenize(remote: &Remote, args: &Args) -> Res<()> {
    let text = match &args.target {
        Some(t) => t.clone(),
        None => {
            let mut all = String::new();
            std::io::Read::read_to_string(&mut std::io::stdin(), &mut all)?;
            all
        }
    };
    let mut body = json!({ "text": text });
    if let Some(m) = &args.model {
        body["model"] = json!(m);
    }
    let split = remote.post("/api/playground/tokenize", &body)?;
    if args.json {
        out::json(&split);
        return Ok(());
    }
    let rows: Vec<Vec<String>> = out::items(&split["tokens"])
        .iter()
        .map(|t| vec![out::s(&t["id"]), format!("{:?}", out::s(&t["token"])), format!("{:?}", out::s(&t["piece"]))])
        .collect();
    out::table(&["ID", "TOKEN", "PIECE"], &rows);
    println!("\n{} tokens for {} characters", out::s(&split["count"]), out::s(&split["characters"]));
    Ok(())
}

// ---------------------------------------------------------------------------
// Generating
// ---------------------------------------------------------------------------

/// The model a generation goes to, in memory on the server by the time this
/// returns.
///
/// The same choice the local commands make — `--model`, else the active
/// model, else the default — with one addition: with no active model set,
/// the one model in memory is taken before the default is loaded beside it.
/// A model that is not in memory is loaded with its progress shown, which is
/// better than a completion that sits silent for the tens of seconds a load
/// takes.
fn ready(remote: &Remote, args: &Args) -> Res<Value> {
    let listing = remote.get("/api/models")?;
    let residents = out::items(&listing["residents"]);
    let backend = backend(args);
    let matches = |r: &Value, name: &str| {
        r["id"] == name
            || match &backend {
                Some(b) => r["id"] == format!("{name}@{b}").as_str(),
                None => r["repo"].as_str().is_some_and(|repo| repo.eq_ignore_ascii_case(name)),
            }
    };
    let name = match (&args.model, listing["active"].as_str(), residents) {
        (Some(name), _, _) => name.clone(),
        (None, Some(active), _) => active.to_string(),
        (None, None, [only]) if backend.is_none() => return said(only.clone()),
        (None, None, []) => crate::DEFAULT_MODEL.to_string(),
        (None, None, several) => {
            let ids: Vec<String> = several.iter().map(|r| out::s(&r["id"])).collect();
            return Err(format!(
                "{} models are in memory, none is the active model, and none was named:\n  {}\n\
                 Name one with --model, or set the active model with `kvad use`.",
                ids.len(),
                ids.join(", ")
            )
            .into());
        }
    };
    // The most recently loaded, when the same weights are in memory twice.
    if let Some(r) = residents.iter().rev().find(|r| matches(r, &name)) {
        return said(r.clone());
    }

    let loaded = load_named(remote, &name, backend, false)?;
    // The resident's id is what later requests name it by, and only the
    // listing has it: a load answers with the model, not with its key.
    let listing = remote.get("/api/models")?;
    let resident = out::items(&listing["residents"])
        .iter()
        .rev()
        .find(|r| r["repo"] == loaded["repo"])
        .cloned()
        .ok_or_else(|| format!("{} loaded, and then was not in memory", out::s(&loaded["repo"])))?;
    said(resident)
}

fn said(resident: Value) -> Res<Value> {
    eprintln!("model: {}", out::s(&resident["id"]));
    Ok(resident)
}

/// The sampling a generation asks for, from the same flags the local
/// commands read.
fn sampling(args: &Args, body: &mut Value) {
    body["temperature"] = json!(args.temperature);
    body["top_k"] = json!(args.top_k);
    body["top_p"] = json!(args.top_p);
    body["max_tokens"] = json!(args.max_tokens);
    body["seed"] = json!(args.seed);
}

/// What a streamed reply came to.
struct Reply {
    text: String,
    usage: Value,
    kvad: Value,
}

/// Stream a chat completion to the terminal as it arrives.
///
/// The answer goes to stdout. A reasoning model's working goes to stderr,
/// dimmed, because it is not the answer: a pipe gets the reply and the
/// terminal gets both.
fn stream_chat(remote: &Remote, body: &Value, raw: bool) -> Res<Reply> {
    let mut reply = Reply { text: String::new(), usage: Value::Null, kvad: Value::Null };
    let mut thinking = false;
    let dim = std::io::IsTerminal::is_terminal(&std::io::stderr());
    for event in remote.stream("post", "/v1/chat/completions", Some(Body::Json(body)))? {
        let event = event?;
        if raw {
            println!("{}", event.to_json());
        }
        if event.name == "error" {
            return Err(out::s(&event.json()?["error"]).into());
        }
        if event.data == "[DONE]" {
            break;
        }
        let chunk = event.json()?;
        let delta = &chunk["choices"][0]["delta"];
        if let Some(t) = delta["reasoning_content"].as_str() {
            if !raw {
                match dim {
                    true => eprint!("\x1b[2m{t}\x1b[0m"),
                    false => eprint!("{t}"),
                }
                thinking = true;
            }
        }
        if let Some(t) = delta["content"].as_str() {
            if !raw {
                if std::mem::take(&mut thinking) {
                    eprintln!("\n");
                }
                print!("{t}");
                std::io::stdout().flush()?;
            }
            reply.text.push_str(t);
        }
        if chunk["usage"].is_object() {
            reply.usage = chunk["usage"].clone();
            reply.kvad = chunk["kvad"].clone();
        }
    }
    Ok(reply)
}

/// `kvad run`.
pub fn run(remote: &Remote, args: &Args) -> Res<()> {
    let model = ready(remote, args)?;
    let id = out::s(&model["id"]);
    // A base model has no template to put a prompt in, so it continues text,
    // as it does locally. `--raw` asks the same of an instruct model.
    let raw = args.raw || model["instruct"] == false;
    let prompt = args.prompt.clone().unwrap_or_else(|| match raw {
        false => "Explain what a neural network is, in two sentences.".into(),
        true => "The first time I saw the sea,".into(),
    });

    if raw {
        let mut body = json!({ "model": id, "prompt": prompt });
        sampling(args, &mut body);
        if !args.json {
            print!("{prompt}");
            std::io::stdout().flush()?;
        }
        for event in remote.stream("post", "/api/playground/complete", Some(Body::Json(&body)))? {
            let event = event?;
            if args.json {
                println!("{}", event.to_json());
            }
            let data = event.json()?;
            match event.name.as_str() {
                "token" if !args.json => {
                    print!("{}", data["text"].as_str().unwrap_or(""));
                    std::io::stdout().flush()?;
                }
                "done" if !args.json => eprintln!(
                    "\n\n[prefill {} tokens in {:.2}s · generated {} in {:.2}s = {:.1} tok/s]",
                    out::s(&data["prompt_tokens"]),
                    data["prefill_secs"].as_f64().unwrap_or(0.0),
                    out::s(&data["generated_tokens"]),
                    data["decode_secs"].as_f64().unwrap_or(0.0),
                    data["decode_per_sec"].as_f64().unwrap_or(0.0),
                ),
                "error" => return Err(out::s(&data["error"]).into()),
                _ => {}
            }
        }
        return Ok(());
    }

    let mut messages = Vec::new();
    if let Some(s) = &args.system {
        messages.push(json!({ "role": "system", "content": s }));
    }
    messages.push(json!({ "role": "user", "content": prompt }));
    let mut body = json!({ "model": id, "messages": messages, "stream": true });
    sampling(args, &mut body);
    let reply = stream_chat(remote, &body, args.json)?;
    if !args.json {
        eprintln!(
            "\n\n[prefill {} tokens in {:.2}s · generated {} in {:.2}s = {:.1} tok/s]",
            out::s(&reply.usage["prompt_tokens"]),
            reply.kvad["prefill_secs"].as_f64().unwrap_or(0.0),
            out::s(&reply.usage["completion_tokens"]),
            reply.kvad["decode_secs"].as_f64().unwrap_or(0.0),
            reply.kvad["decode_tokens_per_sec"].as_f64().unwrap_or(0.0),
        );
    }
    Ok(())
}

/// `kvad chat`.
///
/// The history is kept here and sent whole every turn, as any OpenAI client
/// does; the server's KV cache is what makes that cheap, since only the
/// newest message is new to it. With `--save` or `--conversation` the turns
/// are also written to a conversation on the server, which is then in the
/// web UI's list beside the ones made there.
pub fn chat(remote: &Remote, args: &Args) -> Res<()> {
    let model = ready(remote, args)?;
    let model_id = out::s(&model["id"]);
    if model["instruct"] == false {
        return Err(format!(
            "{model_id} is a base model with no chat template: it continues text rather than\n\
             answering. `kvad run --model {}` gives it something to continue.",
            out::s(&model["repo"])
        )
        .into());
    }

    let mut system = args.system.clone();
    let mut history: Vec<Value> = Vec::new();
    let mut kept: Option<i64> = None;
    if let Some(id) = args.conversation {
        let conversation = remote.get(&format!("/api/conversations/{id}"))?;
        system = system.or_else(|| conversation["system"].as_str().map(str::to_string));
        for m in out::items(&conversation["messages"]) {
            history.push(json!({ "role": m["role"], "content": m["content"] }));
        }
        kept = Some(id);
        eprintln!(
            "carrying on with “{}”: {} messages so far",
            out::s(&conversation["title"]),
            history.len()
        );
    }

    eprintln!("\nType a message, or /reset to clear history, /quit to exit.\n");
    let stdin = std::io::stdin();
    loop {
        print!("\x1b[1m>\x1b[0m ");
        std::io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break;
        }
        let line = line.trim();
        match line {
            "" => continue,
            "/quit" | "/exit" => break,
            "/reset" => {
                history.clear();
                // A kept conversation stays as it was; the next message
                // starts another, if this chat is keeping them.
                kept = None;
                eprintln!("(history cleared)");
                continue;
            }
            _ => {}
        }

        if kept.is_none() && args.save {
            let made = remote.post("/api/conversations", &json!({ "system": system, "model": model_id }))?;
            kept = made["id"].as_i64();
        }
        if let Some(id) = kept {
            remote.post(
                &format!("/api/conversations/{id}/messages"),
                &json!({ "role": "user", "content": line, "title_if_unnamed": true }),
            )?;
        }
        history.push(json!({ "role": "user", "content": line }));

        let mut messages = Vec::new();
        if let Some(s) = &system {
            messages.push(json!({ "role": "system", "content": s }));
        }
        messages.extend(history.iter().cloned());
        let mut body = json!({ "model": model_id, "messages": messages, "stream": true });
        sampling(args, &mut body);

        let reply = stream_chat(remote, &body, false)?;
        println!();
        eprintln!(
            "  [{} tok, {:.1} tok/s, {} cached]\n",
            out::s(&reply.usage["completion_tokens"]),
            reply.kvad["decode_tokens_per_sec"].as_f64().unwrap_or(0.0),
            out::s(&reply.kvad["cached_tokens"]),
        );
        if let Some(id) = kept {
            let stats = json!({
                "model": model["repo"],
                "backend": reply.kvad["backend"],
                "prompt_tokens": reply.usage["prompt_tokens"],
                "cached_tokens": reply.kvad["cached_tokens"],
                "generated_tokens": reply.usage["completion_tokens"],
                "prefill_secs": reply.kvad["prefill_secs"],
                "decode_secs": reply.kvad["decode_secs"],
            });
            remote.post(
                &format!("/api/conversations/{id}/messages"),
                &json!({ "role": "assistant", "content": reply.text, "stats": stats }),
            )?;
        }
        history.push(json!({ "role": "assistant", "content": reply.text }));
    }
    if let Some(id) = kept {
        eprintln!("kept as conversation {id}:  kvad conversations show {id}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

/// `kvad train`, as a job on the server.
///
/// The text has to be a dataset the server has. `--dataset` names one;
/// `--data FILE` uploads the file as one first, which is what makes the same
/// command line work in both places.
pub fn train(remote: &Remote, args: &Args) -> Res<()> {
    if args.words.first().map(String::as_str) == Some("options") {
        return options(remote, args);
    }
    // Four knobs the local trainer has and the server's runs do not take.
    // Refused rather than dropped: a run trained differently from what the
    // command line said is worse than one that did not start.
    for (flag, given) in [
        ("--batch", args.batch.is_some()),
        ("--warmup", args.warmup.is_some()),
        ("--decay-to", args.decay_to.is_some()),
        ("--clip", args.clip.is_some()),
    ] {
        if given {
            return Err(format!(
                "{flag} is not something a run on the server can be asked for.\n  \
                 Drop it, or train in this process with --local."
            )
            .into());
        }
    }

    let dataset = match (&args.dataset, &args.data) {
        (Some(d), _) => super::api::dataset_id(remote, d)?,
        (None, Some(file)) => upload(remote, file)?,
        (None, None) => {
            eprintln!("usage: kvad train --data FILE --name NAME");
            eprintln!("       kvad train --dataset ID|NAME --name NAME     (a dataset the server has)");
            std::process::exit(2);
        }
    };
    if args.name.is_none() && args.from.is_none() {
        eprintln!("a new model needs a name:  kvad train ... --name NAME");
        std::process::exit(2);
    }

    let mut body = json!({ "dataset": dataset, "seed": args.seed });
    for (key, value) in [
        ("name", args.name.clone().map(Value::from)),
        ("from", args.from.clone().map(Value::from)),
        ("size", args.size.clone().map(Value::from)),
        ("steps", args.steps.map(Value::from)),
        ("lr", args.lr.map(Value::from)),
        ("eval_every", args.eval_every.map(Value::from)),
        ("threads", args.threads.map(Value::from)),
        ("sample", args.sample.map(Value::from)),
    ] {
        if let Some(v) = value {
            body[key] = v;
        }
    }
    let job = remote.post("/api/train", &body)?;
    super::api::follow(remote, &job, args)?;
    if !args.json {
        let name = args.name.clone().or_else(|| args.from.clone()).unwrap_or_default();
        println!("\nrun it with:  kvad run --model {name} --prompt \"...\"");
    }
    Ok(())
}

/// Upload a file as a dataset, named after the file.
fn upload(remote: &Remote, file: &str) -> Res<i64> {
    let text = std::fs::read_to_string(file).map_err(|e| format!("could not read {file}: {e}"))?;
    let name = std::path::Path::new(file)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "corpus".into());
    eprintln!("uploading {file} as dataset `{name}` ({})", kvad::hub::human_bytes(text.len() as u64));
    let made = remote.call("post", &format!("/api/datasets?name={}", super::enc(&name)), Some(Body::Text(text)))?;
    made["id"].as_i64().ok_or_else(|| "the server made a dataset and did not say which".into())
}

/// `kvad train options` — what a run can be asked for there.
fn options(remote: &Remote, args: &Args) -> Res<()> {
    let o = remote.get("/api/train/options")?;
    if args.json {
        out::json(&o);
        return Ok(());
    }
    let rows: Vec<Vec<String>> =
        out::items(&o["sizes"]).iter().map(|s| vec![out::s(&s["name"]), out::s(&s["shape"])]).collect();
    out::table(&["SIZE", "SHAPE"], &rows);
    let d = &o["defaults"];
    println!(
        "\ndefaults: {} steps, lr {}, a checkpoint every {}, {} threads, {} characters sampled, seed {}",
        out::s(&d["steps"]),
        out::s(&d["lr"]),
        out::s(&d["eval_every"]),
        out::s(&d["threads"]),
        out::s(&d["sample"]),
        out::s(&d["seed"]),
    );
    let continuable: Vec<String> = out::items(&o["continuable"]).iter().map(out::s).collect();
    if !continuable.is_empty() {
        println!("can be trained further (--from): {}", continuable.join(", "));
    }
    println!("{} cores", out::s(&o["cores"]));
    if o["training"] == true {
        println!("a run is going now, and a second would be refused until it ends");
    }
    Ok(())
}
