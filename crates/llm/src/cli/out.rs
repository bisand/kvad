//! Printing what a server said: tables, questions, and a progress line.

use serde_json::Value;
use std::io::{IsTerminal, Write};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

/// A JSON document, pretty, on stdout. What `--json` prints.
pub fn json(value: &Value) {
    println!("{}", serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()));
}

/// A field as a person reads it: strings bare, nothing as `-`.
pub fn s(value: &Value) -> String {
    match value {
        Value::Null => "-".into(),
        Value::String(s) => s.clone(),
        Value::Bool(true) => "yes".into(),
        Value::Bool(false) => "no".into(),
        other => other.to_string(),
    }
}

/// A byte count as `1.2 GB`, or `-`.
pub fn bytes(value: &Value) -> String {
    value.as_u64().map(kvad::hub::human_bytes).unwrap_or_else(|| "-".into())
}

/// The elements of an array field, or none.
pub fn items(value: &Value) -> &[Value] {
    value.as_array().map(Vec::as_slice).unwrap_or(&[])
}

/// Columns as wide as their widest cell, and the last one left ragged so a
/// long title does not push a row of spaces after it.
pub fn table(header: &[&str], rows: &[Vec<String>]) {
    let mut widths: Vec<usize> = header.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let line = |cells: Vec<&str>| {
        let last = cells.len().saturating_sub(1);
        let mut out = String::new();
        for (i, (cell, w)) in cells.iter().zip(&widths).enumerate() {
            if i == last {
                out.push_str(cell);
            } else {
                out.push_str(cell);
                out.push_str(&" ".repeat(w - cell.chars().count() + 2));
            }
        }
        println!("{}", out.trim_end());
    };
    line(header.to_vec());
    for row in rows {
        line(row.iter().map(String::as_str).collect());
    }
}

/// Cut to `n` characters, with an ellipsis when anything went.
pub fn cut(s: &str, n: usize) -> String {
    let s = s.replace('\n', " ");
    match s.chars().count() <= n {
        true => s,
        false => format!("{}…", s.chars().take(n.saturating_sub(1)).collect::<String>()),
    }
}

/// A yes-or-no question; no is the default. `yes` answers it unasked.
pub fn confirm(question: &str, yes: bool) -> Res<bool> {
    if yes {
        return Ok(true);
    }
    print!("{question} [y/N] ");
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
}

/// A line of text somebody types.
pub fn ask(question: &str) -> Res<String> {
    eprint!("{question}: ");
    std::io::stderr().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(answer.trim().to_string())
}

/// A password or a key: asked on the terminal with echo off.
///
/// From the terminal itself rather than stdin, so that `kvad auth login` in
/// a pipeline still asks the person. With no terminal at all — a script —
/// it is read from stdin, one line, which is how a script gives one.
pub fn secret(question: &str) -> Res<String> {
    #[cfg(unix)]
    if let Some(answer) = tty_secret(question)? {
        return Ok(answer);
    }
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    Ok(answer.trim_end_matches(['\n', '\r']).to_string())
}

#[cfg(unix)]
fn tty_secret(question: &str) -> Res<Option<String>> {
    use std::io::BufRead;
    use std::os::unix::io::AsRawFd;

    let Ok(tty) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/tty") else {
        return Ok(None);
    };
    let fd = tty.as_raw_fd();
    // SAFETY: a zeroed termios is a valid value to hand tcgetattr to fill.
    let mut term: libc::termios = unsafe { std::mem::zeroed() };
    // SAFETY: fd is an open terminal for as long as `tty` lives.
    if unsafe { libc::tcgetattr(fd, &mut term) } != 0 {
        return Ok(None);
    }
    let before = term;
    term.c_lflag &= !libc::ECHO;
    term.c_lflag |= libc::ECHONL;
    // SAFETY: as above; `term` is the terminal's own settings, less echo.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &term) };

    let mut writer = &tty;
    let _ = write!(writer, "{question}: ");
    let _ = writer.flush();
    let mut answer = String::new();
    let read = std::io::BufReader::new(&tty).read_line(&mut answer);
    // Echo back on before anything can go wrong, including the read.
    // SAFETY: restoring what tcgetattr returned.
    unsafe { libc::tcsetattr(fd, libc::TCSANOW, &before) };
    read?;
    Ok(Some(answer.trim_end_matches(['\n', '\r']).to_string()))
}

/// A line that rewrites itself on a terminal: a download's bytes, a count.
///
/// Off a terminal every update would be a line of its own in a log, so only
/// the last one is printed, when [`Progress::done`] is called.
pub struct Progress {
    tty: bool,
    last: Option<String>,
}

impl Progress {
    pub fn new() -> Self {
        Progress { tty: std::io::stderr().is_terminal(), last: None }
    }

    pub fn show(&mut self, line: String) {
        if self.tty {
            eprint!("\r\x1b[2K{line}");
            let _ = std::io::stderr().flush();
        }
        self.last = Some(line);
    }

    /// Leave the line as it stands, and start a new one.
    pub fn done(&mut self) {
        match (self.tty, self.last.take()) {
            (true, Some(_)) => eprintln!(),
            (false, Some(line)) => eprintln!("{line}"),
            (_, None) => {}
        }
    }
}

/// Whether stdout is a terminal, and so whether dim text means anything.
pub fn tty() -> bool {
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cut_counts_characters_and_says_when_it_cut() {
        assert_eq!(cut("short", 10), "short");
        assert_eq!(cut("æøå æøå æøå", 5), "æøå …");
        assert_eq!(cut("two\nlines", 20), "two lines");
    }

    #[test]
    fn a_field_reads_as_a_person_would_say_it() {
        assert_eq!(s(&Value::Null), "-");
        assert_eq!(s(&serde_json::json!("x")), "x");
        assert_eq!(s(&serde_json::json!(3)), "3");
        assert_eq!(s(&serde_json::json!(true)), "yes");
        assert_eq!(bytes(&serde_json::json!(2048)), kvad::hub::human_bytes(2048));
        assert_eq!(bytes(&Value::Null), "-");
    }
}
