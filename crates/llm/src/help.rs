//! Printing a command line's help from its table, for all four binaries.
//!
//! Two sizes. A program with commands has an overview, every command in a
//! line each, and each command has a page of its own: its usage, its options
//! with what each does, and examples. A program with no commands has only
//! the page.
//!
//! Help that was asked for goes to stdout and exits 0, so that it can be
//! piped to a pager. A mistake gets two or three lines on stderr and exits
//! 2: what was wrong, the nearest thing that would have been right, and
//! where the help is. Not the help itself, which pushes the one line that
//! matters off the top of the terminal.

use crate::spec::{self, Command, Flag, Program};

/// Wide enough for `--lora NAME[:SCALE]`.
pub const FLAG: usize = 22;

/// What a line of options is kept within.
pub const WIDTH: usize = 80;

/// What a command is called on its command line: `kvad-gpu tune`, or
/// `kvad-serve` for a program that is its own only command.
fn called(program: &str, c: &Command) -> String {
    match c.name == program {
        true => program.to_string(),
        false => format!("{program} {}", c.name),
    }
}

/// One command's page.
pub fn command(program: &str, c: &Command) -> String {
    let name = called(program, c);
    let mut out = format!("{name} — {}\n", c.about);
    if c.server {
        out.push_str("asks a running server; `kvad service start` starts this machine's\n");
    }
    out.push('\n');
    match c.usage {
        Some(text) => out.push_str(text),
        None => out.push_str(&synopsis(&name, c)),
    }
    out.push('\n');
    if !c.notes.is_empty() {
        out.push_str(&format!("\n{}\n", c.notes));
    }

    // Subcommands, where the usage above is one this module wrote: a
    // hand-written one has already said them.
    if c.usage.is_none() && !c.subs.is_empty() {
        out.push_str("\ncommands:\n");
        for s in c.subs {
            out.push_str(&format!("  {:<FLAG$}{}\n", format!("{} {}", s.name, s.args).trim_end(), s.about));
        }
    }

    let own: Vec<&Flag> = c.flags.iter().flat_map(|g| g.iter()).collect();
    if !own.is_empty() {
        out.push_str("\noptions:\n");
        own.iter().for_each(|f| out.push_str(&flag_line(f)));
    }
    // Each subcommand's, under its own name — but a group of flags two of
    // them share is said once, for the first, and pointed at by the rest.
    let mut said: Vec<(&[Flag], &str)> = Vec::new();
    for s in c.subs.iter().filter(|s| !s.flags.is_empty()) {
        out.push_str(&format!("\noptions of `{name} {}`:\n", s.name));
        for group in s.flags {
            match said.iter().find(|(seen, _)| same(seen, group)) {
                Some((_, first)) => out.push_str(&format!("  and every option of `{first}`\n")),
                None => {
                    group.iter().for_each(|f| out.push_str(&flag_line(f)));
                    said.push((group, s.name));
                }
            }
        }
    }

    if c.routed {
        out.push_str("\nwhere it runs:\n");
        spec::WHERE.iter().for_each(|f| out.push_str(&flag_line(f)));
        out.push_str(
            "  With neither --remote nor --local: $KVAD_URL, then [client] url in kvad.toml,\n  \
             then this machine's service if it answers, then this process.\n",
        );
    }

    if !c.examples.is_empty() {
        out.push_str("\nexamples:\n");
        c.examples.iter().for_each(|e| out.push_str(&format!("  {e}\n")));
    }
    out
}

/// A program's first page: its commands in a line each, or, for a program
/// that has none, the one page there is.
pub fn program(p: &Program) -> String {
    if p.commands.is_empty() {
        let mut out = command(p.name, p.bare);
        out.push_str(&format!("\n{} {}\n", p.name, env!("CARGO_PKG_VERSION")));
        return out;
    }
    let width = p.commands.iter().map(|c| c.name.len()).max().unwrap_or(0) + 2;
    let mut out = format!(
        "{name} {version} — {about}\n\n\
         usage: {name} <command> [options]\n       \
         {name} help <command>      what one command takes, with examples\n\n\
         commands:\n",
        name = p.name,
        version = env!("CARGO_PKG_VERSION"),
        about = p.about,
    );
    for c in p.commands {
        out.push_str(&format!("  {:<width$}{}\n", c.name, c.about));
    }
    out.push_str(&format!("\nWith no command, options are `{} {}`'s.\n-V, --version prints the version.\n", p.name, p.bare.name));
    out
}

/// What `help`, `-h` and `--help` print, and stop: the page of the command
/// named, or the program's first page.
pub fn show(p: &Program, command_name: Option<&str>) -> ! {
    match command_name {
        None => print!("{}", program(p)),
        Some(name) => match p.command(name) {
            Some(c) => print!("{}", command(p.name, c)),
            None => unknown_command(p, name),
        },
    }
    std::process::exit(0);
}

/// A command that is not one.
pub fn unknown_command(p: &Program, name: &str) -> ! {
    eprintln!("`{} {name}` is not a command.", p.name);
    if let Some(near) = nearest(name, p.commands.iter().map(|c| c.name)) {
        eprintln!("Did you mean `{} {near}`?", p.name);
    }
    eprintln!("`{} help` lists them.", p.name);
    std::process::exit(2);
}

/// A flag that is not one, or one with nothing after it: `problem`, the
/// nearest flag `c` does take if `flag` is none it knows, and where the
/// help is.
pub fn bad_flag(p: &Program, c: &Command, problem: &str, flag: &str) -> ! {
    eprintln!("{problem}");
    if !flag.is_empty() && p.takes_value(flag).is_none() {
        let flags = c.all_flags();
        let routed = spec::WHERE.iter().filter(|_| c.routed);
        if let Some(near) = nearest(flag, flags.iter().copied().chain(routed).map(|f| f.name)) {
            eprintln!("Did you mean {near}?");
        }
    }
    eprintln!("`{} --help` says what it takes.", called(p.name, c));
    std::process::exit(2);
}

/// The usage line of a command that has no hand-written one.
fn synopsis(name: &str, c: &Command) -> String {
    let options = if c.flags.is_empty() && !c.routed { "" } else { " [options]" };
    match c.args.is_empty() {
        true => format!("usage: {name}{options}"),
        false => format!("usage: {name} {}{options}", c.args),
    }
}

/// Whether two groups of flags are one group, named twice. By what is in
/// them: a `const` has no address to compare.
fn same(a: &[Flag], b: &[Flag]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| a.name == b.name && a.about == b.about)
}

fn flag_line(f: &Flag) -> String {
    let mut left = match f.short.is_empty() {
        true => f.name.to_string(),
        false => format!("{}, {}", f.short, f.name),
    };
    if f.takes_value() {
        left.push(' ');
        left.push_str(f.value);
    }
    // A left side too long for its column takes a line to itself, so the
    // descriptions still start in one place; and a description too long for
    // the line carries on under its own start, not under the flag.
    let mut lines = wrap(f.about, WIDTH - FLAG - 2).into_iter();
    let first = lines.next().unwrap_or_default();
    let mut out = match left.chars().count() < FLAG {
        true => format!("  {left:<FLAG$}{first}\n"),
        false => format!("  {left}\n  {:<FLAG$}{first}\n", ""),
    };
    lines.for_each(|line| out.push_str(&format!("  {:<FLAG$}{line}\n", "")));
    out
}

/// `text` in lines of at most `width` characters, broken at spaces.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for word in text.split(' ') {
        match lines.last_mut() {
            Some(line) if line.chars().count() + 1 + word.chars().count() <= width => {
                line.push(' ');
                line.push_str(word);
            }
            _ => lines.push(word.to_string()),
        }
    }
    lines
}

/// The candidate a typo most likely meant, if any is close.
///
/// Close is an edit distance of at most a third of the word, and at least
/// one: `chta` is `chat`, `pul` is `pull`, and `frobnicate` is nothing.
pub fn nearest<'a>(word: &str, candidates: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let allowed = (word.chars().count() / 3).max(1);
    candidates
        .map(|c| (distance(word, c), c))
        .filter(|(d, c)| *d <= allowed || (word.len() > 2 && c.starts_with(word)))
        // Of two as near, the one that was being typed: `tun` is a letter
        // from `run` and from `tune`, and is the start of one of them.
        .min_by_key(|(d, c)| (*d, !c.starts_with(word)))
        .map(|(_, c)| c)
}

/// Edit distance, with a swap of two neighbours counted as one edit: the
/// commonest typo there is should not cost two.
fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut rows = vec![vec![0usize; b.len() + 1]; a.len() + 1];
    for (i, row) in rows.iter_mut().enumerate() {
        row[0] = i;
    }
    for j in 0..=b.len() {
        rows[0][j] = j;
    }
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            let change = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (rows[i - 1][j] + 1).min(rows[i][j - 1] + 1).min(rows[i - 1][j - 1] + change);
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(rows[i - 2][j - 2] + 1);
            }
            rows[i][j] = best;
        }
    }
    rows[a.len()][b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every page of the three small programs says every flag it takes, in
    /// lines that fit a terminal, and gives an example.
    #[test]
    fn the_small_programs_pages_are_whole() {
        for p in [&spec::SERVE, &spec::GPU, &spec::TUI] {
            let first = program(p);
            for line in first.lines() {
                assert!(line.chars().count() <= WIDTH, "{}: {line}", p.name);
            }
            let pages: Vec<&Command> = if p.commands.is_empty() { vec![p.bare] } else { p.commands.iter().collect() };
            for c in pages {
                let text = command(p.name, c);
                assert!(first.contains(c.name), "{}'s first page does not name `{}`", p.name, c.name);
                assert!(!c.about.is_empty() && !c.examples.is_empty(), "{} {} wants a line and an example", p.name, c.name);
                for f in c.all_flags() {
                    assert!(text.contains(f.name), "{} {} does not mention {}", p.name, c.name, f.name);
                }
                for line in text.lines().filter(|l| !c.examples.iter().any(|e| l.contains(e))) {
                    assert!(line.chars().count() <= WIDTH, "{} {}: {line}", p.name, c.name);
                }
            }
        }
    }

    #[test]
    fn a_typo_is_told_what_it_was_near() {
        let names = || spec::GPU.commands.iter().map(|c| c.name);
        assert_eq!(nearest("chta", names()), Some("chat"));
        assert_eq!(nearest("tun", names()), Some("tune"));
        assert_eq!(nearest("frobnicate", names()), None);
    }
}
