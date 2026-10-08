//! `kvad help`, `kvad COMMAND --help`, and what a mistyped command is told.
//!
//! Printed from [`super::spec`], so the help cannot name a flag the
//! completion does not offer, or the other way about.
//!
//! Two sizes. The overview is every command in a line each, grouped by what
//! somebody is trying to do, and short enough to read: the page it replaced
//! listed every option of every command, which answered "what does `train`
//! take" and buried "what can this do". A command's own help is where its
//! options are.
//!
//! Help that was asked for goes to stdout and exits 0, so that it can be
//! piped to a pager. A mistake gets two lines on stderr and exits 2: what was
//! wrong, the nearest thing that would have been right, and where the help
//! is. Not the help itself, which pushes the one line that matters off the
//! top of the terminal.

use super::spec::{self, Command, Flag, Group};

/// Wide enough for `conversations` and two spaces.
const NAME: usize = 15;

/// Wide enough for `--lora NAME[:SCALE]`.
const FLAG: usize = 22;

/// What a line of options is kept within.
const WIDTH: usize = 80;

/// `kvad help`.
pub fn overview() -> String {
    let mut out = format!(
        "kvad {} — run, train and serve language, image and video models\n\n\
         usage: kvad <command> [words] [options]\n       \
         kvad help <command>      what one command takes, with examples\n",
        env!("CARGO_PKG_VERSION")
    );
    for group in Group::ALL {
        out.push_str(&format!("\n{}\n", group.title()));
        for c in spec::COMMANDS.iter().filter(|c| c.group == group) {
            let mark = if c.server { "*" } else { " " };
            out.push_str(&format!(" {mark}{:<NAME$}{}\n", c.name, c.about));
        }
    }
    out.push_str(&format!(
        "\n* asks a running server: `kvad service start` starts this machine's, and\n  \
         `--remote URL` names one somewhere else. The other commands go to a server\n  \
         too when one is running, and run in this process when none is, or with\n  \
         `--local`. The first line each prints says which, and why.\n\n\
         start here\n  \
         kvad pull {model}\n  \
         kvad chat\n\n\
         -V, --version prints the version. `kvad completions` sets up Tab completion.\n",
        model = crate::DEFAULT_MODEL,
    ));
    out
}

/// `kvad help COMMAND`.
pub fn command(c: &Command) -> String {
    let mut out = format!("kvad {} — {}\n", c.name, c.about);
    if c.server {
        out.push_str("asks a running server; `kvad service start` starts this machine's\n");
    }
    out.push('\n');
    match c.usage {
        Some(text) => out.push_str(text),
        None => out.push_str(&synopsis(c)),
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
        out.push_str(&format!("\noptions of `kvad {} {}`:\n", c.name, s.name));
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

/// The usage line of a command that has no hand-written one.
fn synopsis(c: &Command) -> String {
    let options = if c.flags.is_empty() && !c.routed { "" } else { " [options]" };
    match (c.subs.is_empty(), c.args.is_empty()) {
        (_, true) => format!("usage: kvad {}{options}", c.name),
        (true, false) | (false, false) => format!("usage: kvad {} {}{options}", c.name, c.args),
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

/// Print a command's help, or the overview, and stop. What `--help` does.
pub fn show(name: Option<&str>) -> ! {
    match name {
        None => print!("{}", overview()),
        Some(name) => match spec::command(name) {
            Some(c) => print!("{}", command(c)),
            None => unknown_command(name),
        },
    }
    std::process::exit(0);
}

/// A command that is not one.
pub fn unknown_command(name: &str) -> ! {
    eprintln!("`kvad {name}` is not a command.");
    if let Some(near) = nearest(name, spec::COMMANDS.iter().map(|c| c.name)) {
        eprintln!("Did you mean `kvad {near}`?");
    }
    eprintln!("`kvad help` lists them.");
    std::process::exit(2);
}

/// A flag that is not one, or one with nothing after it.
pub fn bad_flag(command: &str, problem: &str, flag: &str) -> ! {
    eprintln!("{problem}");
    let known = spec::command(command);
    if let Some(c) = known.filter(|_| spec::takes_value(flag).is_none()) {
        let flags = c.all_flags();
        if let Some(near) = nearest(flag, flags.iter().copied().chain(spec::WHERE).map(|f| f.name)) {
            eprintln!("Did you mean {near}?");
        }
    }
    match known {
        Some(c) => eprintln!("`kvad {} --help` says what it takes.", c.name),
        None => eprintln!("`kvad help` lists the commands."),
    }
    std::process::exit(2);
}

/// The candidate a typo most likely meant, if any is close.
///
/// Close is an edit distance of at most a third of the word, and at least
/// one: `chta` is `chat`, `pul` is `pull`, and `frobnicate` is nothing.
fn nearest<'a>(word: &str, candidates: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let allowed = (word.chars().count() / 3).max(1);
    candidates
        .map(|c| (distance(word, c), c))
        .filter(|(d, c)| *d <= allowed || (word.len() > 2 && c.starts_with(word)))
        .min_by_key(|(d, _)| *d)
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

    #[test]
    fn a_typo_is_told_what_it_was_near() {
        let names = || spec::COMMANDS.iter().map(|c| c.name);
        assert_eq!(nearest("chta", names()), Some("chat"));
        assert_eq!(nearest("pul", names()), Some("pull"));
        assert_eq!(nearest("image", names()), Some("images"));
        assert_eq!(nearest("conv", names()), Some("conversations"));
        assert_eq!(nearest("frobnicate", names()), None);
    }

    /// The overview fits a terminal, and names every command.
    #[test]
    fn the_overview_is_narrow_and_complete() {
        let text = overview();
        for line in text.lines() {
            assert!(line.chars().count() <= 80, "wider than a terminal: {line}");
        }
        for c in spec::COMMANDS {
            assert!(text.contains(&format!("{:<NAME$}", c.name)), "`{}` is not in the overview", c.name);
            assert!(c.name.len() < NAME, "`{}` does not fit its column", c.name);
        }
    }

    /// Every command's help says every flag it takes, in lines that fit.
    #[test]
    fn a_commands_help_says_all_its_options() {
        for c in spec::COMMANDS {
            let text = command(c);
            for f in c.all_flags() {
                assert!(text.contains(f.name), "`kvad help {}` does not mention {}", c.name, f.name);
            }
            // The lines this module wrote. A usage written by hand is as
            // wide as whoever wrote it chose.
            let by_hand = c.usage.unwrap_or("");
            for line in text.lines().filter(|l| !by_hand.contains(l) && !c.examples.iter().any(|e| l.contains(e))) {
                assert!(line.chars().count() <= WIDTH, "`kvad help {}` has a line too long to read: {line}", c.name);
            }
        }
    }
}
