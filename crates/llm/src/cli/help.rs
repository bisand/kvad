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

use super::spec::{self, Command, Group};
use kvad::help::nearest;
#[cfg(test)]
use kvad::help::WIDTH;

/// Wide enough for `conversations` and two spaces.
const NAME: usize = 15;

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
    kvad::help::command("kvad", c)
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
    match spec::command(command) {
        Some(c) => kvad::help::bad_flag(&spec::PROGRAMS[0], c, problem, flag),
        None => {
            eprintln!("{problem}\n`kvad help` lists the commands.");
            std::process::exit(2);
        }
    }
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
