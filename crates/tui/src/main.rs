//! A terminal UI for browsing, downloading and talking to local models.
//!
//!     cargo run --release -p kvad-tui
//!     cargo run --release -p kvad-tui -- out/readme    # open with a model loading
//!
//! The argument is a repo id or a directory, such as one `nanograd`'s
//! `train_text --save` wrote.
//!
//! Three threads' worth of concerns, kept apart:
//!
//! * this loop reads keys and draws, and never blocks for longer than the
//!   poll timeout;
//! * the engine thread downloads, loads and generates;
//! * `rayon` fans each matmul across cores underneath the engine.

mod app;
mod gpu;
mod ui;

use app::App;
use kvad::service::Engine;
use ratatui::crossterm::event::{self, Event, KeyEventKind};
use ratatui::crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::crossterm::ExecutableCommand;
use ratatui::prelude::*;
use std::io::stdout;
use std::time::Duration;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn main() -> Res<()> {
    // A panic with the terminal in raw mode leaves the shell unusable and the
    // message invisible. Restore first, then let the default hook print.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = restore();
        default_hook(info);
    }));

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;
    terminal.clear()?;

    let result = run(&mut terminal);

    restore()?;
    result
}

fn restore() -> Res<()> {
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    Ok(())
}

fn run(terminal: &mut Terminal<CrosstermBackend<std::io::Stdout>>) -> Res<()> {
    // The GPU half of the loader is this crate's, because `kvad` cannot
    // reach the GPU crate; see `gpu`.
    let engine = Engine::spawn(gpu::loader());
    let mut app = App::new();
    if let Some(model) = std::env::args().nth(1) {
        // By absolute path if it is a directory: loading makes it the active
        // model, and that has to survive a change of working directory.
        app.load(kvad::weights::model_id(&model), &engine);
    }

    while !app.should_quit {
        // Drain whatever the engine has produced since the last frame. During
        // generation this is a burst of tokens; taking them all before
        // redrawing keeps the frame rate independent of the token rate.
        while let Ok(evt) = engine.rx.try_recv() {
            app.apply(evt);
        }

        terminal.draw(|f| ui::draw(f, &app))?;

        // A short timeout, not a blocking read: streamed tokens have to reach
        // the screen even when nobody is touching the keyboard.
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) if key.kind == KeyEventKind::Press => app.on_key(key, &engine),
                Event::Resize(_, _) => {}
                _ => {}
            }
        }
    }
    Ok(())
}
