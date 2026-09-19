//! Rendering. Pure: takes `&App`, draws a frame, changes nothing.

use crate::app::{App, Focus, Tab};
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph, Wrap};

const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;

pub fn draw(f: &mut Frame, app: &App) {
    let chunks = Layout::vertical([
        Constraint::Length(1), // tab bar
        Constraint::Min(5),    // body
        Constraint::Length(1), // status
        Constraint::Length(1), // help
    ])
    .split(f.area());

    draw_tabs(f, chunks[0], app);
    match app.tab {
        Tab::Models => draw_models(f, chunks[1], app),
        Tab::Chat => draw_chat(f, chunks[1], app),
    }
    draw_status(f, chunks[2], app);
    draw_help(f, chunks[3], app);

    if let Some(id) = &app.confirm_delete {
        draw_confirm(f, id);
    }
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let tab = |label: &str, active: bool| {
        if active {
            Span::styled(format!(" {label} "), Style::new().fg(Color::Black).bg(ACCENT).bold())
        } else {
            Span::styled(format!(" {label} "), Style::new().fg(MUTED))
        }
    };
    let right = match &app.active {
        Some(a) => format!(
            "{} · {:.0}M · {} {:.0}MB · {} ",
            a.repo,
            a.params as f64 / 1e6,
            a.backend,
            a.weight_bytes as f64 / 1e6,
            if a.instruct { "chat" } else { "completion" }
        ),
        None => format!("no model loaded · next: {} ", app.backend),
    };

    let line = Line::from(vec![
        tab("Models", app.tab == Tab::Models),
        Span::raw(" "),
        tab("Chat", app.tab == Tab::Chat),
    ]);
    f.render_widget(Paragraph::new(line), area);
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(right, Style::new().fg(MUTED))))
            .alignment(Alignment::Right),
        area,
    );
}

fn draw_models(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([Constraint::Length(3), Constraint::Min(3)]).split(area);

    let searching = app.focus == Focus::Search;
    let search = Paragraph::new(Line::from(vec![
        Span::styled(&app.query, Style::new().fg(Color::White)),
        Span::styled(if searching { "▌" } else { "" }, Style::new().fg(ACCENT)),
    ]))
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(Style::new().fg(if searching { ACCENT } else { MUTED }))
            .title(" search the Hub "),
    );
    f.render_widget(search, rows[0]);

    let entries = app.entries();
    let title = if app.showing_results {
        format!(" Hub results ({}) ", entries.len())
    } else {
        format!(" downloaded ({}) ", entries.len())
    };

    if entries.is_empty() {
        let hint = if app.showing_results {
            "no matches"
        } else {
            "nothing downloaded yet — press / and search for e.g. `smollm`"
        };
        f.render_widget(
            Paragraph::new(hint)
                .style(Style::new().fg(MUTED))
                .block(Block::default().borders(Borders::ALL).title(title)),
            rows[1],
        );
        return;
    }

    let items: Vec<ListItem> = entries
        .iter()
        .map(|e| {
            let mark = if e.local { "●" } else { "○" };
            let name_style = if e.runnable {
                Style::new().fg(Color::White)
            } else {
                Style::new().fg(MUTED).add_modifier(Modifier::DIM)
            };
            ListItem::new(Line::from(vec![
                Span::styled(format!(" {mark} "), Style::new().fg(if e.local { Color::Green } else { MUTED })),
                Span::styled(format!("{:<44}", truncate(&e.id, 44)), name_style),
                Span::styled(
                    format!("{:<7}", e.arch.map(|a| a.to_string()).unwrap_or_else(|| "-".into())),
                    Style::new().fg(ACCENT),
                ),
                Span::styled(e.detail.clone(), Style::new().fg(MUTED)),
            ]))
        })
        .collect();

    let mut state = ListState::default();
    state.select(Some(app.selected));
    f.render_stateful_widget(
        List::new(items)
            .block(Block::default().borders(Borders::ALL).title(title))
            .highlight_style(Style::new().bg(Color::Rgb(40, 44, 52)).bold()),
        rows[1],
        &mut state,
    );
}

fn draw_chat(f: &mut Frame, area: Rect, app: &App) {
    let rows = Layout::vertical([Constraint::Min(3), Constraint::Length(3)]).split(area);

    let mut lines: Vec<Line> = Vec::new();
    if app.messages.is_empty() && app.streaming.is_none() {
        if let Some(a) = &app.active {
            lines.push(Line::from(Span::styled(
                format!("{}  ", a.summary),
                Style::new().fg(MUTED),
            )));
            lines.push(Line::raw(""));
        }
        lines.push(Line::from(Span::styled(
            match &app.active {
                Some(a) if a.instruct => "Ask it something.".to_string(),
                Some(a) => format!(
                    "{} is a base model: it continues text rather than answering. \
                     Type the start of a sentence.",
                    a.repo
                ),
                None => "No model loaded. Press Tab, choose one, press Enter.".to_string(),
            },
            Style::new().fg(MUTED).italic(),
        )));
    }

    for m in &app.messages {
        let (label, colour) = match m.role.as_str() {
            "user" => ("you", Color::Green),
            "assistant" => ("ai ", ACCENT),
            other => (other, MUTED),
        };
        lines.push(Line::from(Span::styled(
            format!("{label} ▸"),
            Style::new().fg(colour).bold(),
        )));
        for l in m.content.lines() {
            lines.push(Line::from(Span::raw(format!("  {l}"))));
        }
        lines.push(Line::raw(""));
    }

    if let Some(partial) = &app.streaming {
        lines.push(Line::from(Span::styled("ai ▸", Style::new().fg(ACCENT).bold())));
        for l in partial.lines() {
            lines.push(Line::from(Span::raw(format!("  {l}"))));
        }
        if app.busy {
            lines.push(Line::from(Span::styled("  ▌", Style::new().fg(ACCENT))));
        }
    }

    // Keep the newest text in view. Wrapping means the true rendered height is
    // not known here, so approximate: it is only ever a scroll position.
    let height = rows[0].height.saturating_sub(2) as usize;
    let offset = lines.len().saturating_sub(height) as u16;
    let scroll = if app.scroll == u16::MAX { offset } else { app.scroll.min(offset) };

    f.render_widget(
        Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .scroll((scroll, 0))
            .block(Block::default().borders(Borders::ALL).title(" conversation ")),
        rows[0],
    );

    let prompt_style = if app.busy { Style::new().fg(MUTED) } else { Style::new().fg(Color::White) };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(&app.input, prompt_style),
            Span::styled(if app.busy { "" } else { "▌" }, Style::new().fg(ACCENT)),
        ]))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(if app.busy { MUTED } else { ACCENT }))
                .title(if app.busy { " generating — esc to stop " } else { " message " }),
        ),
        rows[1],
    );
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let mut spans = vec![];
    if let Some(err) = &app.error {
        spans.push(Span::styled(format!(" error: {err}"), Style::new().fg(Color::Red)));
    } else {
        if app.busy {
            spans.push(Span::styled(" ● ", Style::new().fg(Color::Yellow)));
        }
        spans.push(Span::styled(format!(" {}", app.status), Style::new().fg(Color::Gray)));
        if let Some(s) = &app.last_stats {
            let reused = if s.cached_tokens > 0 {
                format!(" · {} cached", s.cached_tokens)
            } else {
                String::new()
            };
            spans.push(Span::styled(
                format!("  [{:.1} tok/s{reused}]", s.tokens_per_sec()),
                Style::new().fg(MUTED),
            ));
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_help(f: &mut Frame, area: Rect, app: &App) {
    let keys: &[(&str, &str)] = match app.tab {
        Tab::Models if app.focus == Focus::Search => {
            &[("enter", "search"), ("esc", "back"), ("tab", "chat")]
        }
        Tab::Models => &[
            ("/", "search"),
            ("↑↓", "select"),
            ("enter", "load"),
            ("d", "delete"),
            ("u", "unload"),
            ("p", "backend"),
            ("l", "local"),
            ("tab", "chat"),
            ("q", "quit"),
        ],
        Tab::Chat => &[
            ("enter", "send"),
            ("esc", "stop"),
            ("^l", "clear"),
            ("pgup/pgdn", "scroll"),
            ("tab", "models"),
            ("^c", "quit"),
        ],
    };
    let mut spans = Vec::new();
    for (k, label) in keys {
        spans.push(Span::styled(format!(" {k}"), Style::new().fg(ACCENT)));
        spans.push(Span::styled(format!(" {label}"), Style::new().fg(MUTED)));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_confirm(f: &mut Frame, id: &str) {
    let area = centered(60, 7, f.area());
    f.render_widget(Clear, area);
    f.render_widget(
        Paragraph::new(vec![
            Line::raw(""),
            Line::from(Span::raw(format!("  Delete {id}?"))),
            Line::from(Span::styled(
                "  The files are removed from the HuggingFace cache.",
                Style::new().fg(MUTED),
            )),
            Line::raw(""),
            Line::from(vec![
                Span::styled("  y", Style::new().fg(Color::Red).bold()),
                Span::raw(" delete    "),
                Span::styled("any other key", Style::new().fg(ACCENT)),
                Span::raw(" cancel"),
            ]),
        ])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::new().fg(Color::Red))
                .title(" confirm "),
        ),
        area,
    );
}

fn centered(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width);
    let h = height.min(area.height);
    Rect {
        x: area.x + (area.width - w) / 2,
        y: area.y + (area.height - h) / 2,
        width: w,
        height: h,
    }
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_string()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::Active;
    use kvad::chat::Message;
    use ratatui::backend::TestBackend;

    /// Render one frame and return it as text, so assertions can be written
    /// against what a user would actually see.
    fn render(app: &App, width: u16, height: u16) -> String {
        let mut terminal = Terminal::new(TestBackend::new(width, height)).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        let buffer = terminal.backend().buffer().clone();
        (0..buffer.area.height)
            .map(|y| {
                (0..buffer.area.width)
                    .map(|x| buffer[(x, y)].symbol().to_string())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn empty_models_tab_explains_itself() {
        let app = App::new();
        let out = render(&app, 100, 24);
        assert!(out.contains("Models"), "{out}");
        assert!(out.contains("nothing downloaded yet"), "{out}");
        assert!(out.contains("no model loaded"), "{out}");
    }

    #[test]
    fn chat_shows_history_and_the_streaming_reply() {
        let mut app = App::new();
        app.tab = Tab::Chat;
        app.active = Some(Active {
            repo: "HuggingFaceTB/SmolLM2-135M-Instruct".into(),
            summary: "llama · 30 layers".into(),
            params: 134_515_008,
            instruct: true,
            backend: "cpu q8".into(),
            weight_bytes: 151_329_384,
        });
        app.messages.push(Message::user("why is the sky blue?"));
        app.streaming = Some("Because shorter wavelengths".into());
        app.busy = true;

        let out = render(&app, 100, 24);
        assert!(out.contains("why is the sky blue?"), "{out}");
        assert!(out.contains("Because shorter wavelengths"), "{out}");
        // While generating, the input box says so rather than inviting typing.
        assert!(out.contains("esc to stop"), "{out}");
        assert!(out.contains("135M"), "{out}"); // 134.5M, rounded
    }

    #[test]
    fn base_model_warns_that_it_will_not_answer() {
        let mut app = App::new();
        app.tab = Tab::Chat;
        app.active = Some(Active {
            repo: "openai-community/gpt2".into(),
            summary: "gpt2 · 12 layers".into(),
            params: 124_000_000,
            instruct: false,
            backend: "cpu f32".into(),
            weight_bytes: 496_000_000,
        });
        let out = render(&app, 100, 24);
        assert!(out.contains("continues text"), "{out}");
    }

    #[test]
    fn delete_confirmation_names_the_model() {
        let mut app = App::new();
        app.confirm_delete = Some("openai-community/gpt2".into());
        let out = render(&app, 100, 24);
        assert!(out.contains("Delete openai-community/gpt2?"), "{out}");
        assert!(out.contains("y delete"), "{out}");
    }

    /// Narrow terminals must not panic: every Layout here has to stay solvable.
    #[test]
    fn survives_a_tiny_terminal() {
        let app = App::new();
        render(&app, 20, 8);
        render(&app, 8, 4);
    }
}
