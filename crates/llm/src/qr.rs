//! QR codes for pairing links: in a terminal for `kvad remote pair`, and as
//! SVG for the web UI's *Pair a device*, which kvad-serve draws so that the
//! page needs no encoder of its own. Both use level L, the lowest error
//! correction: a pairing link is about 490 characters, and the code is read
//! off a screen, not a crumpled label.

use qrcode::types::QrError;
use qrcode::{Color, EcLevel, QrCode};

/// Modules of light margin round the code, as the standard asks.
const QUIET: usize = 4;

fn code(text: &str) -> Result<QrCode, QrError> {
    QrCode::with_error_correction_level(text.as_bytes(), EcLevel::L)
}

/// For a terminal: two rows of modules to a line, dark on light.
pub fn terminal(text: &str) -> Result<String, QrError> {
    use qrcode::render::unicode::Dense1x2;
    Ok(code(text)?.render::<Dense1x2>().dark_color(Dense1x2::Light).light_color(Dense1x2::Dark).quiet_zone(true).build())
}

/// As an SVG: one unit to a module, and each row's runs of dark modules a
/// rectangle in one path. Black on white whatever the page's theme, since
/// some readers will not take a light code on dark. It has a `viewBox` and
/// no size, so the page sizes it.
pub fn svg(text: &str) -> Result<String, QrError> {
    let code = code(text)?;
    let width = code.width();
    let span = width + 2 * QUIET;
    let colours = code.to_colors();
    let mut path = String::new();
    for (y, row) in colours.chunks(width).enumerate() {
        let mut x = 0;
        while x < width {
            let run = row[x..].iter().take_while(|c| **c == Color::Dark).count();
            if run > 0 {
                path += &format!("M{} {}h{run}v1h-{run}z", x + QUIET, y + QUIET);
            }
            x += run.max(1);
        }
    }
    Ok(format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {span} {span}\" shape-rendering=\"crispEdges\">\
         <rect width=\"{span}\" height=\"{span}\" fill=\"#fff\"/><path d=\"{path}\" fill=\"#000\"/></svg>"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_svg_draws_every_dark_module_and_nothing_of_the_text() {
        let text = "kvad://example?pair=0123456789abcdef";
        let svg = svg(text).unwrap();
        let code = code(text).unwrap();
        let dark = code.to_colors().iter().filter(|c| **c == Color::Dark).count();
        let path = svg.split(" d=\"").nth(1).unwrap().split('"').next().unwrap();
        let run = |r: &str| r.split_once('h').unwrap().1.split_once('v').unwrap().0.parse::<usize>().unwrap();
        let drawn: usize = path.split('M').skip(1).map(run).sum();
        assert_eq!(drawn, dark);
        let span = code.width() + 2 * QUIET;
        assert!(svg.contains(&format!("viewBox=\"0 0 {span} {span}\"")));
        assert!(!svg.contains("pair="), "the link itself is not in the drawing");
    }
}
