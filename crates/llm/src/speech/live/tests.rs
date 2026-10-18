//! A live session driven over the scripted stand-in for a Whisper in
//! [`super::script`]: what is settled, and where a window is cut, checked
//! against what the script says, with no model downloaded.

use super::script::{script_text, sound, Script};
use super::*;

/// Sound pushed a second at a time, an update after each: what is settled
/// is the script, in order, never taken back, and the window is cut where
/// a segment ends rather than forced.
#[test]
fn settled_text_is_the_script_and_never_changes() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    let (mut settled, mut cuts) = (String::new(), 0);
    let mut shown_first = 0;
    for second in 1..=40 {
        live.push(&sound(1.0));
        let u = live.update(&tr).unwrap();
        assert!(!u.forced, "forced at {second} s");
        cuts += u.cut as usize;
        settled.push_str(&u.settled);
        let now = format!("{settled}{}", u.partial);
        assert!(script_text(40).starts_with(&settled), "settled `{settled}` at {second} s");
        // The partial is the script too, a word or two ahead.
        assert!(script_text(40).starts_with(&now), "shown `{now}` at {second} s");
        if second == 1 {
            shown_first = u.partial.len();
        }
        assert_eq!(u.heard, second as f32);
    }
    assert!(shown_first > 0, "the first word showed as a partial before it settled");
    assert!(cuts >= 1, "a 40 s session is cut at least once");
    settled.push_str(&live.finish(&tr).unwrap().settled);
    assert_eq!(settled, script_text(40));
}

/// A word whose first half is heard, and then all of it, is not settled
/// on the half the two guesses share: whole words are compared.
#[test]
fn half_a_word_is_not_agreed_with_all_of_it() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    // Word 3 is two tokens; at 3.7 s its first is written.
    live.push(&sound(3.7));
    let a = live.update(&tr).unwrap();
    assert_eq!((a.settled.as_str(), a.partial.as_str()), ("", " w0 w1 w2 w3"));
    live.push(&sound(0.6));
    let b = live.update(&tr).unwrap();
    assert_eq!((b.settled.as_str(), b.partial.as_str()), (" w0 w1 w2", " w3x"));
    let c = live.update(&tr).unwrap();
    assert_eq!((c.settled.as_str(), c.partial.as_str()), (" w3x", ""));
}

/// Silence settles nothing, and is not kept: the window never fills with
/// it, and speech after it settles as before.
#[test]
fn silence_is_let_go_of() {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    for _ in 0..45 {
        live.push(&vec![0.0; SAMPLE_RATE]);
        let u = live.update(&tr).unwrap();
        assert!(u.silent && u.settled.is_empty() && !u.forced);
    }
    assert!(live.window.len() <= QUIET_KEPT);
    assert_eq!(live.finish(&tr).unwrap().settled, "", "finishing on silence settles nothing");
    live.push(&sound(2.0));
    live.update(&tr).unwrap();
    assert_eq!(live.finish(&tr).unwrap().settled, " w0 w1");
    assert_eq!(live.received(), 47.0);
}

/// What a session settles of 40 s of the script, pushed in `pieces` (in
/// seconds) with an update after each, and finished.
fn settled_from(pieces: &[f32]) -> String {
    let tr = Script::transcriber();
    let mut live = Live::new(&tr, &Request::default()).unwrap();
    let mut settled = String::new();
    for &s in pieces {
        live.push(&sound(s));
        let u = live.update(&tr).unwrap();
        assert!(!u.forced, "forced after {} s", live.received());
        settled.push_str(&u.settled);
    }
    settled.push_str(&live.finish(&tr).unwrap().settled);
    assert_eq!(live.received(), pieces.iter().sum::<f32>());
    settled
}

/// A link cut mid-sentence, as the server sees it: updates a second apart
/// to 12.5 s, then nothing for a while, then the rest of the sentence all
/// at once as the phone resends it, then a second at a time again. The
/// session is the one it held, and what it settles is what a session never
/// cut settles.
#[test]
fn a_session_resumed_after_a_gap_settles_as_one_never_cut() {
    let never_cut = settled_from(&[1.0; 40]);
    assert_eq!(never_cut, script_text(40));
    let mut cut = vec![1.0; 12];
    cut.push(0.5);
    cut.push(4.5);
    cut.extend([1.0; 23]);
    assert_eq!(settled_from(&cut), never_cut);
}

/// A decoder that keeps up with one update in three: the same audio, in
/// pieces three times the size, settles the same text.
#[test]
fn fewer_updates_settle_the_same_text() {
    let mut slow = vec![3.0; 13];
    slow.push(1.0);
    assert_eq!(settled_from(&slow), script_text(40));
}
