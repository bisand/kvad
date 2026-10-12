//! What a model cut into shards is held to, for every architecture that
//! can be cut: the answer of the model run whole, to the bit.

#![allow(dead_code)]

use kvad::cluster::session::PipelineSession;
use kvad::model::{CpuSession, Session, Shard, Spec, Transformer};
use kvad::quant::Precision;

fn bits(xs: &[f32]) -> Vec<u32> {
    xs.iter().map(|x| x.to_bits()).collect()
}

fn argmax(row: &[f32]) -> u32 {
    row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |(i, _)| i as u32)
}

/// Every way of cutting `n_layer` layers that is worth telling apart: not
/// at all, in two, in three, with the ends on a shard that has no layers —
/// a coordinator that only embeds and reads the head — and with the ends
/// on the last shard instead of the first.
pub fn cuts(n_layer: usize) -> Vec<Vec<Shard>> {
    let mut all = vec![Shard::cut(n_layer, 1), Shard::cut(n_layer, 2)];
    if n_layer >= 3 {
        all.push(Shard::cut(n_layer, 3));
    }
    all.push(vec![Shard { layers: 0..0, ends: true }, Shard { layers: 0..1, ends: false }, Shard { layers: 1..n_layer, ends: false }]);
    all.push(vec![Shard { layers: 0..1, ends: false }, Shard { layers: 1..n_layer, ends: true }]);
    all
}

/// The same tokens through both, and the same bits out of both: a prompt
/// read in several passes, tokens decoded one at a time, a rewind and a
/// different continuation, and every position's logits at once.
pub fn same_answers(whole: &mut dyn Session, cut: &mut dyn Session, what: &str) {
    let vocab = whole.spec().vocab_size as u32;
    let prompt: Vec<u32> = (0..21u32).map(|i| (i * 7 + 3) % vocab).collect();
    fn both(whole: &mut dyn Session, cut: &mut dyn Session, tokens: &[u32], what: &str, step: &str) -> Vec<f32> {
        let (a, b) = (whole.forward(tokens).unwrap(), cut.forward(tokens).unwrap());
        assert!(a.iter().all(|x| x.is_finite()), "{what}, {step}: not finite");
        assert_eq!(bits(&a), bits(&b), "{what}, {step}: the cut model answers differently");
        assert_eq!(whole.cached(), cut.cached(), "{what}, {step}: they hold different lengths");
        a
    }

    // Prefill: 21 tokens at 8 a pass is three passes, the last a short one.
    let mut logits = both(whole, cut, &prompt, what, "prefill");
    for i in 0..5 {
        logits = both(whole, cut, &[argmax(&logits)], what, &format!("decode {i}"));
    }

    // Back to the middle of the prompt, and on with something else.
    assert_eq!(whole.truncate(9).unwrap(), 9);
    assert_eq!(cut.truncate(9).unwrap(), 9, "{what}: every stage rewinds to where it was told");
    let other: Vec<u32> = (0..6u32).map(|i| (i * 11 + 1) % vocab).collect();
    logits = both(whole, cut, &other, what, "after a rewind");
    both(whole, cut, &[argmax(&logits)], what, "decode after a rewind");

    // Rewinding to more than is held changes nothing, and says what is held.
    assert_eq!(cut.truncate(999).unwrap(), whole.cached());

    // Every position's logits, which is the head over every row.
    assert_eq!(whole.truncate(0).unwrap(), 0);
    assert_eq!(cut.truncate(0).unwrap(), 0);
    let (a, b) = (whole.forward_all(&prompt[..19]).unwrap(), cut.forward_all(&prompt[..19]).unwrap());
    assert_eq!(a.len(), 19 * vocab as usize);
    assert_eq!(bits(&a), bits(&b), "{what}, every position: the cut model answers differently");
}

/// Hold every cut of a model to the model. `load` reads one shard from the
/// checkpoint and must leave nothing in it unaccounted for.
pub fn every_cut_is_the_model(spec: &Spec, precision: Precision, what: &str, load: &dyn Fn(&Shard) -> Box<dyn Transformer>) {
    let whole_shard = Shard::whole(spec);
    for cut in cuts(spec.n_layer) {
        let names: Vec<String> = cut.iter().map(Shard::to_string).collect();
        let what = format!("{what} {precision}, cut as [{}]", names.join("; "));
        let mut whole = CpuSession::new(load(&whole_shard), precision).reading(8);
        let shards: Vec<Box<dyn Transformer>> = cut.iter().map(load).collect();
        let (params, bytes) = shards.iter().fold((0, 0), |a, s| (a.0 + s.param_count(), a.1 + s.memory_bytes()));
        let mut split = PipelineSession::local(shards, precision).unwrap().reading(8);
        assert_eq!((split.param_count(), split.weight_bytes()), (params, bytes), "{what}: the pipeline holds what its shards hold");
        assert_eq!(split.param_count(), whole.param_count(), "{what}: and between them that is the model");
        same_answers(&mut whole, &mut split, &what);
    }
}
