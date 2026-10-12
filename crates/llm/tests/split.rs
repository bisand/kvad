//! A model cut into shards is the same model.
//!
//! This is the claim everything about a cluster rests on, and it is a claim
//! about bits, not about closeness: a layer split moves where a matmul runs
//! and nothing about what it computes, so the logits of a model cut in
//! three have to be the logits of the model whole, exactly. "Close" would
//! pass with a row attending over the wrong cache, a position counted from
//! the wrong shard, or a norm applied on both sides of a cut.
//!
//! No sockets here. Every stage is in this process; what is under test is
//! the cut.

#![cfg(all(feature = "cluster", feature = "arch-llama", feature = "arch-gpt2"))]

mod common;

use kvad::cluster::session::PipelineSession;
use kvad::cluster::stage::{LocalStage, Stage};
use common::{gpt2_model, llama, VOCAB};
use kvad::model::Shard;
use kvad::quant::Precision;
use std::sync::Arc;

const PRECISIONS: [Precision; 3] = [Precision::F32, Precision::Q8, Precision::Q4];

#[test]
fn a_llama_cut_into_shards_answers_to_the_bit() {
    for model_type in ["llama", "qwen3"] {
        for precision in PRECISIONS {
            let fixture = llama(model_type, &format!("{model_type}-{precision}"));
            common::every_cut_is_the_model(&fixture.spec, precision, model_type, &|shard| fixture.load(precision, shard));
        }
    }
}

#[test]
fn a_gpt2_cut_into_shards_answers_to_the_bit() {
    for precision in PRECISIONS {
        let fixture = gpt2_model(&format!("gpt2-{precision}"));
        common::every_cut_is_the_model(&fixture.spec, precision, "gpt2", &|shard| fixture.load(precision, shard));
    }
}

/// A shard holds what it was asked for and no more, and says so in what it
/// counts: the parameters of the pieces are the parameters of the model.
#[test]
fn a_shard_holds_its_own_layers_and_nothing_else() {
    let fixture = llama("llama", "holds");
    let whole = fixture.load(Precision::F32, &Shard::whole(&fixture.spec));
    let middle = fixture.load(Precision::F32, &Shard { layers: 1..3, ends: false });
    let ends = fixture.load(Precision::F32, &Shard { layers: 0..0, ends: true });
    let per_layer = middle.param_count() / 2;
    assert_eq!(whole.param_count(), ends.param_count() + 4 * per_layer);
    assert_eq!(ends.param_count(), 2 * VOCAB * 64 + 64, "the table, the head and the final norm");
    assert!(middle.memory_bytes() > 0 && middle.memory_bytes() < whole.memory_bytes());
}

#[test]
#[should_panic(expected = "which another shard holds")]
fn a_shard_without_the_ends_cannot_embed() {
    let fixture = llama("llama", "no-ends");
    fixture.load(Precision::F32, &Shard { layers: 1..3, ends: false }).embed(&[1, 2, 3], 0);
}

/// Shards that are not one model are refused before a token is run.
#[test]
fn shards_that_are_not_one_model_are_not_a_pipeline() {
    let fixture = llama("llama", "refused");
    let pipeline = |shards: &[Shard]| {
        let loaded = shards.iter().map(|s| fixture.load(Precision::F32, s)).collect();
        PipelineSession::local(loaded, Precision::F32).map(|_| ()).map_err(|e| e.to_string())
    };
    let part = |layers: std::ops::Range<usize>, ends: bool| Shard { layers, ends };

    assert!(pipeline(&[part(0..2, true), part(2..4, false)]).is_ok());
    // In any order: a pipeline follows the layers, not the list.
    assert!(pipeline(&[part(2..4, false), part(0..2, true)]).is_ok());

    let gap = pipeline(&[part(0..1, true), part(2..4, false)]).unwrap_err();
    assert!(gap.contains("layer 1 is nowhere"), "{gap}");
    let overlap = pipeline(&[part(0..3, true), part(2..4, false)]).unwrap_err();
    assert!(overlap.contains("layer 2 is twice"), "{overlap}");
    let short = pipeline(&[part(0..3, true)]).unwrap_err();
    assert!(short.contains("3 layers of 4"), "{short}");
    let headless = pipeline(&[part(0..2, false), part(2..4, false)]).unwrap_err();
    assert!(headless.contains("0 shards hold the embedding"), "{headless}");
    let two_heads = pipeline(&[part(0..2, true), part(2..4, true)]).unwrap_err();
    assert!(two_heads.contains("2 shards hold the embedding"), "{two_heads}");
    assert!(PipelineSession::local(Vec::new(), Precision::F32).is_err());
}

/// A stage is told the position of the rows it is handed, and refuses rows
/// from anywhere but the end of what it holds: a token lost or run twice
/// between two machines is an error at that token, not a wrong reply.
#[test]
fn a_stage_refuses_rows_it_is_not_up_to() {
    let fixture = llama("llama", "stage");
    let e = fixture.spec.n_embd;
    let shard = Shard { layers: 1..3, ends: false };
    let mut stage = LocalStage::new(Arc::from(fixture.load(Precision::F32, &shard)));
    assert_eq!(stage.shard(), shard);

    let rows = |m: usize| vec![0.125f32; m * e];
    assert_eq!(stage.run(7, 0, 3, rows(3)).unwrap().len(), 3 * e);
    // The same rows again, and rows from further on.
    let again = stage.run(7, 0, 3, rows(3)).unwrap_err().to_string();
    assert!(again.contains("handed position 0 of a conversation it holds 3 positions of"), "{again}");
    assert!(stage.run(7, 5, 1, rows(1)).is_err());
    // A refusal changes nothing: the next rows are still the next rows.
    assert!(stage.run(7, 3, 1, rows(1)).is_ok());

    // Another conversation starts from nothing, beside the first.
    assert!(stage.run(8, 0, 2, rows(2)).is_ok());
    assert!(stage.run(7, 4, 1, rows(1)).is_ok());

    // Rows that are not the shape they were said to be.
    let ragged = stage.run(8, 2, 2, rows(1)).unwrap_err().to_string();
    assert!(ragged.contains("2 rows of 64 were announced"), "{ragged}");

    assert_eq!(stage.truncate(7, 2).unwrap(), 2);
    assert!(stage.run(7, 2, 1, rows(1)).is_ok());
    assert_eq!(stage.truncate(99, 5).unwrap(), 0, "a conversation never seen holds nothing");
}
