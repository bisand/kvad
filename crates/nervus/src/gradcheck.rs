//! Shared by the gradient-check tests.

/// How far apart two gradients are, as a fraction of their size.
///
/// The MLP check in `nn` compares a few weights one at a time. Attention and
/// the norms have many gradients that are legitimately tiny, where f32
/// rounding in the numerical estimate swamps the value, so compare whole
/// vectors instead: a wrong sign or a missing term moves this to ~1, rounding
/// noise does not.
pub fn relative_error(analytic: &[f32], numerical: &[f32]) -> f32 {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let diff: Vec<f32> = analytic.iter().zip(numerical).map(|(a, n)| a - n).collect();
    norm(&diff) / (norm(analytic) + norm(numerical)).max(1e-12)
}

use crate::matrix::Matrix;
use crate::nn::{softmax_cross_entropy, Layer, Param};
use crate::rng::Rng;

/// One gradient, measured two ways.
pub struct Checked {
    /// The parameter's name, or `"dx"` for the gradient handed below.
    pub name: String,
    pub rel: f32,
    /// The sizes of the two gradients. `rel` is a ratio, and a ratio of two
    /// zeros is meaningless: a gradient that is *supposed* to vanish has to be
    /// recognised by its size instead.
    pub analytic_norm: f32,
    pub numerical_norm: f32,
}

fn checked(name: String, analytic: &[f32], numerical: &[f32]) -> Checked {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt();
    Checked {
        name,
        rel: relative_error(analytic, numerical),
        analytic_norm: norm(analytic),
        numerical_norm: norm(numerical),
    }
}

/// Gradient-check every parameter of anything that has parameters and a loss.
///
/// Call it *after* a backward pass, so the gradients it reads are the ones to
/// be judged. `params` is fetched afresh around every nudge, because the
/// forward pass inside `loss` needs the model back in between.
pub fn check_params<M: ?Sized>(
    model: &mut M,
    params: impl for<'a> Fn(&'a mut M) -> Vec<Param<'a>>,
    loss: impl Fn(&mut M) -> f32,
    nudge: f32,
) -> Vec<Checked> {
    let analytic: Vec<(String, Vec<f32>)> =
        params(model).into_iter().map(|p| (p.name, p.grad.to_vec())).collect();

    let mut report = Vec::new();
    for (p, (name, grad)) in analytic.into_iter().enumerate() {
        let mut numerical = vec![0.0; grad.len()];
        for (i, slot) in numerical.iter_mut().enumerate() {
            params(model)[p].value[i] += nudge;
            let up = loss(model);
            params(model)[p].value[i] -= 2.0 * nudge;
            let down = loss(model);
            params(model)[p].value[i] += nudge;
            *slot = (up - down) / (2.0 * nudge);
        }
        report.push(checked(name, &grad, &numerical));
    }
    report
}

/// Gradient-check a whole layer: every parameter tensor it reports, and the
/// gradient it hands to the layer below, which comes last, as `"dx"`.
///
/// The loss is softmax cross-entropy on the layer's output, one target per row.
pub fn check_layer(layer: &mut dyn Layer, x: &Matrix, targets: &[usize], nudge: f32) -> Vec<Checked> {
    let loss = |layer: &mut dyn Layer, x: &Matrix| softmax_cross_entropy(&layer.forward(x), targets).0;
    let centred = |up: f32, down: f32| (up - down) / (2.0 * nudge);

    layer.zero_grad();
    let (_, dlogits) = softmax_cross_entropy(&layer.forward(x), targets);
    let dx = layer.backward(&dlogits);
    let mut report = check_params(layer, |l| l.params(), |l| loss(l, x), nudge);

    let mut x = x.clone();
    let mut numerical = vec![0.0; x.data.len()];
    for (i, slot) in numerical.iter_mut().enumerate() {
        x.data[i] += nudge;
        let up = loss(layer, &x);
        x.data[i] -= 2.0 * nudge;
        let down = loss(layer, &x);
        x.data[i] += nudge;
        *slot = centred(up, down);
    }
    report.push(checked("dx".to_string(), &dx.data, &numerical));
    report
}

/// Move every parameter off its initial value.
///
/// A fresh layer is the worst place to check a gradient: gammas are exactly 1
/// and biases exactly 0, and a backward pass that forgot to multiply by one,
/// or that mishandled the other, passes. See the norm tests.
pub fn scramble(params: Vec<Param<'_>>, rng: &mut Rng) {
    for p in params {
        p.value.iter_mut().for_each(|v| *v += 0.3 * rng.normal());
    }
}
