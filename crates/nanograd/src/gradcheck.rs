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
