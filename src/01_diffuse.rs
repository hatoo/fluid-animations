use fluid_animations::{lin_solve, Ghost};
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 64;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2).f());
    x0[[N / 2, N / 2]] = 500.0;
    let mut x = x0.clone();

    let dt = 1.0 / 24.0;
    let diff = 0.01;
    let a = dt * diff * N as f32 * N as f32;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        lin_solve(&mut x, &mut x0, a, 1.0 + 4.0 * a, Ghost::Both);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
