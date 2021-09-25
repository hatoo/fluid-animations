use fluid_animations::{diffuse, Float, Ghost};
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    x0[[N / 2, N / 2]] = 500.0;
    let mut x = x0.clone();

    let dt = 1.0 / 24.0;
    let diff = 0.0005;
    let a = dt * diff * N as Float * N as Float;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        diffuse(&mut x, &x0, a);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
