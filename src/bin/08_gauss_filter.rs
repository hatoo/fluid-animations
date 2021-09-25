use fluid_animations::{v2::gauss_filter, Float, Ghost};
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    x0[[N / 2, N / 2]] = 10000.0;

    let dt = 1.0 / 24.0;
    let unit = 1.0 / N as Float;
    let k_t = 0.1;
    let sigma2 = 0.5 * k_t * dt;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        let mut x = gauss_filter(&mut x0, sigma2, unit);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
