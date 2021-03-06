use cgmath::vec2;
use fluid_animations::{diffuse, v2::advect, Float, Ghost};
use ndarray::prelude::*;
use noise::{NoiseFn, Perlin};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    let mut x = x0.clone();

    x0[[N / 2, N / 2]] = 100000.0;

    let perlin = Perlin::new();
    let freq = 10.0;
    let uv = Array::from_shape_fn(x.dim(), |(i, j)| {
        let dy = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]);
        let dx = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]);
        vec2(dx as Float, dy as Float) * 4.0
    });

    let dt = 1.0 / 24.0;
    let unit = 1.0 / N as Float;
    let diff = 0.025;
    let a = dt * diff * (1.0 / unit) * (1.0 / unit);

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        diffuse(&mut x, &x0, a);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);
        advect(&mut x, &x0, &uv, dt / unit);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
