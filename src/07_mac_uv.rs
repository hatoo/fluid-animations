use fluid_animations::{
    diffuse,
    v2::{advect, Mac},
    Float, Ghost,
};
use ndarray::prelude::*;
use noise::{NoiseFn, Perlin};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    let mut x = x0.clone();

    // x0[[N / 2, N / 2]] = 40000.0;

    let perlin = Perlin::new();
    let freq = 10.0;

    let u = Array::from_shape_fn((N + 3, N + 2), |(i, j)| {
        perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]) * 4.0
    });

    let v = Array::from_shape_fn((N + 2, N + 3), |(i, j)| {
        perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]) * 4.0
    });

    let mut mac = Mac::new(u, v);

    let dt = 1.0 / 24.0;
    let unit = 1.0 / N as Float;
    let diff = 0.025;
    let a = dt * diff * (1.0 / unit) * (1.0 / unit);

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;
        x0[[N / 2, N / 2]] += dt * 10000.0;

        mac.diffuse(a * 0.1);
        mac.project();
        mac.self_advect(dt / unit);
        mac.project();
        let uv = mac.create_uv();

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
