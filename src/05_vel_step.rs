use cgmath::vec2;
use fluid_animations::{dens_step, vel_step};
use ndarray::{prelude::*, Zip};
use noise::{NoiseFn, Perlin};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    let mut x = x0.clone();

    let dx = {
        let mut x = x0.clone();
        x[[N / 2, N / 2]] = 7500.0;
        x
    };

    let perlin = Perlin::new();
    let freq = 8.0;
    let duv = Array::from_shape_fn(x.dim(), |(i, j)| {
        let dx = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]);
        let dy = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]);
        vec2(dx as f32, dy as f32) * 4.0
    });

    let mut uv0 = duv;
    let mut uv = Array2::zeros(x.dim());

    let mut p = Array2::zeros(x.dim());
    let mut div = Array2::zeros(x.dim());

    let dt = 1.0 / 24.0;
    let diff = 0.025;

    let visc = 0.005;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        Zip::from(&mut x0).and(&dx).for_each(|a, &b| {
            *a += dt * b;
        });

        vel_step(&mut uv, &mut uv0, &mut p, &mut div, visc, dt);
        std::mem::swap(&mut uv, &mut uv0);

        dens_step(&mut x, &mut x0, &mut uv0, diff, dt);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
