use fluid_animations::{advect, diffuse, Ghost};
use glam::Vec2;
use ndarray::{prelude::*, Zip};
use noise::{NoiseFn, Perlin, Seedable};
use rayon::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 120;

    let create_uv = |dim, seed| {
        let perlin = Perlin::new().set_seed(seed);
        let freq = 10.0;
        Array::from_shape_fn(dim, |(i, j)| {
            let dx = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]);
            let dy = perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]);
            Vec2::new(dx as f32, dy as f32) * 8.0
        })
    };

    let x0 = Array::zeros((N + 2, N + 2));
    let mut s = x0.clone();
    s[[N / 2, N / 2]] = 100000.0;

    let mut rgb = [
        (x0.clone(), x0.clone(), create_uv(x0.dim(), 1)),
        (x0.clone(), x0.clone(), create_uv(x0.dim(), 2)),
        (x0.clone(), x0.clone(), create_uv(x0.dim(), 3)),
    ];

    let dt = 1.0 / 24.0;
    let diff = 0.05;
    let a = dt * diff * N as f32 * N as f32;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save_rgb(f, &rgb[0].1, &rgb[1].1, &rgb[2].1)?;

        rgb.par_iter_mut().for_each(|(x, x0, uv)| {
            Zip::from(&mut *x0).and(&s).for_each(|x, &s| {
                *x += dt * s;
            });

            diffuse(x, x0, a);
            Ghost::Both.set_border(x);
            std::mem::swap(x, x0);
            advect(x, x0, uv, dt);
            Ghost::Both.set_border(x);
            std::mem::swap(x, x0);
        });

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
