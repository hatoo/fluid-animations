use fluid_animations::{advect, diffuse, Ghost};
use glam::Vec2;
use ndarray::{prelude::*, Zip};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::zeros((N + 2, N + 2));
    let mut x = x0.clone();

    let s = {
        let mut s = x.clone();
        s[[N / 2, N / 2]] = 10000.0;
        s
    };

    let uv = Array::from_elem(x.dim(), Vec2::new(0.0, -1.0));

    let dt = 1.0 / 24.0;
    let diff = 0.015;
    let a = dt * diff * N as f32 * N as f32;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;

        Zip::from(&mut x0).and(&s).for_each(|x, &s| {
            *x += dt * s;
        });
        diffuse(&mut x, &x0, a);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);
        advect(&mut x, &x0, &uv, dt);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
