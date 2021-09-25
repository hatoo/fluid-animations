use fluid_animations::{
    v2::{advect, Mac},
    Float, Ghost,
};
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut x0 = Array::from_elem((N + 2, N + 2), 0.1);

    let mut mac = Mac::zeros((N + 2, N + 2));

    let dt = 1.0 / 24.0;
    let unit = 1.0 / N as Float;

    let density = Array::from_elem((N + 2, N + 2), 1.0);

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &x0)?;
        x0[[N / 2, N / 2]] += dt * 100.0;

        let mut div = Array::zeros((N + 2, N + 2));

        div[[N / 2, N / 2]] = 40.0;

        // mac.gauss_filter(sigma2, unit);
        mac.self_advect(dt / unit);
        mac.project_variable_density_div_control(dt, unit, &density, &div);

        let uv = mac.create_uv();

        // let mut x = gauss_filter(&x0, s_sigma2, unit);
        let mut x = x0.clone();
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);
        advect(&mut x, &x0, &uv, dt / unit);
        Ghost::Both.set_border(&mut x);
        std::mem::swap(&mut x, &mut x0);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
