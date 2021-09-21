use fluid_animations::{
    v2::{advect, gauss_filter, gauss_filter_amb, Mac},
    Float, Ghost,
};
use ndarray::prelude::*;
// use noise::{NoiseFn, Perlin};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let mut s = Array::zeros((N + 2, N + 2));
    let mut t = Array::from_elem((N + 2, N + 2), 273.0);

    /*
    let perlin = Perlin::new();
    let freq = 10.0;

    let u = Array::from_shape_fn((N + 3, N + 2), |(i, j)| {
        perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.5]) * 4.0
    });

    let v = Array::from_shape_fn((N + 2, N + 3), |(i, j)| {
        perlin.get([i as f64 / N as f64 * freq, j as f64 / N as f64 * freq, 0.0]) * 4.0
    });

    let mut uv_mac = Mac::new(u, v);
    */

    let mut uv_mac = Mac::zeros((N + 2, N + 2));

    let dt: Float = 1.0 / 24.0;
    let unit = 1.0 / N as Float;
    let k_t = 0.1;
    let k_s = 0.1;
    let r_t = 1.0;

    let t_sigma2 = 0.5 * k_t * dt;
    let s_sigma2 = 0.5 * k_s * dt;
    let uv_sigma2 = 0.5 * 0.1 * dt;

    let alpha = 0.025;
    let beta = 0.025;

    let t_amb = 273.0;

    let g = 9.8;

    for f in 1..N_FRAME + 1 {
        fluid_animations::image::save(f, &s)?;
        // fluid_animations::image::save(f, &((&t - t_amb) / t_amb))?;

        // dbg!(t[[N / 2, N / 2]]);
        // dbg!(t.sum() / (N as Float * N as Float));

        uv_mac.self_advect(dt / unit);
        uv_mac.project();
        uv_mac.gauss_filter(uv_sigma2, unit);
        uv_mac.project();
        uv_mac.buoyancy(&s, &t, alpha, beta, t_amb, g * dt);
        uv_mac.project();

        let uv = uv_mac.create_uv();

        t[[N / 2, N / 2]] +=
            (1.0 - (-r_t * dt).exp()) * ((50.0) * N as Float * N as Float - t[[N / 2, N / 2]]);

        s[[N / 2, N / 2]] += dt * N as Float * N as Float * 0.5;

        let mut s1 = gauss_filter(&s, s_sigma2, unit);
        Ghost::Both.set_border(&mut s1);
        let mut s2 = Array::zeros(s1.dim());
        advect(&mut s2, &s1, &uv, dt / unit);
        Ghost::Both.set_border(&mut s2);
        std::mem::swap(&mut s, &mut s2);

        let mut t1 = gauss_filter_amb(&t, t_sigma2, unit, t_amb);
        Ghost::Both.set_border(&mut t1);
        let mut t2 = Array::zeros(t1.dim());
        advect(&mut t2, &t1, &uv, dt / unit);
        Ghost::Both.set_border(&mut t2);
        std::mem::swap(&mut t, &mut t2);

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
