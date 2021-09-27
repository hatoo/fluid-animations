use cgmath::{vec2, InnerSpace};
use fluid_animations::{
    v2::{advect, gauss_filter, gauss_filter_amb, Mac},
    Float, Ghost,
};
use ndarray::{prelude::*, Zip};

fn main() -> anyhow::Result<()> {
    const N: usize = 400;
    const N_FRAME: usize = 64;

    let t_amb = 273.0;

    let mut fuel = Array::from_elem((N + 2, N + 2), 0.0);
    let mut t = Array::from_elem((N + 2, N + 2), t_amb);

    let mut uv_mac = Mac::zeros((N + 2, N + 2));

    let dt: Float = 1.0 / 24.0;
    let unit = 1.0 / N as Float;
    let k_t = 0.002;
    let k_fuel = 0.05;

    let t_sigma2 = 0.5 * k_t * dt;
    let fuel_sigma2 = 0.5 * k_fuel * dt;
    let uv_sigma2 = 0.5 * 0.1 * dt;

    let alpha = 0.1;

    let g = 9.8;

    let density_amb = 1.01e5 / (285.0 * t_amb);

    let z = 7.5;
    let t_ignite = 0.0; //t_amb + 30.0;
    let t_max = 2000.0;

    let c = 0.1;

    t[[N / 2, N / 2]] = 10000.0;
    fuel[[N / 2, N / 2]] = 100.0;

    let mut prev_density = Array::from_elem(fuel.dim(), density_amb);

    for f in 1..N_FRAME + 1 {
        // fluid_animations::image::save(f, &fuel)?;
        // fluid_animations::image::save(f, &((&t - t_amb) / t_amb))?;

        // dbg!(t[[N / 2, N / 2]]);
        // dbg!(t.sum() / (N as Float * N as Float));

        let mut cnt = 0;
        let d_fuel = Array::from_shape_fn(fuel.dim(), |(i, j)| {
            if t[[i, j]] > t_ignite {
                if fuel[[i, j]] > 0.0 {
                    cnt += 1;
                }

                (z * dt).min(fuel[[i, j]])
            } else {
                0.0
            }
        });

        dbg!(cnt);

        fluid_animations::image::save(
            f,
            &Array::from_shape_fn(fuel.dim(), |(i, j)| {
                (t[[i, j]] - t_amb) / t_max * (fuel[[i, j]])
            }),
        )?;

        // fluid_animations::image::save(f, &fuel)?;
        // fluid_animations::image::save(f, &d_fuel)?;
        // fluid_animations::image::save(f, &((&t - t_amb) / 2000.0))?;

        let density = Array::from_shape_fn(fuel.dim(), |(i, j)| {
            // (density_amb * (1.0 + alpha * s[[i, j]] - beta * (t[[i, j]] - t_amb))).max(0.05)
            // density0 * (1.0 + alpha * s[[i, j]]) * (1.0 + beta * (t[[i, j]] - t_amb))
            1.01e5 / (285.0 * t[[i, j]]) * (1.0 + alpha * fuel[[i, j]])
        });

        // let div = Array::from_shape_fn((N + 2, N + 2), |(i, j)| d_fuel[[i, j]] / dt / 3.0);
        let div = Array::from_shape_fn((N + 2, N + 2), |(i, j)| {
            -1.0 / dt * (density[[i, j]] - prev_density[[i, j]]) / density[[i, j]]
        });

        /*
        dbg!(density[[N / 2, N / 2]]);
        dbg!(t[[N / 2, N / 2]]);
        dbg!(s[[N / 2, N / 2]]);
        */
        // dbg!(div[[N / 2, N / 2]]);

        uv_mac.self_advect(dt / unit);
        uv_mac.buoyancy2(&density, density_amb, g * dt);
        uv_mac.gauss_filter(uv_sigma2, unit);
        // uv_mac.project();
        uv_mac.project_variable_density_div_control(dt, unit, &density, density_amb, &div);

        // uv_mac.v = uv_mac.v.map(|v| v.clamp(-0.4, 0.4));
        // uv_mac.u = uv_mac.u.map(|u| u.clamp(-0.4, 0.4));

        /*
        dbg!(uv_mac
            .v
            .iter()
            .fold(0.0 as Float, |a, &b| { a.max(b.abs()) }));
            */

        let mut uv = uv_mac.create_uv();

        /*
        uv = uv.map(|v| {
            let m = v.magnitude().min(4.0);
            v * m / v.magnitude()
        });
        */

        dbg!(uv.iter().fold(0.0 as Float, |a, &b| a.max(b.magnitude())));

        Zip::from(&mut fuel).and(&d_fuel).for_each(|a, &b| {
            *a -= b;
        });

        dbg!(fuel.iter().fold(0.0 as Float, |a, &b| a.max(b)));

        fuel[[N / 2, N / 2]] += dt * N as Float * N as Float * 0.25;

        // uv[[N / 2, N / 2]] = vec2(0.0, -8.0);

        // dbg!(uv[[N / 2, N / 2]]);

        Zip::from(&mut t)
            .and(&d_fuel)
            .and(&density)
            .for_each(|a, &b, &c| {
                *a += 100.0 / (z * dt) * b / c;
            });

        let mut s1 = Array::zeros(fuel.dim());
        advect(&mut s1, &fuel, &uv, dt / unit);
        Ghost::Both.set_border(&mut s1);
        let mut s2 = gauss_filter(&s1, fuel_sigma2, unit);
        Ghost::Both.set_border(&mut s2);
        std::mem::swap(&mut fuel, &mut s2);

        let mut t1 = Array::zeros(t.dim());
        advect(&mut t1, &t, &uv, dt / unit);
        Ghost::Both.set_border(&mut t1);
        let mut t2 = gauss_filter_amb(&t1, t_sigma2, unit, t_amb);
        Ghost::Both.set_border(&mut t2);
        std::mem::swap(&mut t, &mut t2);

        dbg!(t[[N / 2, N / 2]]);

        t = t.map(|x| {
            t_amb
                + (1.0 / (*x - t_amb).max(1e-2).powi(3) + 3.0 * c * dt / (t_max - t_amb).powi(4))
                    .powf(-1.0 / 3.0)
        });

        // dbg!(t[[N / 2, N / 2]]);

        prev_density = density;

        dbg!(t.iter().fold(0.0 as Float, |a, &b| a.max(b)));

        eprint!("\rframe: {} / {} done", f, N_FRAME);
    }

    Ok(())
}
