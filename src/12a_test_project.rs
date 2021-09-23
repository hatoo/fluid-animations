use fluid_animations::{
    v2::{advect, gauss_filter, gauss_filter_amb, Mac},
    Float, Ghost,
};
use ndarray::prelude::*;

fn main() -> anyhow::Result<()> {
    const N: usize = 3;

    let u = array![
        [0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ];

    let v = array![
        [0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ];

    let mut uv_mac = Mac::new(u, v);
    let mut uv_mac = Mac::zeros((N, N));

    dbg!(uv_mac.create_uv());

    let dt: Float = 1.0 / 24.0;
    let unit = 1.0 / N as Float;

    let density = Array::from_elem((N, N), 1.0);

    let mut div = Array::zeros((N, N));

    div[[N / 2, N / 2]] = 10.0;

    dbg!(uv_mac.div());
    dbg!(uv_mac.create_uv());

    // uv_mac.project_variable_density_div_control(dt, unit, &density, &div);
    uv_mac.project_variable_density(dt, unit, &density);
    // uv_mac.project();
    /*
    uv_mac.self_advect(dt / unit);
    uv_mac.gauss_filter(uv_sigma2, unit);
    uv_mac.project_variable_density_div_control(dt, unit, &density, &div);
    */

    dbg!(uv_mac.div());

    let uv = uv_mac.create_uv();

    dbg!(&uv);

    Ok(())
}
