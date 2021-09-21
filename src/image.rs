use image::{Rgb, RgbImage};
use ndarray::prelude::*;

use crate::Float;

pub fn save(index: usize, x: &Array2<Float>) -> anyhow::Result<()> {
    let shape = x.dim();

    let mut img = RgbImage::new(shape.0 as u32, shape.1 as u32);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let l = (x[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            img.put_pixel(i as u32, j as u32, Rgb([l, l, l]));
        }
    }

    img.save(format!("out/{:06}.png", index))?;

    Ok(())
}

pub fn save_rgb(
    index: usize,
    r: &Array2<Float>,
    g: &Array2<Float>,
    b: &Array2<Float>,
) -> anyhow::Result<()> {
    assert_eq!(r.dim(), g.dim());
    assert_eq!(g.dim(), b.dim());

    let shape = r.dim();

    let mut img = RgbImage::new(shape.1 as u32, shape.0 as u32);

    for i in 0..shape.0 {
        for j in 0..shape.1 {
            let rv = (r[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            let gv = (g[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            let bv = (b[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            img.put_pixel(j as u32, i as u32, Rgb([rv, gv, bv]));
        }
    }

    img.save(format!("out/{:06}.png", index))?;

    Ok(())
}
