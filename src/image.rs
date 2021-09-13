use image::{Rgb, RgbImage};
use ndarray::prelude::*;

pub fn save(index: usize, x: &Array2<f32>) -> anyhow::Result<()> {
    let shape = x.shape();

    let mut img = RgbImage::new(shape[1] as u32, shape[0] as u32);

    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let l = (x[[i, j]] * 256.0).max(0.0).min(255.0) as u8;
            img.put_pixel(j as u32, i as u32, Rgb([l, l, l]));
        }
    }

    img.save(format!("out/{:06}.png", index))?;

    Ok(())
}
