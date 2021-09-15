use glam::Vec2;
use ndarray::prelude::*;
use ndarray::Zip;

pub mod image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ghost {
    Both,
    Horizontal,
    Vetical,
}

pub fn lin_solve(x: &mut Array2<f32>, x0: &Array2<f32>, a: f32, c: f32) {
    assert_eq!(x.dim(), x0.dim());

    x.clone_from(x0);

    for _ in 0..150 {
        for i in 1..x.dim().0 - 1 {
            for j in 1..x.dim().1 - 1 {
                x[[i, j]] = (x0[[i, j]]
                    + a * (x[[i - 1, j]] + x[[i + 1, j]] + x[[i, j - 1]] + x[[i, j + 1]]))
                    / c;
            }
        }
    }
}

pub fn lin_solve_rayon(x: &mut Array2<f32>, x0: &Array2<f32>, a: f32, c: f32) {
    assert_eq!(x.dim(), x0.dim());

    let mut t = Array::zeros(x.dim());
    let x1 = &mut t;

    x.clone_from(x0);

    let h = x.dim().0 - 2;
    let w = x.dim().1 - 2;

    for _ in 0..300 {
        Zip::from(x1.slice_mut(s![1..h + 1, 1..w + 1]))
            .and(x0.slice(s![1..h + 1, 1..w + 1]))
            .and(x.slice(s![0..h, 1..w + 1]))
            .and(x.slice(s![2..h + 2, 1..w + 1]))
            .and(x.slice(s![1..h + 1, 0..w]))
            .and(x.slice(s![1..h + 1, 2..w + 2]))
            .par_for_each(|x1, &z, &b1, &b2, &b3, &b4| {
                *x1 = (z + a * (b1 + b2 + b3 + b4)) / c;
            });

        std::mem::swap(x1, x);
    }
}

pub fn diffuse(x: &mut Array2<f32>, x0: &Array2<f32>, a: f32) {
    lin_solve(x, x0, a, 1.0 + 4.0 * a);
}

pub fn advect(d: &mut Array2<f32>, d0: &Array2<f32>, uv: &Array2<Vec2>, dt: f32) {
    assert_eq!(d.dim(), d0.dim());
    assert_eq!(uv.dim(), d0.dim());

    let h = d.dim().0 - 2;
    let w = d.dim().1 - 2;

    let dt0 = dt * ((h * w) as f32).sqrt();

    for i in 1..h + 1 {
        for j in 1..w + 1 {
            let y = i as f32 - dt0 * uv[[i, j]].y;
            let x = j as f32 - dt0 * uv[[i, j]].x;

            let y = y.max(0.5).min(h as f32 + 0.5);
            let x = x.max(0.5).min(w as f32 + 0.5);

            let i0 = y as usize;
            let i1 = i0 + 1;

            let j0 = x as usize;
            let j1 = j0 + 1;

            let t1 = y - i0 as f32;
            let t0 = 1.0 - t1;

            let s1 = x - j0 as f32;
            let s0 = 1.0 - s1;

            d[[i, j]] = s0 * (t0 * d0[[i0, j0]] + t1 * d0[[i0, j1]])
                + s1 * (t0 * d0[[i1, j0]] + t1 * d0[[i1, j1]]);
        }
    }
}

impl Ghost {
    pub fn set_border(self, x: &mut Array2<f32>) {
        let h = x.dim().0 - 2;
        let w = x.dim().1 - 2;

        for i in 0..h + 2 {
            x[[i, 0]] = if self == Ghost::Vetical {
                -x[[i, 1]]
            } else {
                x[[i, 1]]
            };
            x[[i, w + 1]] = if self == Ghost::Vetical {
                -x[[i, w]]
            } else {
                x[[i, w]]
            };
        }

        for j in 0..w + 2 {
            x[[0, j]] = if self == Ghost::Horizontal {
                -x[[1, j]]
            } else {
                x[[1, j]]
            };
            x[[h + 1, j]] = if self == Ghost::Horizontal {
                -x[[h, j]]
            } else {
                x[[h, j]]
            };
        }

        x[[0, 0]] = 0.5 * (x[[1, 0]] + x[[1, 0]]);
        x[[0, w + 1]] = 0.5 * (x[[1, w + 1]] + x[[0, w]]);
        x[[h + 1, 0]] = 0.5 * (x[[h, 0]] + x[[h + 1, 1]]);
        x[[h + 1, w + 1]] = 0.5 * (x[[h, w + 1]] + x[[h + 1, w]]);
    }
}
