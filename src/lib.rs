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
    assert_eq!(x.shape(), x0.shape());

    for _ in 0..20 {
        for i in 1..x.shape()[0] - 1 {
            for j in 1..x.shape()[1] - 1 {
                x[[i, j]] = (x0[[i, j]]
                    + a * (x[[i - 1, j]] + x[[i + 1, j]] + x[[i, j - 1]] + x[[i, j + 1]]))
                    / c;
            }
        }
    }
}

pub fn lin_solve_rayon(x: &mut Array2<f32>, x0: &Array2<f32>, a: f32, c: f32) {
    assert_eq!(x.shape(), x0.shape());

    let mut t = Array::zeros(x.dim());
    let x1 = &mut t;

    let h = x.shape()[0] - 2;
    let w = x.shape()[1] - 2;

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

impl Ghost {
    pub fn set_border(self, x: &mut Array2<f32>) {
        let h = x.shape()[0] - 2;
        let w = x.shape()[1] - 2;

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
