use ndarray::prelude::*;

pub mod image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ghost {
    Both,
    Horizontal,
    Vetical,
}

pub fn lin_solve(x: &mut Array2<f32>, x0: &mut Array2<f32>, a: f32, c: f32, g: Ghost) {
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
    g.set_border(x);
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
