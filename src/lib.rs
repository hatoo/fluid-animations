use std::iter::Sum;
use std::ops::Add;
use std::ops::Div;
use std::ops::Mul;
use std::ops::Rem;
use std::ops::Sub;

use cgmath::vec2;
use cgmath::Vector2;
use ndarray::prelude::*;
use ndarray::Zip;

pub mod image;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ghost {
    Both,
    Horizontal,
    Vetical,
}

pub trait Vector: Copy + Clone + Sync + Send
where
    Self: num_traits::Zero,
    Self: Add<Self, Output = Self>,
    Self: Sub<Self, Output = Self>,
    Self: Sum<Self>,
    Self: Mul<f32, Output = Self>,
    Self: Div<f32, Output = Self>,
    Self: Rem<f32, Output = Self>,
{
}

impl Vector for f32 {}
impl Vector for Vector2<f32> {}

pub fn lin_solve<V: Vector>(x: &mut Array2<V>, x0: &Array2<V>, a: f32, c: f32) {
    assert_eq!(x.dim(), x0.dim());

    x.clone_from(x0);

    for _ in 0..150 {
        for i in 1..x.dim().0 - 1 {
            for j in 1..x.dim().1 - 1 {
                x[[i, j]] = (x0[[i, j]]
                    + (x[[i - 1, j]] + x[[i + 1, j]] + x[[i, j - 1]] + x[[i, j + 1]]) * a)
                    / c;
            }
        }
    }
}

pub fn lin_solve_rayon<V: Vector>(x: &mut Array2<V>, x0: &Array2<V>, a: f32, c: f32) {
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
                *x1 = (z + (b1 + b2 + b3 + b4) * a) / c;
            });

        std::mem::swap(x1, x);
    }
}

pub fn diffuse<V: Vector>(x: &mut Array2<V>, x0: &Array2<V>, a: f32) {
    lin_solve(x, x0, a, 1.0 + 4.0 * a);
}

pub fn advect<V: Vector>(d: &mut Array2<V>, d0: &Array2<V>, uv: &Array2<Vector2<f32>>, dt: f32) {
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

            d[[i, j]] = (d0[[i0, j0]] * t0 + d0[[i0, j1]] * t1) * s0
                + (d0[[i1, j0]] * t0 + d0[[i1, j1]] * t1) * s1;
        }
    }
}

pub fn project(uv: &mut Array2<Vector2<f32>>, p: &mut Array2<f32>, div: &mut Array2<f32>) {
    assert_eq!(uv.dim(), p.dim());
    assert_eq!(uv.dim(), div.dim());

    let h = uv.dim().0 - 2;
    let w = uv.dim().1 - 2;

    let n = ((h * w) as f32).sqrt();

    for i in 1..h + 1 {
        for j in 1..w + 1 {
            div[[i, j]] = -0.5
                * (uv[[i + 1, j]].y - uv[[i - 1, j]].y + uv[[i, j + 1]].x - uv[[i, j - 1]].x)
                / n;
            p[[i, j]] = 0.0;
        }
    }

    Ghost::Both.set_border(div);
    Ghost::Both.set_border(p);

    lin_solve(p, div, 1.0, 4.0);
    Ghost::Both.set_border(p);

    for i in 1..h + 1 {
        for j in 1..w + 1 {
            uv[[i, j]] -=
                0.5 * n * vec2(p[[i, j + 1]] - p[[i, j - 1]], p[[i + 1, j]] - p[[i - 1, j]]);
        }
    }
}

pub fn dens_step(
    x: &mut Array2<f32>,
    x0: &mut Array2<f32>,
    uv: &Array2<Vector2<f32>>,
    diff: f32,
    dt: f32,
) {
    assert_eq!(x.dim(), x0.dim());
    assert_eq!(x.dim(), uv.dim());
    let a = (x.dim().0 * x.dim().1) as f32 * dt * diff;
    diffuse(x, x0, a);
    Ghost::Both.set_border(x);
    std::mem::swap(x, x0);
    advect(x, x0, uv, dt);
    Ghost::Both.set_border(x);
}

pub fn vel_step(
    uv: &mut Array2<Vector2<f32>>,
    uv0: &mut Array2<Vector2<f32>>,
    p: &mut Array2<f32>,
    div: &mut Array2<f32>,
    visc: f32,
    dt: f32,
) {
    assert_eq!(uv.dim(), uv0.dim());
    assert_eq!(uv.dim(), p.dim());
    assert_eq!(uv.dim(), div.dim());

    let a = (uv.dim().0 * uv.dim().1) as f32 * dt * visc;
    diffuse(uv, uv0, a);
    set_border_v2(uv);
    std::mem::swap(uv, uv0);
    project(uv0, p, div);
    set_border_v2(uv0);
    advect(uv, uv0, uv0, dt);
    set_border_v2(uv);
    project(uv, p, div);
    set_border_v2(uv);
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

        x[[0, 0]] = 0.5 * (x[[1, 0]] + x[[0, 1]]);
        x[[0, w + 1]] = 0.5 * (x[[1, w + 1]] + x[[0, w]]);
        x[[h + 1, 0]] = 0.5 * (x[[h, 0]] + x[[h + 1, 1]]);
        x[[h + 1, w + 1]] = 0.5 * (x[[h, w + 1]] + x[[h + 1, w]]);
    }

    fn set_border_v2_elem(self, x: &mut Array2<Vector2<f32>>, e: usize) {
        let h = x.dim().0 - 2;
        let w = x.dim().1 - 2;

        for i in 0..h + 2 {
            x[[i, 0]][e] = if self == Ghost::Vetical {
                -x[[i, 1]][e]
            } else {
                x[[i, 1]][e]
            };
            x[[i, w + 1]][e] = if self == Ghost::Vetical {
                -x[[i, w]][e]
            } else {
                x[[i, w]][e]
            };
        }

        for j in 0..w + 2 {
            x[[0, j]][e] = if self == Ghost::Horizontal {
                -x[[1, j]][e]
            } else {
                x[[1, j]][e]
            };
            x[[h + 1, j]][e] = if self == Ghost::Horizontal {
                -x[[h, j]][e]
            } else {
                x[[h, j]][e]
            };
        }

        x[[0, 0]][e] = 0.5 * (x[[1, 0]][e] + x[[0, 1]][e]);
        x[[0, w + 1]][e] = 0.5 * (x[[1, w + 1]][e] + x[[0, w]][e]);
        x[[h + 1, 0]][e] = 0.5 * (x[[h, 0]][e] + x[[h + 1, 1]][e]);
        x[[h + 1, w + 1]][e] = 0.5 * (x[[h, w + 1]][e] + x[[h + 1, w]][e]);
    }
}

pub fn set_border_v2(uv: &mut Array2<Vector2<f32>>) {
    Ghost::Horizontal.set_border_v2_elem(uv, 0);
    Ghost::Vetical.set_border_v2_elem(uv, 1);
}
