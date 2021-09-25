use num_traits::FloatConst;

use cgmath::{vec2, Vector2};
use ndarray::{Array, Array2, Zip};

use crate::{lin_solve, linear, Float, Vector};

#[derive(Clone)]
pub struct Mac {
    pub u: Array2<Float>,
    pub v: Array2<Float>,
}

impl Mac {
    pub fn new(u: Array2<Float>, v: Array2<Float>) -> Self {
        let u_dim = u.dim();
        let v_dim = v.dim();

        assert_eq!((u_dim.0 - 1, u_dim.1), (v_dim.0, v_dim.1 - 1));

        Self { u, v }
    }

    pub fn zeros((w, h): (usize, usize)) -> Self {
        Self {
            u: Array::zeros((w + 1, h)),
            v: Array::zeros((w, h + 1)),
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.v.dim().0, self.u.dim().1)
    }

    pub fn create_uv(&self) -> Array2<Vector2<Float>> {
        Array::from_shape_fn(self.dim(), |(i, j)| {
            let u = 0.5 * (self.u[[i, j]] + self.u[[i + 1, j]]);
            let v = 0.5 * (self.v[[i, j]] + self.v[[i, j + 1]]);

            vec2(u, v)
        })
    }

    pub fn add(&mut self, other: &Mac, dt: Float) {
        assert_eq!(self.dim(), other.dim());

        Zip::from(&mut self.u).and(&other.u).for_each(|a, &b| {
            *a += dt * b;
        });

        Zip::from(&mut self.v).and(&other.v).for_each(|a, &b| {
            *a += dt * b;
        });
    }

    pub fn self_advect(&mut self, weight: Float) {
        let u_uv = Array::from_shape_fn(self.u.dim(), |(i, j)| {
            let u = self.u[[i, j]];
            let v = if i == 0 {
                0.5 * (self.v[[i, j]] + self.v[[i, j + 1]])
            } else if i == self.u.dim().0 - 1 {
                0.5 * (self.v[[i - 1, j]] + self.v[[i - 1, j + 1]])
            } else {
                0.25 * (self.v[[i, j]]
                    + self.v[[i, j + 1]]
                    + self.v[[i - 1, j]]
                    + self.v[[i - 1, j + 1]])
            };
            vec2(u, v)
        });

        let v_uv = Array::from_shape_fn(self.v.dim(), |(i, j)| {
            let v = self.v[[i, j]];
            let u = if j == 0 {
                0.5 * (self.u[[i, j]] + self.u[[i + 1, j]])
            } else if j == self.v.dim().1 - 1 {
                0.5 * (self.u[[i, j - 1]] + self.u[[i + 1, j - 1]])
            } else {
                0.25 * (self.u[[i, j]]
                    + self.u[[i, j - 1]]
                    + self.u[[i + 1, j]]
                    + self.u[[i + 1, j - 1]])
            };
            vec2(u, v)
        });

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        advect(&mut self.u, &u0, &u_uv, weight);
        advect(&mut self.v, &v0, &v_uv, weight);
    }

    pub fn project(&mut self) {
        let div = Array::from_shape_fn(self.dim(), |(i, j)| {
            0.5 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]])
        });

        let mut p = Array::zeros(div.dim());

        // lin_solve(&mut p, &div, 1.0, 4.0);
        linear::lin_solve(&mut p, &div, -1.0, 4.0);

        let (w, h) = self.dim();
        for i in 1..w {
            for j in 0..h {
                self.u[[i, j]] += 2.0 * (p[[i, j]] - p[[i - 1, j]]);
            }
        }

        for i in 0..w {
            for j in 1..h {
                self.v[[i, j]] += 2.0 * (p[[i, j]] - p[[i, j - 1]]);
            }
        }
    }

    pub fn div(&self) -> Array2<Float> {
        let div = Array::from_shape_fn(self.dim(), |(i, j)| {
            0.5 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]])
        });

        div
    }

    pub fn project2(&mut self, dt: Float, dx: Float, density: Float) {
        let div = Array::from_shape_fn(self.dim(), |(i, j)| {
            -0.5 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]]) / dx
        });

        let mut p = Array::zeros(div.dim());

        let scale = dt / (density * dx * dx);

        lin_solve(&mut p, &div, 1.0 * scale, 4.0 * scale);

        let l = dt / (density * dx);

        let (w, h) = self.dim();
        for i in 1..w {
            for j in 0..h {
                self.u[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i - 1, j]]);
            }
        }

        for i in 0..w {
            for j in 1..h {
                self.v[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i, j - 1]]);
            }
        }
    }

    pub fn project_variable_density(&mut self, dt: Float, dx: Float, density: &Array2<Float>) {
        let div = Array::from_shape_fn(self.dim(), |(i, j)| {
            -0.5 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]]) / dx
        });

        let mut p = Array::zeros(div.dim());

        let scale = dt / (dx * dx);

        lin_solve_variable_density(&mut p, &div, density, -1.0 * scale, 4.0 * scale);

        let l = dt / (dx);

        let (w, h) = self.dim();
        for i in 1..w {
            for j in 0..h {
                self.u[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i - 1, j]])
                    / (0.5 * (density[[i - 1, j]] + density[[i, j]]));
            }
        }

        for i in 0..w {
            for j in 1..h {
                self.v[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i, j - 1]])
                    / (0.5 * (density[[i, j - 1]] + density[[i, j]]));
            }
        }
    }

    pub fn project_variable_density_div_control(
        &mut self,
        dt: Float,
        dx: Float,
        density: &Array2<Float>,
        divergence: &Array2<Float>,
    ) {
        let div = Array::from_shape_fn(self.dim(), |(i, j)| {
            -0.5 * (self.u[[i + 1, j]] - self.u[[i, j]] + self.v[[i, j + 1]] - self.v[[i, j]]) / dx
                + 3.0 * divergence[[i, j]]
        });

        let mut p = Array::zeros(div.dim());

        let scale = dt / (dx * dx);

        lin_solve_variable_density(&mut p, &div, density, -1.0 * scale, 4.0 * scale);

        let l = dt / (dx);

        let (w, h) = self.dim();
        for i in 1..w {
            for j in 0..h {
                self.u[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i - 1, j]])
                    / (0.5 * (density[[i - 1, j]] + density[[i, j]]));
            }
        }

        for i in 0..w {
            for j in 1..h {
                self.v[[i, j]] -= 2.0 * l * (p[[i, j]] - p[[i, j - 1]])
                    / (0.5 * (density[[i, j - 1]] + density[[i, j]]));
            }
        }
    }

    pub fn diffuse(&mut self, a: Float) {
        let u0 = self.u.clone();
        let v0 = self.v.clone();
        lin_solve(&mut self.u, &u0, a, 1.0 + 4.0 * a);
        lin_solve(&mut self.v, &v0, a, 1.0 + 4.0 * a);
    }

    pub fn gauss_filter(&mut self, sigma2: Float, unit: Float) {
        self.u = gauss_filter(&self.u, sigma2, unit);
        self.v = gauss_filter(&self.v, sigma2, unit);
    }

    pub fn buoyancy(
        &mut self,
        s: &Array2<Float>,
        t: &Array2<Float>,
        alpha: Float,
        beta: Float,
        t_amb: Float,
        g: Float,
    ) {
        let (w, h) = self.dim();

        for i in 0..w {
            for j in 0..h + 1 {
                let s = {
                    let up = if j == 0 { 0.0 } else { s[[i, j - 1]] };

                    let down = if j == h { 0.0 } else { s[[i, j]] };

                    0.5 * (up + down)
                };

                let t = {
                    let up = if j == 0 { t_amb } else { t[[i, j - 1]] };

                    let down = if j == h { t_amb } else { t[[i, j]] };

                    0.5 * (up + down)
                };

                self.v[[i, j]] += (alpha * s - beta * (t - t_amb)) * g;
            }
        }
    }

    pub fn buoyancy2(&mut self, density: &Array2<Float>, density_amb: Float, g: Float) {
        let (w, h) = self.dim();

        let mut x: Float = 0.0;
        for i in 0..w {
            for j in 0..h + 1 {
                let d = {
                    let up = if j == 0 {
                        density_amb
                    } else {
                        density[[i, j - 1]]
                    };

                    let down = if j == h { density_amb } else { density[[i, j]] };

                    0.5 * (up + down)
                };

                self.v[[i, j]] += (d - density_amb) / d * g;
                x = x.max(((d - density_amb) / d * g).abs());
            }
        }

        dbg!(x);
    }
}

pub fn lin_solve_variable_density(
    x: &mut Array2<Float>,
    x0: &Array2<Float>,
    density: &Array2<Float>,
    a: Float,
    c: Float,
) {
    assert_eq!(x.dim(), x0.dim());
    assert_eq!(x.dim(), density.dim());

    // x.clone_from(x0);

    /*
    for _ in 0..150 {
        for i in 1..x.dim().0 - 1 {
            for j in 1..x.dim().1 - 1 {
                x[[i, j]] = (x0[[i, j]]
                    - (x[[i - 1, j]] / density[[i - 1, j]]
                        + x[[i + 1, j]] / density[[i + 1, j]]
                        + x[[i, j - 1]] / density[[i, j - 1]]
                        + x[[i, j + 1]] / density[[i, j + 1]])
                        * a)
                    / (c / density[[i, j]]);
            }
        }
    }
    */
    let (w, h) = x.dim();
    for _ in 0..150 {
        for i in 0..w {
            for j in 0..h {
                let mut t = 0.0;

                if i > 0 {
                    t = t + x[[i - 1, j]] / density[[i - 1, j]];
                }

                if i + 1 < w {
                    t = t + x[[i + 1, j]] / density[[i + 1, j]];
                }

                if j > 0 {
                    t = t + x[[i, j - 1]] / density[[i, j - 1]];
                }

                if j + 1 < h {
                    t = t + x[[i, j + 1]] / density[[i, j + 1]];
                }

                x[[i, j]] = (x0[[i, j]] - t * a) / (c / density[[i, j]]);
            }
        }
    }

    // let rev = linear::rev_density(x, density, a, c);
    // dbg!((rev - x0).iter().map(|f| f.abs()).sum::<Float>());
}

pub fn interpolate_linear<V: Vector>(q: &Array2<V>, ij: Vector2<Float>) -> V {
    let w = q.dim().0 - 2;
    let h = q.dim().1 - 2;

    let x = ij.x.max(0.5).min(w as Float + 0.5);
    let y = ij.y.max(0.5).min(h as Float + 0.5);

    let i0 = x as usize;
    let i1 = i0 + 1;

    let j0 = y as usize;
    let j1 = j0 + 1;

    let s1 = x - i0 as Float;
    let s0 = 1.0 - s1;

    let t1 = y - j0 as Float;
    let t0 = 1.0 - t1;

    (q[[i0, j0]] * t0 + q[[i0, j1]] * t1) * s0 + (q[[i1, j0]] * t0 + q[[i1, j1]] * t1) * s1
}

pub fn interpolate_bicubic<V: Vector>(q: &Array2<V>, ij: Vector2<Float>) -> V {
    let w = q.dim().0 - 2;
    let h = q.dim().1 - 2;

    let x = ij.x.max(1.5).min((w - 1) as Float + 0.5);
    let y = ij.y.max(1.5).min((h - 1) as Float + 0.5);

    let i = x as usize;
    let j = y as usize;

    let s = x - i as Float;
    let t = y - j as Float;

    let s_m1 = -1.0 / 3.0 * s + 0.5 * s * s - 1.0 / 6.0 * s * s * s;
    let s_0 = 1.0 - s * s + 0.5 * (s * s * s - s);
    let s_p1 = s + 0.5 * (s * s - s * s * s);
    let s_p2 = 1.0 / 6.0 * (s * s * s - s);

    let t_m1 = -1.0 / 3.0 * t + 0.5 * t * t - 1.0 / 6.0 * t * t * t;
    let t_0 = 1.0 - t * t + 0.5 * (t * t * t - t);
    let t_p1 = t + 0.5 * (t * t - t * t * t);
    let t_p2 = 1.0 / 6.0 * (t * t * t - t);

    let q_m1 = q[[i - 1, j - 1]] * s_m1
        + q[[i, j - 1]] * s_0
        + q[[i + 1, j - 1]] * s_p1
        + q[[i + 2, j - 1]] * s_p2;

    let q_0 = q[[i - 1, j]] * s_m1 + q[[i, j]] * s_0 + q[[i + 1, j]] * s_p1 + q[[i + 2, j]] * s_p2;

    let q_p1 = q[[i - 1, j + 1]] * s_m1
        + q[[i, j + 1]] * s_0
        + q[[i + 1, j + 1]] * s_p1
        + q[[i + 2, j + 1]] * s_p2;

    let q_p2 = q[[i - 1, j + 2]] * s_m1
        + q[[i, j + 2]] * s_0
        + q[[i + 1, j + 2]] * s_p1
        + q[[i + 2, j + 2]] * s_p2;

    q_m1 * t_m1 + q_0 * t_0 + q_p1 * t_p1 + q_p2 * t_p2
}

pub fn advect<V: Vector>(
    d: &mut Array2<V>,
    d0: &Array2<V>,
    uv: &Array2<Vector2<Float>>,
    weight: Float,
) {
    assert_eq!(d.dim(), d0.dim());
    assert_eq!(uv.dim(), d0.dim());

    let w = d.dim().0 - 2;
    let h = d.dim().1 - 2;

    for i in 1..w + 1 {
        for j in 1..h + 1 {
            let x0 = vec2(i as Float, j as Float);
            let k1 = uv[[i, j]];
            let k2 = interpolate_linear(uv, x0 - 0.5 * weight * k1);
            let k3 = interpolate_linear(uv, x0 - 0.75 * weight * k2);

            let v =
                x0 - 2.0 / 9.0 * weight * k1 - 3.0 / 9.0 * weight * k2 - 4.0 / 9.0 * weight * k3;

            d[[i, j]] = interpolate_linear(d0, v);
        }
    }
}

pub fn gauss_filter(x: &Array2<Float>, sigma2: Float, unit: Float) -> Array2<Float> {
    let cut_off = 0.1 * sigma2.sqrt();
    let left = 1.0 / (2.0 * Float::PI() * sigma2).sqrt();
    let coeff: Vec<Float> = (0..)
        .map(|i| left * (-(i as Float * unit).powi(2) / (2.0 * sigma2)).exp())
        .take_while(|&f| f > cut_off)
        .take(x.dim().0.max(x.dim().1))
        .collect();

    let coeff_sum_inv = 1.0 / (coeff.iter().sum::<Float>() * 2.0 - coeff[0]);

    let x1 = Array::from_shape_fn(x.dim(), |(i, j)| {
        let mut sum = coeff[0] * x[[i, j]];
        for c in 1..coeff.len() {
            if i >= c {
                sum += coeff[c] * x[[i - c, j]];
            }

            if i + c < x.dim().0 {
                sum += coeff[c] * x[[i + c, j]];
            }
        }
        sum * coeff_sum_inv
    });

    Array::from_shape_fn(x.dim(), |(i, j)| {
        let mut sum = coeff[0] * x1[[i, j]];
        for c in 1..coeff.len() {
            if j >= c {
                sum += coeff[c] * x1[[i, j - c]];
            }

            if j + c < x.dim().1 {
                sum += coeff[c] * x1[[i, j + c]];
            }
        }
        sum * coeff_sum_inv
    })
}

pub fn gauss_filter_amb(
    x: &Array2<Float>,
    sigma2: Float,
    unit: Float,
    amb: Float,
) -> Array2<Float> {
    let cut_off = 0.1 * sigma2.sqrt();
    let left = 1.0 / (2.0 * Float::PI() * sigma2).sqrt();
    let coeff: Vec<Float> = (0..)
        .map(|i| left * (-(i as Float * unit).powi(2) / (2.0 * sigma2)).exp())
        .take_while(|&f| f > cut_off)
        .take(x.dim().0.max(x.dim().1))
        .collect();

    let coeff_sum_inv = 1.0 / (coeff.iter().sum::<Float>() * 2.0 - coeff[0]);

    let x1 = Array::from_shape_fn(x.dim(), |(i, j)| {
        let mut sum = coeff[0] * x[[i, j]];
        for c in 1..coeff.len() {
            if i >= c {
                sum += coeff[c] * x[[i - c, j]];
            } else {
                sum += coeff[c] * amb;
            }

            if i + c < x.dim().0 {
                sum += coeff[c] * x[[i + c, j]];
            } else {
                sum += coeff[c] * amb;
            }
        }
        sum * coeff_sum_inv
    });

    Array::from_shape_fn(x.dim(), |(i, j)| {
        let mut sum = coeff[0] * x1[[i, j]];
        for c in 1..coeff.len() {
            if j >= c {
                sum += coeff[c] * x1[[i, j - c]];
            } else {
                sum += coeff[c] * amb;
            }

            if j + c < x.dim().1 {
                sum += coeff[c] * x1[[i, j + c]];
            } else {
                sum += coeff[c] * amb;
            }
        }
        sum * coeff_sum_inv
    })
}
