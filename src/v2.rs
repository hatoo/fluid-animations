use cgmath::{vec2, Vector2};
use ndarray::{Array, Array2};

use crate::{Float, Vector};

pub struct Mac {
    u: Array2<Float>,
    v: Array2<Float>,
}

impl Mac {
    pub fn new(u: Array2<Float>, v: Array2<Float>) -> Self {
        let u_dim = u.dim();
        let v_dim = v.dim();

        assert_eq!((u_dim.0 - 1, u_dim.1), (v_dim.0, v_dim.1 - 1));

        Self { u, v }
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

    pub fn self_advect(&mut self, dt: Float) {
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

        let v_uv = Array::from_shape_fn(self.u.dim(), |(i, j)| {
            let v = self.v[[i, j]];
            let u = if j == 0 {
                0.5 * (self.u[[i, j]] + self.u[[i + 1, j]])
            } else if i == self.u.dim().0 - 1 {
                0.5 * (self.u[[i, j - 1]] + self.u[[i + 1, j - 1]])
            } else {
                0.25 * (self.u[[i, j]]
                    + self.u[[i + 1, j]]
                    + self.u[[i, j - 1]]
                    + self.u[[i + 1, j - 1]])
            };
            vec2(u, v)
        });

        let u0 = self.u.clone();
        let v0 = self.v.clone();
        advect(&mut self.u, &u0, &u_uv, dt);
        advect(&mut self.v, &v0, &v_uv, dt);
    }
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
    dt: Float,
) {
    assert_eq!(d.dim(), d0.dim());
    assert_eq!(uv.dim(), d0.dim());

    let w = d.dim().0 - 2;
    let h = d.dim().1 - 2;

    let dt0 = dt * ((w * h) as Float).sqrt();

    for i in 1..w + 1 {
        for j in 1..h + 1 {
            let x0 = vec2(i as Float, j as Float);
            let k1 = uv[[i, j]];
            let k2 = interpolate_linear(uv, x0 - 0.5 * dt0 * k1);
            let k3 = interpolate_linear(uv, x0 - 0.75 * dt0 * k2);

            let v = x0 - 2.0 / 9.0 * dt0 * k1 - 3.0 / 9.0 * dt0 * k2 - 4.0 / 9.0 * dt0 * k3;

            d[[i, j]] = interpolate_linear(d0, v);
        }
    }
}
