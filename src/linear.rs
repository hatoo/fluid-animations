use ndarray::{Array, Array2, Zip};

use crate::Float;

pub fn lin_solve(x: &mut Array2<Float>, x0: &Array2<Float>, a: Float, c: Float) {
    assert_eq!(x.dim(), x0.dim());

    let (w, h) = x.dim();

    for _ in 0..150 {
        for i in 0..w {
            for j in 0..h {
                let mut t = 0.0;

                if i > 0 {
                    t = t + x[[i - 1, j]];
                }

                if i + 1 < w {
                    t = t + x[[i + 1, j]];
                }

                if j > 0 {
                    t = t + x[[i, j - 1]];
                }

                if j + 1 < h {
                    t = t + x[[i, j + 1]];
                }

                x[[i, j]] = (x0[[i, j]] - t * a) / c;
            }
        }
    }
}
fn apply_a(out: &mut Array2<Float>, ans: &Array2<Float>, a: Float, c: Float) {
    let (w, h) = ans.dim();

    out.indexed_iter_mut().for_each(|((i, j), e)| {
        *e = c * ans[[i, j]];

        if i > 0 {
            *e += a * ans[[i - 1, j]];
        }

        if i + 1 < w {
            *e += a * ans[[i + 1, j]];
        }

        if j > 0 {
            *e += a * ans[[i, j - 1]];
        }

        if j + 1 < h {
            *e += a * ans[[i, j + 1]];
        }
    })
}

fn dot_product(a: &Array2<Float>, b: &Array2<Float>) -> Float {
    a.indexed_iter()
        .map(|((i, j), e)| *e * b[[i, j]])
        .sum::<Float>()
}

fn apply_precon(z: &mut Array2<Float>, r: &Array2<Float>, a: Float, c: Float) {
    let tuning = 0.97;
    let sigma = 0.25;

    let (w, h) = z.dim();
    let mut precon = Array::from_elem(z.dim(), 0.0 as Float);

    for i in 0..w {
        for j in 0..h {
            let e = c
                - (a * precon.get([i - 1, j]).copied().unwrap_or_default()).powi(2)
                - (a * precon.get([i, j - 1]).copied().unwrap_or_default()).powi(2)
                - tuning
                    * (a * a * precon.get([i - 1, j]).copied().unwrap_or_default().powi(2)
                        + a * a * precon.get([i, j - 1]).copied().unwrap_or_default().powi(2));

            let e = if e < sigma * c { c } else { e };
            precon[[i, j]] = 1.0 / e.sqrt();

            // assert!(precon[[i, j]].is_finite());
        }
    }

    // dbg!(precon.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
    // dbg!(r.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));

    let mut q = Array::zeros(z.dim());

    for i in 0..w {
        for j in 0..h {
            let t: Float = r[[i, j]]
                - a * precon.get([i - 1, j]).copied().unwrap_or_default()
                    * q.get([i - 1, j]).copied().unwrap_or_default()
                - a * precon.get([i, j - 1]).copied().unwrap_or_default()
                    * q.get([i, j - 1]).copied().unwrap_or_default();
            q[[i, j]] = t * precon[[i, j]];

            // dbg!(q[[i, j]]);
            // assert!(q[[i, j]].is_finite());
        }
    }
    // dbg!(q.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
    for i in (0..w).rev() {
        for j in (0..h).rev() {
            let t = q[[i, j]]
                - a * precon[[i, j]] * z.get([i + 1, j]).copied().unwrap_or_default()
                - a * precon[[i, j]] * z.get([i, j + 1]).copied().unwrap_or_default();

            z[[i, j]] = t * precon[[i, j]];
            // assert!(z[[i, j]].is_finite());
        }
    }
    // dbg!(z.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
}

pub fn lin_solve_pcg(p: &mut Array2<Float>, d: &Array2<Float>, a: Float, c: Float) {
    assert_eq!(p.dim(), d.dim());

    if d.iter().fold(0.0 as Float, |a, &b| a.max(b)) < 1e-6 {
        return;
    }

    let tol = 1e-6 * d.iter().fold(0.0 as Float, |a, &b| a.max(b));

    let mut r = d.clone();
    let mut z = Array::zeros(p.dim());
    apply_precon(&mut z, &r, a, c);
    let mut s = z.clone();

    let mut sigma = dot_product(&z, &r);

    for _ in 0..200 {
        apply_a(&mut z, &s, a, c);
        let alpha = sigma / dot_product(&z, &s);

        Zip::from(&mut *p).and(&s).for_each(|a, &b| {
            *a += alpha * b;
        });

        Zip::from(&mut r).and(&z).for_each(|a, &b| {
            *a -= alpha * b;
        });

        if r.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())) < tol {
            dbg!("early return");
            return;
        }

        apply_precon(&mut z, &r, a, c);

        let sigma_new = dot_product(&z, &r);
        let beta = sigma_new / sigma;

        Zip::from(&mut s).and(&z).for_each(|a, &b| {
            *a = b + beta * *a;
        });

        sigma = sigma_new;
    }
}

//

pub fn apply_a2(
    out: &mut Array2<Float>,
    ans: &Array2<Float>,
    a: &Array2<Float>,
    c: &Array2<Float>,
) {
    let (w, h) = ans.dim();

    out.indexed_iter_mut().for_each(|((i, j), e)| {
        *e = c[[i, j]] * ans[[i, j]];

        if i > 0 {
            *e += a[[i - 1, j]] * ans[[i - 1, j]];
        }

        if i + 1 < w {
            *e += a[[i + 1, j]] * ans[[i + 1, j]];
        }

        if j > 0 {
            *e += a[[i, j - 1]] * ans[[i, j - 1]];
        }

        if j + 1 < h {
            *e += a[[i, j + 1]] * ans[[i, j + 1]];
        }
    })
}

fn apply_precon2(z: &mut Array2<Float>, r: &Array2<Float>, a: &Array2<Float>, c: &Array2<Float>) {
    let tuning = 0.97;
    let sigma = 0.25;

    let (w, h) = z.dim();
    let mut precon = Array::from_elem(z.dim(), 0.0 as Float);

    for i in 0..w {
        for j in 0..h {
            let e = c[[i, j]]
                - (a.get([i - 1, j]).copied().unwrap_or_default()
                    * precon.get([i - 1, j]).copied().unwrap_or_default())
                .powi(2)
                - (a.get([i, j - 1]).copied().unwrap_or_default()
                    * precon.get([i, j - 1]).copied().unwrap_or_default())
                .powi(2)
                - tuning
                    * (a.get([i - 1, j]).copied().unwrap_or_default()
                        * a.get([i - 1, j]).copied().unwrap_or_default()
                        * precon.get([i - 1, j]).copied().unwrap_or_default().powi(2)
                        + a.get([i, j - 1]).copied().unwrap_or_default()
                            * a.get([i, j - 1]).copied().unwrap_or_default()
                            * precon.get([i, j - 1]).copied().unwrap_or_default().powi(2));

            let e = if e < sigma * c[[i, j]] { c[[i, j]] } else { e };
            precon[[i, j]] = 1.0 / e.sqrt();

            // assert!(precon[[i, j]].is_finite());
        }
    }

    // dbg!(precon.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
    // dbg!(r.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));

    let mut q = Array::zeros(z.dim());

    for i in 0..w {
        for j in 0..h {
            let t: Float = r[[i, j]]
                - a.get([i - 1, j]).copied().unwrap_or_default()
                    * precon.get([i - 1, j]).copied().unwrap_or_default()
                    * q.get([i - 1, j]).copied().unwrap_or_default()
                - a.get([i, j - 1]).copied().unwrap_or_default()
                    * precon.get([i, j - 1]).copied().unwrap_or_default()
                    * q.get([i, j - 1]).copied().unwrap_or_default();
            q[[i, j]] = t * precon[[i, j]];

            // dbg!(q[[i, j]]);
            // assert!(q[[i, j]].is_finite());
        }
    }
    // dbg!(q.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
    for i in (0..w).rev() {
        for j in (0..h).rev() {
            let t = q[[i, j]]
                - a[[i, j]] * precon[[i, j]] * z.get([i + 1, j]).copied().unwrap_or_default()
                - a[[i, j]] * precon[[i, j]] * z.get([i, j + 1]).copied().unwrap_or_default();

            z[[i, j]] = t * precon[[i, j]];
            // assert!(z[[i, j]].is_finite());
        }
    }
    // dbg!(z.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())));
}

pub fn lin_solve_pcg2(
    p: &mut Array2<Float>,
    d: &Array2<Float>,
    a: &Array2<Float>,
    c: &Array2<Float>,
) {
    assert_eq!(p.dim(), d.dim());

    if d.iter().fold(0.0 as Float, |a, &b| a.max(b)) < 1e-6 {
        return;
    }

    let tol = 1e-6 * d.iter().fold(0.0 as Float, |a, &b| a.max(b));

    let mut r = d.clone();
    let mut z = Array::zeros(p.dim());
    apply_precon2(&mut z, &r, a, c);
    let mut s = z.clone();

    let mut sigma = dot_product(&z, &r);

    for _ in 0..200 {
        apply_a2(&mut z, &s, a, c);
        let alpha = sigma / dot_product(&z, &s);

        Zip::from(&mut *p).and(&s).for_each(|a, &b| {
            *a += alpha * b;
        });

        Zip::from(&mut r).and(&z).for_each(|a, &b| {
            *a -= alpha * b;
        });

        if r.iter().fold(0.0 as Float, |a, &b| a.max(b.abs())) < tol {
            dbg!("early return");
            return;
        }

        apply_precon2(&mut z, &r, a, c);

        let sigma_new = dot_product(&z, &r);
        let beta = sigma_new / sigma;

        Zip::from(&mut s).and(&z).for_each(|a, &b| {
            *a = b + beta * *a;
        });

        sigma = sigma_new;
    }
}

pub fn rev(ans: &Array2<Float>, a: Float, c: Float) -> Array2<Float> {
    let (w, h) = ans.dim();
    Array::from_shape_fn(ans.dim(), |(i, j)| {
        let mut t = c * ans[[i, j]];

        if i > 0 {
            t += a * ans[[i - 1, j]];
        }

        if i + 1 < w {
            t += a * ans[[i + 1, j]];
        }

        if j > 0 {
            t += a * ans[[i, j - 1]];
        }

        if j + 1 < h {
            t += a * ans[[i, j + 1]];
        }
        t
    })
}

pub fn rev_density(
    ans: &Array2<Float>,
    density: &Array2<Float>,
    a: Float,
    c: Float,
) -> Array2<Float> {
    let (w, h) = ans.dim();
    Array::from_shape_fn(ans.dim(), |(i, j)| {
        let mut t = c * ans[[i, j]] / density[[i, j]];

        if i > 0 {
            t += a * ans[[i - 1, j]] / density[[i - 1, j]];
        }

        if i + 1 < w {
            t += a * ans[[i + 1, j]] / density[[i + 1, j]];
        }

        if j > 0 {
            t += a * ans[[i, j - 1]] / density[[i, j - 1]];
        }

        if j + 1 < h {
            t += a * ans[[i, j + 1]] / density[[i, j + 1]];
        }
        t
    })
}

#[cfg(test)]
mod test {
    use ndarray::{array, Array, Array2};
    use ndarray_linalg::{assert_close_l1, solve::Solve};

    use crate::{
        linear::{lin_solve, lin_solve_pcg},
        Float,
    };

    #[allow(dead_code)]
    fn lin_solve_by_linalg(x0: &Array2<Float>, a: Float, c: Float) -> Array2<Float> {
        let (w, h) = x0.dim();
        let m = get_matrix(w, h, a, c);

        let xv = x0.clone().into_shape(x0.len()).unwrap();

        let ans = m.solve(&xv).unwrap();

        ans.into_shape((w, h)).unwrap()
    }

    #[allow(dead_code)]
    fn rev(ans: &Array2<Float>, a: Float, c: Float) -> Array2<Float> {
        let (w, h) = ans.dim();
        Array::from_shape_fn(ans.dim(), |(i, j)| {
            let mut t = c * ans[[i, j]];

            if i > 0 {
                t += a * ans[[i - 1, j]];
            }

            if i + 1 < w {
                t += a * ans[[i + 1, j]];
            }

            if j > 0 {
                t += a * ans[[i, j - 1]];
            }

            if j + 1 < h {
                t += a * ans[[i, j + 1]];
            }
            t
        })
    }

    fn get_matrix(w: usize, h: usize, a: Float, c: Float) -> Array2<Float> {
        let mut m = Array::zeros((w * h, w * h));
        for i in 0..w {
            for j in 0..h {
                let r = i * h + j;
                m[[r, i * h + j]] = c;

                if i > 0 {
                    m[[r, (i - 1) * h + j]] = a;
                }

                if i + 1 < w {
                    m[[r, (i + 1) * h + j]] = a;
                }

                if j > 0 {
                    m[[r, i * h + (j - 1)]] = a;
                }

                if j + 1 < h {
                    m[[r, i * h + (j + 1)]] = a;
                }
            }
        }
        m
    }

    #[test]
    fn lin_sovle() {
        let b = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],];
        let mut t = Array::zeros(b.dim());

        lin_solve(&mut t, &b, -1.0, 4.0);
        // let ans = lin_solve_by_linalg(&b, -1.0, 4.0);

        assert_close_l1!(&rev(&t, -1.0, 4.0), &b, 1e-5);
    }

    #[test]
    fn lin_sovle_pcg() {
        let b = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],];
        let mut t = Array::zeros(b.dim());

        lin_solve_pcg(&mut t, &b, -1.0, 4.0);
        // let ans = lin_solve_by_linalg(&b, -1.0, 4.0);

        assert_close_l1!(&rev(&t, -1.0, 4.0), &b, 1e-5);
    }
}
