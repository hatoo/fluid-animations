use ndarray::{Array, Array2};

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

pub fn lin_solve_pcg(x: &mut Array2<Float>, x0: &Array2<Float>, a: Float, c: Float) {
    todo!()
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

    use crate::{linear::lin_solve, Float};

    fn lin_solve_by_linalg(x0: &Array2<Float>, a: Float, c: Float) -> Array2<Float> {
        let (w, h) = x0.dim();
        let m = get_matrix(w, h, a, c);

        let xv = x0.clone().into_shape(x0.len()).unwrap();

        let ans = m.solve(&xv).unwrap();

        ans.into_shape((w, h)).unwrap()
    }

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
}
