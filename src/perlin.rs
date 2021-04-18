use itertools::Itertools;
use ndarray::prelude::*;

fn c(x: f64) -> f64 {
    1. - 3. * x * x + 2. * x.abs().powf(3.)
}

fn xorshift64(x: u64) -> u64 {
    let x = x ^ (x << 13);
    let x = x ^ (x >> 7);
    let x = x ^ (x << 17);
    x
}

struct Mute<T> {
    x: Vec<T>,
    size: usize,
    state: usize,
    max_state: usize,
}

impl<T> Mute<T> {
    fn new(values: Vec<T>, n: usize) -> Self {
        let max = values.len().pow(n as u32);
        Self {
            x: values,
            size: n,
            state: 0,
            max_state: max,
        }
    }
}

impl<T> Iterator for Mute<T>
where
    T: Copy,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.state >= self.max_state {
            None
        } else {
            let mut s = self.state;
            let mut vs = Vec::<T>::new();
            for _ in 0..self.size {
                vs.push(self.x[s % self.x.len()]);
                s /= self.x.len();
            }
            self.state += 1;
            Some(vs)
        }
    }
}

pub struct PerlinNoise {
    seed: usize,
    grid_size: usize,
}

impl PerlinNoise {
    pub fn new(seed: usize, grid_size: usize) -> Self {
        if grid_size == 0 {
            panic!("grid_size is expected larger than 0 but found 0");
        }
        PerlinNoise {
            seed: seed,
            grid_size: grid_size,
        }
    }

    fn grad(&self, x: &ArrayView1<f64>) -> Array1<f64> {
        let mut seed = self.seed;
        for &x in x.iter() {
            seed = seed
                .wrapping_mul(1 + self.grid_size)
                .wrapping_add(x as usize);
        }
        let mut rand = xorshift64(seed as u64);
        let mut y = Array1::<f64>::zeros(x.shape()[0]);
        for y in y.iter_mut() {
            rand = 0x3ff << 52 | (xorshift64(rand) & 0xFFFFFFFFFFFFF);
            let r = f64::from_bits(rand);
            debug_assert!(r >= 1. && r < 2.);
            //*y = 2. * (rand as f64 / f64::powf(2., 64.)) - 1.;
            *y = 2. * r - 3.;
        }
        y
    }

    pub fn get(&self, pos: &ArrayView1<f64>) -> f64 {
        let imax = pos.fold(1. as f64, |a, &b| a.max(b));
        let imin = pos.fold(0. as f64, |a, &b| a.min(b));
        if imax > 1. {
            panic!("Expected between 0 and 1 but found {}", imax);
        }
        if imin > 0. {
            panic!("Expected between 0 and 1 but found {}", imin);
        }
        self._get(pos)
    }

    fn _get(&self, pos: &ArrayView1<f64>) -> f64 {
        let pos = pos * (1. + self.grid_size as f64);
        let x0 = pos.map(|&x| x.floor());
        let mut v = 0.;
        for q in Mute::<f64>::new(vec![0., 1.], pos.shape()[0]) {
            let q = Array1::<f64>::from(q) + &x0;
            let dx = &pos - &q;
            v += self.grad(&q.view()).dot(&dx) * dx.map(|&x| c(x)).fold(1., |a, &b| a * b);
        }
        v
    }

    pub fn gen1d(&self, size: usize) -> Array1<f64> {
        let x = Array1::<f64>::linspace(0., 1., size);
        x.map(|&v| self._get(&aview1(&[v])))
    }

    pub fn gen2d(&self, nx: usize, ny: usize) -> Array2<f64> {
        let n = nx.max(ny) as f64;
        let x = Array1::<f64>::linspace(0., nx as f64 / n, nx);
        let y = Array1::<f64>::linspace(0., ny as f64 / n, ny);
        let v: Array1<_> = x
            .iter()
            .cartesian_product(y.iter())
            .map(|(&x, &y)| self._get(&aview1(&[x, y])))
            .collect();
        v.into_shape((nx, ny)).unwrap()
    }

    pub fn gen3d(&self, nx: usize, ny: usize, nz: usize) -> Array3<f64> {
        let n = nx.max(ny).max(nz) as f64;
        let x = Array1::<f64>::linspace(0., nx as f64 / n, nx);
        let y = Array1::<f64>::linspace(0., ny as f64 / n, ny);
        let z = Array1::<f64>::linspace(0., nz as f64 / n, nz);
        let v: Array1<_> = x
            .iter()
            .cartesian_product(y.iter().cartesian_product(z.iter()))
            .map(|(&x, (&y, &z))| self._get(&aview1(&[x, y, z])))
            .collect();
        v.into_shape((nx, ny, nz)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mute() {
        let mut iter = Mute::<f64>::new(vec![0., 1.], 2);
        assert_eq!(iter.next(), Some(vec![0., 0.]));
        assert_eq!(iter.next(), Some(vec![1., 0.]));
        assert_eq!(iter.next(), Some(vec![0., 1.]));
        assert_eq!(iter.next(), Some(vec![1., 1.]));
    }

    #[test]
    fn cartesian_product() {
        let mut iter = (0..3).cartesian_product(0..2);
        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next(), Some((0, 1)));
        assert_eq!(iter.next(), Some((1, 0)));
        assert_eq!(iter.next(), Some((1, 1)));
        assert_eq!(iter.next(), Some((2, 0)));
        assert_eq!(iter.next(), Some((2, 1)));
    }

    #[test]
    fn perlin1d() {
        PerlinNoise::new(2000000000, 8).gen1d(200);
    }

    #[test]
    fn perlin2d() {
        PerlinNoise::new(2000000000, 8).gen2d(20, 20);
    }

    #[test]
    fn perlin3d() {
        PerlinNoise::new(2000000000, 2).gen3d(10, 10, 10);
    }
}
