mod perlin;
mod poisson_disk;

pub use perlin::*;

use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn rust_ext(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "perlin1d")]
    fn perlin1d_py<'py>(
        py: Python<'py>,
        seed: usize,
        grid_size: usize,
        image_size: usize,
    ) -> &'py PyArray1<f64> {
        perlin::PerlinNoise::new(seed, grid_size)
            .gen1d(image_size)
            .into_pyarray(py)
    }

    #[pyfn(m, "perlin2d")]
    fn perlin2d_py<'py>(
        py: Python<'py>,
        seed: usize,
        grid_size: usize,
        image_size: (usize, usize),
    ) -> &'py PyArray2<f64> {
        perlin::PerlinNoise::new(seed, grid_size)
            .gen2d(image_size.0, image_size.1)
            .into_pyarray(py)
    }

    #[pyfn(m, "perlin3d")]
    fn perlin3d_py<'py>(
        py: Python<'py>,
        seed: usize,
        grid_size: usize,
        image_size: (usize, usize, usize),
    ) -> &'py PyArray3<f64> {
        perlin::PerlinNoise::new(seed, grid_size)
            .gen3d(image_size.0, image_size.1, image_size.2)
            .into_pyarray(py)
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
